from pathlib import Path
import sys
import os
import random
path_root = Path(__file__).parents[3]
sys.path.append(str(path_root))

import torch 
import torch.nn as nn
import torch.distributed as dist
from flash.core.optimizers import LAMB

from ML.gc import GlobalContext
gc = GlobalContext("/work/ta127/ta127/chrisrae/chris-ml-intern/ML/BERT_Large/Torch/config.yaml")
from ML.BERT_Large.Torch.model.BERT import get_bertlarge
from ML.BERT_Large.Torch.data.data_loader import get_pretrian_dataloader, get_eval_dataset
from ML.BERT_Large.Torch.lr_scheduler.schedulers import LinearWarmupPolyDecayScheduler, LinearWarmUpScheduler
from ML_HPC.CosmoFlow.Torch.lr_schedule.scheduler import CosmoLRScheduler

def per_worker_files(files):
    return [files[i::gc.world_size] for i in range(gc.world_size)][gc.rank]


def calc_mlm_acc(outputs, masked_lm_labels, dense_seq_output=False):
    prediction_scores = outputs
    masked_lm_labels_flat = masked_lm_labels.view(-1)
    mlm_labels = masked_lm_labels_flat[masked_lm_labels_flat != -100]
    if not dense_seq_output:
        prediction_scores_flat = prediction_scores.view(-1, prediction_scores.shape[-1])
        mlm_predictions_scores = prediction_scores_flat[masked_lm_labels_flat != -100]
        mlm_predictions = mlm_predictions_scores.argmax(dim=-1)
    else:
        mlm_predictions = prediction_scores.argmax(dim=-1)

    num_masked = mlm_labels.numel()
    mlm_acc = (mlm_predictions == mlm_labels).sum(dtype=torch.float) / num_masked

    return mlm_acc, num_masked


def run_eval(model,eval_dataloader,):
    model.eval()
    total_eval_loss, total_eval_mlm_acc = 0.0, 0.0
    total_masked = 0
    with torch.no_grad():
        for batch in eval_dataloader:
            input_ids,segment_ids, input_mask, masked_lm_labels, next_sentence_labels = batch
            outputs, _ = model(
                input_ids=input_ids,
                token_type_ids=segment_ids,
                attention_mask=input_mask,
                labels=masked_lm_labels,
                next_sentence_label=None,
            )
            mlm_acc, num_masked = calc_mlm_acc(outputs, masked_lm_labels)
            total_eval_loss += outputs.loss.item() * num_masked
            total_eval_mlm_acc += mlm_acc * num_masked
            total_masked += num_masked
    model.train()
    total_masked = torch.tensor(total_masked, device=gc.device, dtype=torch.int64)
    total_eval_loss = torch.tensor(total_eval_loss, device=gc.device, dtype=torch.float64)
    if torch.distributed.is_initialized():
        # Collect total scores from all ranks
        dist.all_reduce(total_eval_mlm_acc, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_eval_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_masked, op=dist.ReduceOp.SUM)

    # Average by number of examples
    total_eval_mlm_acc /= total_masked
    total_eval_loss /= total_masked

    return total_eval_loss, total_eval_mlm_acc


def main():
    torch.manual_seed(0)
    random.seed(0)
    if dist.is_mpi_available():
        backend = "mpi"
    elif gc.device == "cuda":
        backend = "nccl"
    else:
        backend = "gloo"
    dist.init_process_group(backend)

    if gc.device == "cuda":
        local_rank = os.environ["LOCAL_RANK"]
        torch.cuda.set_device("cuda:" + local_rank)

    model = get_bertlarge().to(gc.device)
    if gc.world_size > 1:
        model = nn.parallel.DistributedDataParallel(model)
        
    if gc["opt"]["name"].upper() == "LAMB":
        opt = LAMB(model.parameters(), lr=gc["lr_schedule"]["base_lr"], betas=gc["opt"]["betas"], weight_decay=gc["opt"]["wight_decay"])
    else:
        raise NameError(f"Optimiser {gc['opt']['name']} not supported please use LAMB")
    
    lr_scheduler = LinearWarmupPolyDecayScheduler(opt, 
                                                  start_warmup_steps=gc["lr_schedule"]["start_warmup_steps"],
                                                  warmup_steps=gc["lr_schedule"]["warmup_steps"],
                                                  total_steps=gc["lr_schedule"]["total_steps"])
    
    """
    --- Basic Idea ---
    for e in Epoch:
        for x, y in data:
            loss = model.forward(x, y)
            loss.backward()
            opt.step()
            lr.step()
        
        mlm = val(model, val_data)
        if mlm >= 0.72:
            exit()
    """
    data_dir_files = os.listdir(gc["data"]["data_dir"], "hdf5")
    for file in data_dir_files:
        if "training_" in file:
            train_data_dir = file
            num_shards = int(file.replace("training_", ""))
    if num_shards % gc["data"]["global_batch_size"] != 0:
        raise NotImplementedError("If the number of shards is not devisable the remainder of files wont be used in the epoch")
    
    training_path = os.path.join(gc["data"]["data_dir"], "hdf5", train_data_dir, train_data_dir + "_shards_varlength")

    files = list([os.path.join(training_path, i) for i in os.listdir(training_path)])
    files.sort()
    my_files = per_worker_files(files)

    stop_training = False
    epoch = 0

    eval_dataloader = get_eval_dataset()

    while True:
        for file in my_files:
            dataloader = get_pretrian_dataloader(file)
            for batch in dataloader:
                opt.zero_grad()
                input_ids, token_type_ids, attention_mask, masked_lm_labels, next_sequence_labels = batch
                loss = model.forward(input_ids, token_type_ids, attention_mask, masked_lm_labels, next_sequence_labels)
                loss.backward()

                if hasattr(opt, "clip_grad_norm_"):
                    ggnorm = opt.clip_grad_norm_(1.0)
                else:
                    ggnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                opt.step()
                lr_scheduler.step()

        eval_loss, eval_accuracy = run_eval(model, eval_dataloader)

        stop_training = eval_accuracy >= gc["training"]["target_mlm_accuracy"]
        epoch+=1
        if epoch >= gc["data"]["n_epochs"] or stop_training:
            break

        random.shuffle(files)
        my_files = per_worker_files(files)

