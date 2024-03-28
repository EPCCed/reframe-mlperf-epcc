from email.policy import default
import os
import sys


path_root = "/".join(sys.argv[0].split("/")[:-4])
sys.path.append(str(path_root))
import click
import time

import torch
import torch.distributed as dist
from torchmetrics.classification import Accuracy
from tqdm import tqdm


from ML.gc import GlobalContext
gc = GlobalContext()
import ML.ResNet50.Torch.data.data_loader as dl
from ML.ResNet50.Torch.opt import Lars as LARS
from ML.ResNet50.Torch.model.ResNet import ResNet50

def train_step(x, y, model, loss_fn, opt, batch_idx):
    if (batch_idx+1)% gc["data"]["gradient_accumulation_freq"] != 0:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            with model.no_sync():
                logits = model(x)
                loss = loss_fn(logits, y)/gc["data"]["gradient_accumulation_freq"]
                #metric_tracker.update(logits, y)
                loss.backward()
        else:
            logits = model(x)
            loss = loss_fn(logits, y)/gc["data"]["gradient_accumulation_freq"]
            #metric_tracker.update(logits, y)
            loss.backward()
    else:
        logits = model(x)
        loss = loss_fn(logits, y)/gc["data"]["gradient_accumulation_freq"]
        #metric_tracker.update(logits, y)
        loss.backward()
        opt.step()
        opt.zero_grad()
    return loss


def valid_step(x, y, model, loss_fn, metric_tracker):
    with torch.no_grad():
        logits = model(x)
        loss = loss_fn(logits, y)
        metric_tracker(logits, y)
        return loss

def get_comm_time(prof: torch.profiler.profile):
    total_time = 0
    if prof is None:
        return total_time
    backend = "mpi:" if dist.get_backend() == "mpi" else "nccl:"
    for event in list(prof.key_averages()):
        if backend in event.key:
            total_time += event.cpu_time_total * 1e-6
            total_time += event.cuda_time_total * 1e-6
    return total_time


@click.command()
@click.option("--device", "-d", default="", show_default=True, type=str, help="The device type to run the benchmark on (cpu|gpu|cuda). If not provided will default to config.yaml")
@click.option("--config", "-c", default=os.path.join(os.getcwd(), "config.yaml"), show_default=True, type=str, help="Path to config.yaml. If not provided will default to config.yaml in the cwd")
@click.option("--data-dir", default=None, show_default=True, type=str, help="Path To DeepCAM dataset. If not provided will deafault to what is provided in the config.yaml")
@click.option("--global_batchsize", "-gbs", default=None, show_default=True, type=int, help="The Global Batchsize")
@click.option("--local_batchsize", "-lbs", default=None, show_default=True, type=int, help="The Local Batchsize, Leave as 0 to use the Global Batchsize")
@click.option("--t_subset_size", default=None, show_default=True, type=int, help="Size of the Training Subset, dont call to use full dataset")
@click.option("--v_subset_size", default=None, show_default=True, type=int, help="Size of the Validation Subset, dont call to use full dataset")
def main(device, config, data_dir, global_batchsize, local_batchsize, t_subset_size, v_subset_size):
    if config:
        gc.update_config(config)
    if device.lower() in ('cpu', "gpu", "cuda"):
        gc["device"] = device.lower()
    if data_dir:
        gc["data"]["data_dir"] = data_dir
    if global_batchsize is not None:
        gc["data"]["global_batch_size"] = global_batchsize
    if local_batchsize is not None:
        gc["data"]["local_batch_size"]= local_batchsize
    if t_subset_size is not None:
        gc["data"]["train_subset"] = t_subset_size
    if v_subset_size is not None:
        gc["data"]["val_subset"] = v_subset_size
    
    torch.manual_seed(1)
    gc.init_dist()
    if gc.device == "cuda":
        torch.cuda.set_device("cuda:" + str(gc.local_rank))

    
    gc.start_init()
    gc.log_seed(1)

    train_data = dl.get_train_dataloader()
    val_data = dl.get_val_dataloader()
    gc.log_resnet()
    if gc.rank == -1:  # change to -1 to turn off 0 to turn on
        train_data = tqdm(train_data, unit="images", unit_scale=(gc["data"]["global_batch_size"] // gc.world_size)//gc["data"]["gradient_accumulation_freq"]) 
    
    model = ResNet50(num_classes=1000).to(gc.device)
    if gc.world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(model)
        
        if gc.device == "cpu":
            pass
        else:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if gc["opt"]["name"].upper() == "SGD":
        opt = torch.optim.SGD(
            model.parameters(),
            lr=gc["lr_schedule"]["base_lr"],
            momentum=gc["opt"]["momentum"],
            weight_decay=gc["opt"]["weight_decay"],
        )
    elif gc["opt"]["name"].upper() == "LARS":
        opt = LARS(
            model.parameters(),
            lr=gc["lr_schedule"]["base_lr"],
            momentum=gc["opt"]["momentum"],
            weight_decay=gc["opt"]["weight_decay"],
        )
    else:
        raise ValueError(
            f"Optimiser {gc['opt']['name']} not supported please use SGD|LARS"
        )

    scheduler = torch.optim.lr_scheduler.PolynomialLR(
        opt, total_iters=gc["data"]["n_epochs"], power=gc["lr_schedule"]["poly_power"]
    )

    loss_fn = torch.nn.CrossEntropyLoss()

    val_metric = Accuracy(task="multiclass", num_classes=1000)

    val_metric.to(gc.device)

    gc.stop_init()

    model.eval()
    loss_fn.eval()
    sample_x = torch.ones(1, 3, 244, 244, dtype=torch.float32).to(gc.device)
    sample_y = torch.randint(1, 1000, (1,), dtype=torch.int64).to(gc.device)
    initial_loss = loss_fn.forward(model.forward(sample_x), sample_y)
    model.train()
    loss_fn.train()
    
    gc.start_run()
    
    model.train()

    E = 1
    
    while True:
        start = time.time()
        gc.start_epoch(metadata={"epoch_num": E})
        total_io_time = 0
        power_draw = []
        gpu_utilization = []
        with gc.profiler(f"Epoch: {E}") as prof:
            start_io = time.time_ns()
            for i, data in enumerate(train_data):
                x, y = data
                x, y = x.to(gc.device), y.to(gc.device)
                total_io_time += time.time_ns() - start_io
                loss = train_step(x, y, model, loss_fn, opt, i)
                power_draw.append(gc.gpu_power)
                gpu_utilization.append(gc.gpu_util)
                torch.cuda.synchronize()
                start_io = time.time_ns()
        total_io_time *= 1e-9

        total_time = time.time()-start
        total_time = torch.tensor(total_time).to(gc.device)
        avg_power_draw = torch.mean(torch.tensor(power_draw, dtype=torch.float64)).to(gc.device)
        avg_gpu_util = torch.mean(torch.tensor(gpu_utilization, dtype=torch.float64)).to(gc.device)
        dist.all_reduce(avg_gpu_util)
        dist.all_reduce(avg_power_draw, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_time)
        total_time /= gc.world_size
        if gc.rank == 0:
            if E == 1:
                print(f"Change In Train Loss at Epoch: {initial_loss - loss}")
                
            print(f"Train Loss at Epoch {E}: {loss}")
            dataset_size = gc["data"]["train_subset"] if gc["data"]["train_subset"] else 1281167
            print(f"Processing Speed: {(dataset_size/total_time).item()}")
            print(f"Time For Epoch: {total_time}")
            if gc["data"]["n_epochs"] == E:
                print(f"Communication Time: {get_comm_time(prof)}")
            print(f"Total IO Time: {total_io_time}")
            if gc.device == "cuda":
                print(f"Avg GPU Power Draw: {avg_power_draw*1e-3:.5f}")
                print(f"Avg GPU Utilization: {avg_gpu_util/gc.world_size:.2f}")
        dist.barrier()
        
        gc.start_eval(metadata={"epoch_num": E})
        if E % 4 == 0:
            for x, y in val_data:
                loss = valid_step(x.to(gc.device), y.to(gc.device), model, loss_fn, val_metric)
            val_accuracy = val_metric.compute().to(gc.device)
            dist.all_reduce(val_accuracy)
            if gc.rank == 0 and gc["training"]["benchmark"]:
                print(f"Train Accuracy at Epoch {E}: {val_accuracy/gc.world_size}")
                print(f"Validation Loss at Epoch {E}: {loss}")
        gc.stop_eval(metadata={"epoch_num": E})
        gc.stop_epoch(metadata={"epoch_num": E})
        if E >= gc["data"]["n_epochs"]:
                break
        if "val_accuracy" in dir(): 
            if val_accuracy/gc.world_size >= gc["training"]["target_accuracy"]:
                break
        E += 1
        scheduler.step()
    
    if "val_accuracy" in dir(): 
        if val_accuracy/gc.world_size >= gc["training"]["target_accuracy"]:
            gc.stop_run(metadata={"status": "success"})
            gc.log_event(key="target_accuracy_reached", value=gc["training"]["target_accuracy"], metadata={"epoch_num": E-1})
        else:
            gc.stop_run(metadata={"status": "target not met"})
    else:
        gc.stop_run(metadata={"status": "target not met"})

if __name__ == "__main__":
    main()
