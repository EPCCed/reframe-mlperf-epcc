import os

import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, RandomSampler, DataLoader, SequentialSampler

from ML.gc import GlobalContext
gc = GlobalContext()

class pretraining_dataset_v1(Dataset):
    def __init__(self, f, input_file, max_pred_length=76):
        self.input_file = input_file
        self.max_pred_length = max_pred_length
        # f = h5py.File(input_file, "r")
        keys = [
            "input_ids",
            "input_mask",
            "segment_ids",
            "masked_lm_positions",
            "masked_lm_ids",
            "next_sentence_labels",
        ]
        self.inputs = [np.asarray(f[key][:]) for key in keys]
        print(f"Loaded {len(self.inputs[0]):d} samples from datafile: {input_file}")
        f.close()

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.inputs[0])

    def __getitem__(self, index):
        [
            input_ids,
            input_mask,
            segment_ids,
            masked_lm_positions,
            masked_lm_ids,
            next_sentence_labels,
        ] = [
            torch.from_numpy(input[index].astype(np.int64))
            if indice < 5
            else torch.from_numpy(np.asarray(input[index].astype(np.int64)))
            for indice, input in enumerate(self.inputs)
        ]
        masked_lm_labels = torch.zeros(input_ids.shape, dtype=torch.long) - 100
        index = self.max_pred_length
        masked_token_count = torch.count_nonzero(masked_lm_positions)
        if masked_token_count != 0:
            index = masked_token_count
        masked_lm_labels[masked_lm_positions[:index]] = masked_lm_ids[:index]
        # print(f"input_mask_len = {torch.count_nonzero(input_ids)}  index = {index}")

        return [
            input_ids,
            segment_ids,
            input_mask,
            masked_lm_labels,
            next_sentence_labels,
        ]


class pretraining_dataset_v2(Dataset):
    def __init__(
        self, f, input_file, max_pred_length=76, max_seq_length=512, packed_samples=False
    ):
        self.input_file = input_file
        self.max_pred_length = max_pred_length
        self.max_seq_length = max_seq_length
        self.packed_samples = packed_samples

        # f = h5py.File(input_file, "r")
        if not self.packed_samples:
            keys = [
                "input_ids",
                "segment_ids",
                "masked_lm_positions",
                "masked_lm_ids",
                "next_sentence_labels",
            ]
        else:
            keys = [
                "input_ids",
                "segment_ids",
                "masked_lm_positions",
                "masked_lm_ids",
                "packed_input_len",
                "packed_masked_lm_len",
                "next_sentence_labels",
            ]

        self.inputs = [np.asarray(f[key][:]) for key in keys]
        print(f"Loaded {len(self.inputs[0]):d} samples from datafile: {input_file}")
        f.close()

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.inputs[0])

    def __getitem__(self, index):
        input_ids = np.zeros((self.max_seq_length)).astype(np.int64)
        input_mask = np.zeros((self.max_seq_length)).astype(np.int64)
        segment_ids = np.zeros((self.max_seq_length)).astype(np.int64)
        next_sentence_labels = np.zeros((3)).astype(np.int64)
        packed_input_len = np.zeros((3)).astype(np.int64)

        if not self.packed_samples:
            [
                _input_ids,
                _segment_ids,
                _masked_lm_positions,
                _masked_lm_ids,
                _next_sentence_labels,
            ] = [
                input[index].astype(np.int64)
                if indice < 4
                else np.asarray(input[index].astype(np.int64))
                for indice, input in enumerate(self.inputs)
            ]
        else:
            [
                _input_ids,
                _segment_ids,
                _masked_lm_positions,
                _masked_lm_ids,
                _packed_input_len,
                _packed_masked_lm_len,
                _next_sentence_labels,
            ] = [
                input[index].astype(np.int64)
                for indice, input in enumerate(self.inputs)
            ]

        input_mask_len = _input_ids.shape[-1]
        input_ids[:input_mask_len] = _input_ids
        input_mask[:input_mask_len] = np.ones((1, input_mask_len)).astype(np.int64)
        segment_ids[:input_mask_len] = _segment_ids
        masked_lm_labels = np.zeros(input_ids.shape, dtype=np.int64) - 100
        masked_lm_labels[_masked_lm_positions] = _masked_lm_ids

        if not self.packed_samples:
            next_sentence_labels = _next_sentence_labels

            return [
                torch.from_numpy(input_ids),
                torch.from_numpy(segment_ids),
                torch.from_numpy(input_mask),
                torch.from_numpy(masked_lm_labels),
                torch.from_numpy(next_sentence_labels),
            ]
        else:
            packed_seqs = _packed_input_len.shape[-1]
            next_sentence_labels[:packed_seqs] = _next_sentence_labels
            packed_input_len[:packed_seqs] = _packed_input_len

            return [
                torch.from_numpy(input_ids),
                torch.from_numpy(segment_ids),
                torch.from_numpy(input_mask),
                torch.from_numpy(masked_lm_labels),
                torch.from_numpy(next_sentence_labels),
                torch.from_numpy(packed_input_len),
            ]


def get_dataset(input_file, max_pred_len=76, max_seq_len=512, pack_samples=False):
    f = h5py.File(input_file, "r")
    if "input_mask" not in f.keys():
        return pretraining_dataset_v2(f, input_file, max_pred_len, max_seq_len, pack_samples)
    else:
        return pretraining_dataset_v1(f, input_file, max_pred_len)


def get_pretrian_dataloader(input_file):
    train_data = get_dataset(input_file)
    sampler = RandomSampler(train_data)
    local_bs = gc["data"]["global_batch_size"] // gc.world_size
    return DataLoader(train_data, 
                      sampler=sampler, 
                      batch_size=local_bs, 
                      shuffle=True,
                      drop_last=True,
                      pin_memory=True if gc.device != "cpu" else False)


def get_eval_dataset():
    eval_data = []
    eval_dir = os.path.join(gc["data"]["data_dir"], "hdf5", "eval_varlength")
    for eval_file in sorted(os.listdir(eval_dir)):
        eval_file_path = os.path.join(eval_dir, eval_file)

        if os.path.isfile(eval_file_path) and "part" in eval_file_path:
            eval_data.extend(
                get_dataset(
                    eval_file_path, max_pred_length=76
                )
            )
            if len(eval_data) > 10000:  # 10000 eval samples
                eval_data = eval_data[: 10000]
                break
    if torch.distributed.is_initialized():
        chunk_size = 10000 // gc.world_size
        rank = gc.rank
        remainder = 10000 % gc.world_size
        if rank < remainder:
            eval_data = eval_data[
                (chunk_size + 1) * rank : (chunk_size + 1) * (rank + 1)
            ]
        else:
            eval_data = eval_data[
                chunk_size * rank + remainder : chunk_size * (rank + 1) + remainder
            ]

    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(
        eval_data, sampler=eval_sampler, batch_size=gc["data"]["global_eval_batch_size"], num_workers=0
    )
    return eval_dataloader