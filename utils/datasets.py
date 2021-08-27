import torch
from torch.utils.data import Dataset
import json
import pickle as pkl
import os

class PAN2020(Dataset):
    def __init__(self, path, tokenizer, block_size, special_tokens=3, make_lower=False):
        assert os.path.isfile(path)

        is_pickle = False
        if path[-3:] == "pkl":
            is_pickle = True
            print("Using pickle.")

        print(f"Opening dataset from {path}.")

        self.tokenizer = tokenizer
        self.block_size = block_size - special_tokens
        self.data = []
        self.labels = []
        self.make_lower = make_lower
        if is_pickle:
            samples = pkl.load(open(path, "rb"))
            for sample in samples:
                self.data.append(sample["pair"])
                self.labels.append(1 if sample["same"] == True else 0)
        else:
            for sample in open(path):
                sample = json.loads(sample)
                self.data.append(sample["pair"])
                self.labels.append(1 if sample["same"] == True else 0)
        self.ds_len = len(self.labels)

        print("len(self.labels", len(self.labels))
        print("len(data)", len(self.data))
        print("self.ds_len", self.ds_len)

    def __len__(self):
        return self.ds_len

    def __getitem__(self, idx):
        sep_token = self.tokenizer.sep_token
        cls_token = self.tokenizer.cls_token
        pad_token = self.tokenizer.pad_token

        if self.make_lower:
            sample1 = self.data[idx][0].lower()
            sample2 = self.data[idx][1].lower()
        else:
            sample1 = self.data[idx][0]
            sample2 = self.data[idx][1]

        sample1_tokens = self.tokenizer.tokenize(sample1)
        sample2_tokens = self.tokenizer.tokenize(sample2)

        len_s1 = len(sample1_tokens)
        len_s2 = len(sample2_tokens)

        min_size = min(len_s1, len_s2)

        sample1_tokens = sample1_tokens[:min_size]
        sample2_tokens = sample2_tokens[:min_size]

        sequence_length = int(self.block_size / 2)

        sample_list = []

        for idx_1 in range(0, min_size, sequence_length):
            seq1 = sample1_tokens[idx_1 : idx_1 + sequence_length]
            seq2 = sample2_tokens[idx_1 : idx_1 + sequence_length]

            len_s1 = len(seq1)
            len_s2 = len(seq2)

            if len_s1 < self.block_size:
                seq1.extend([pad_token] * (sequence_length - len_s1))

            if len_s2 < self.block_size:
                seq2.extend([pad_token] * (sequence_length - len_s2))

            entire_sequence = [cls_token] + seq1 + [sep_token] + seq2 + [sep_token]
            attention_mask = [
                1 if token != pad_token else 0 for token in entire_sequence
            ]
            token_type_ids = [
                0 if idx_2 < sequence_length else 1
                for idx_2 in range(self.block_size + 2)
            ]

            sample_list.append(
                {
                    "token_ids": torch.tensor(
                        self.tokenizer.convert_tokens_to_ids(entire_sequence)
                    ),
                    "attention_mask": torch.tensor(attention_mask),
                    "token_type_ids": torch.tensor(token_type_ids),
                }
            )

        label = self.labels[idx]

        return sample_list, torch.tensor(label)