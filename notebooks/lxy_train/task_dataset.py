from cProfile import label
from importlib.resources import contents
import json

import pandas as pd
import torch
from datasets import load_dataset
from torch.utils.data import Dataset
# import sys
# sys.path.insert(0, '/nas/xd/projects/transformers/notebooks')
# from child_utils import Ranges
import pickle
from transformers import AutoTokenizer
class TasksDataset(Dataset):
    def __init__(self, train_path, tokenizer, train_mode = 'train', split = 0.8, max_length = 1024, k_shot = 1):

        with open(train_path, 'r') as f:   
            content = f.read()
        self.dataset = json.loads(content)

        split_threhold = int(len(self.dataset) * split)
        task_keys = list(self.dataset.keys())[:split_threhold] if train_mode == 'train' \
            else list(self.dataset.keys())[split_threhold:]

        self.post_lists = [] # (tuple: key,texts,label)
        
        for key, value in self.dataset.items():
            if key not in task_keys: continue
            pos_tuple = (key, )
            self.post_lists.extend([pos_tuple + text_range_ids for text_range_ids in zip(*value.values())])

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.k_shot = k_shot
        

    def __len__(self):
        return len(self.post_lists)
 
    def __getitem__(self, idx):
        key, texts, ranges, origin_ids = self.post_lists[idx]

        encodings_dict = self.tokenizer(texts, truncation=True, max_length=self.max_length, padding="max_length")

        input_ids = torch.tensor(encodings_dict["input_ids"])
        attn_masks = torch.tensor(encodings_dict["attention_mask"])

        # assert和原来一致
        origin_ids = torch.tensor(origin_ids).reshape(-1)
        # print(origin_ids.equal(input_ids[:origin_ids.size(0)]))
        assert origin_ids.equal(input_ids[:origin_ids.size(0)])
        # ssert 0 == ((x != y).sum())
        assert (input_ids[origin_ids.size(0):] != self.tokenizer.pad_token_id).sum() == 0

        labels = torch.ones_like(input_ids) * (-100)

        bos_indices = [r[-1] - 1 for r in ranges]
        eos_indices = [bos_i + 2 for bos_i in bos_indices]
        for bos_i, eos_i in zip(bos_indices, eos_indices):
            labels[bos_i + 1: eos_i] = input_ids[bos_i + 1: eos_i]
            # ans_ids = input_ids[0, bos_i + 1: eos_i]
            # labels[0, bos_i: eos_i - 1] = ans_ids
        labels[:bos_indices[self.k_shot]] = -100

        return {
            # 'key':key,
            # 'texts':texts,
            "input_ids": input_ids,
            "attention_mask": attn_masks,
            "labels": labels
            # "ranges": ranges,
        }



class TLDRDataset(Dataset):
    def __init__(self, train_path, tokenizer, split, max_length=550):
        self.post_list = []
        dataset = load_dataset(train_path, split=split)
        for sample in dataset:
            self.post_list.append(sample["prompt"] + sample["label"])
        if "valid" in split:
            self.post_list = self.post_list[0:2000]
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.input_ids = []
        self.attn_masks = []

    def __len__(self):
        return len(self.post_list)

    def __getitem__(self, idx):
        txt = self.post_list[idx]
        encodings_dict = self.tokenizer(txt, truncation=True, max_length=self.max_length, padding="max_length")
        input_ids = torch.tensor(encodings_dict["input_ids"])
        attn_masks = torch.tensor(encodings_dict["attention_mask"])

        return {
            "input_ids": input_ids,
            "attention_mask": attn_masks,
            "labels": input_ids,
        }


class ComparisonDataset(Dataset):
    def __init__(self, comparison_path, tokenizer, max_length=550):
        with open(comparison_path, "r") as f:
            dataset = [json.loads(line) for line in f]

        self.tokenizer = tokenizer
        self.post_list = []
        self.summaries_0 = []
        self.summaries_1 = []
        self.labels = []
        self.max_length = max_length

        def make_text(post, summarize):
            return f"SUBREDDIT: r/{post['subreddit']}\nTITLE: {post['title']}\nPOST: {post['post']}\nTL;DR: {summarize}"

        for sample in dataset:  # chosen summary is always the first one
            self.post_list.append(sample["info"]["post"])
            # NOTE: The chosen summary is always the first one, i.e. `sample["summaries"][0]`
            if sample["choice"] == 0:
                self.summaries_0.append(make_text(sample["info"], sample["summaries"][0]["text"]))
                self.summaries_1.append(make_text(sample["info"], sample["summaries"][1]["text"]))
            else:
                self.summaries_0.append(make_text(sample["info"], sample["summaries"][1]["text"]))
                self.summaries_1.append(make_text(sample["info"], sample["summaries"][0]["text"]))
            self.labels.append(0)

    def __len__(self):
        return len(self.post_list)

    def __getitem__(self, idx):
        summ0 = self.summaries_0[idx]
        summ1 = self.summaries_1[idx]
        encodings_dict = self.tokenizer(
            [summ0, summ1],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )
        input_ids = torch.tensor(encodings_dict["input_ids"])
        attention_mask = torch.tensor(encodings_dict["attention_mask"])
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class AllSummDataset(Dataset):
    def __init__(self, train_path, tokenizer, split, max_length=1024):
        df = pd.read_parquet(train_path)
        if split == "valid":
            df = df.sample(n=5000)
        self.summarizes = []
        for i, row in df.iterrows():
            self.summarizes.append(f"Summarize: {row['text']}. TL;DR: {row['summary']}")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.input_ids = []
        self.attn_masks = []

    def __len__(self):
        return len(self.summarizes)

    def __getitem__(self, idx):
        txt = self.summarizes[idx]
        encodings_dict = self.tokenizer(txt, truncation=True, max_length=self.max_length, padding="max_length")
        input_ids = torch.tensor(encodings_dict["input_ids"])
        attn_masks = torch.tensor(encodings_dict["attention_mask"])

        return {
            "input_ids": input_ids,
            "attention_mask": attn_masks,
            "labels": input_ids,
        }


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-j-6B')
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    dataset = TasksDataset('/nas/xd/projects/transformers/notebooks/lxy_train/task_datasets.pickle', tokenizer)
    print(dataset[0])