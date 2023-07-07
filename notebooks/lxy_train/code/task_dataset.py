from cProfile import label
from importlib.resources import contents
import json

import pandas as pd
import torch
from datasets import load_dataset
from torch.utils.data import Dataset
import random
import pickle
from transformers import AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint, is_main_process
class TasksDataset(Dataset):
    def __init__(self, train_path, tokenizer, train_mode = 'train', split = 80, max_length = 1024, \
                 k_shot = 1, seed = 42 , local_rank = 0, eval_task_file = ""):   #需更改split的值修改训练与测试集比例 8:2（当前），5:5，2:8

        # 读取制作的数据集，进行反序列化
        with open(train_path, 'r') as f:   
            content = f.read()
        self.dataset = json.loads(content)

        # 设置种子的目的，方便随机选取的测试集可复现，task_key是固定的。
        random.seed(seed)
        split_threhold = int(len(self.dataset) * split / 100.0) 
        random_keys = random.sample(self.dataset.keys(), len(self.dataset.keys()))    #random.shaffle()?
        # 根据split_threhold切分训练集测试集
        task_abstract_keys = random_keys[:split_threhold] if train_mode == 'train' \
            else random_keys[split_threhold:]
        # task_abstract_keys = ["2","0"]
        # print(len(task_abstract_keys))
        # 保存评估时的任务，方便在child_utils中测试模型微调后每个任务的准确率
        if train_mode == 'eval' and is_main_process(local_rank):
            dictss = {}
            for task_abstract_key ,task_dataset in self.dataset.items():
                if task_abstract_key not in task_abstract_keys: continue
                for task_key, value in task_dataset.items():
                    text = value['texts'][0]
                    dictss[task_key] = text
            print(len(dictss))
            contents = json.dumps(dictss)
            with open(eval_task_file,'w') as f:
                f.write(contents)

        # 保存每个样本 tuple(key,text,bos,input_ids)
        self.post_lists = [] 
        for task_abstract_key, task_dataset in self.dataset.items():
            if task_abstract_key not in task_abstract_keys: continue
            for task_key, value in task_dataset.items():
                pos_tuple = (task_key, )  
                self.post_lists.extend([pos_tuple + text_range_ids for text_range_ids in zip(*value.values())])     #zip(*zip(text,bos,input_ids))  pos_tuple：(0,)  post_lists：[(0, 1, 2, 3), (0, 4, 5, 6), (0, 7, 8, 9)]

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
        origin_ids = torch.tensor(origin_ids).reshape(-1)

        # 确保该tokenizer的input_ids和origin_id内容一致
        assert origin_ids.equal(input_ids[:origin_ids.size(0)])
        # 确保padding后面都是pad_token_id
        assert (input_ids[origin_ids.size(0):] != self.tokenizer.pad_token_id).sum() == 0

        #                                                                                ids[0,1,2,3,4,5...]                      ids[0,1,2,3,4,5...]
        #构造label,将计算loss的位置设置为非-100的值,相对于xd代码，label没有进行偏移。      labels[  0,1,2,3,4,5...]      ------>    labels[0,1,2,3,4,5...]
        labels = torch.ones_like(input_ids) * (-100)
        bos_indices = [r[-1] - 1 for r in ranges]
        eos_indices = [bos_i + 2 for bos_i in bos_indices]
        for bos_i, eos_i in zip(bos_indices, eos_indices):
            labels[bos_i + 1: eos_i] = input_ids[bos_i + 1: eos_i]

        # 计算k-shot后的loss
        label_indices = [bos + 1 for bos in bos_indices]
        labels[:label_indices[self.k_shot]] = -100   #训练数据置-100，不计算梯度

        return {
            # 'key':key,
            # 'texts':texts,
            "input_ids": input_ids,
            "attention_mask": attn_masks,
            "labels": labels
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

    def __getitem__(self, idx):    # 使用index(简写为idx)获取某个数据
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