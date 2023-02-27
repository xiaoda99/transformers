from common_utils import *
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from types import MethodType
from tqdm import tqdm
from collections import defaultdict, OrderedDict, Counter
from datetime import datetime
from io import StringIO
from itertools import chain
import math
from functools import reduce
import numpy as np 
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns

import random
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data.sampler import RandomSampler, Sampler, SequentialSampler
from torch.utils.data.dataloader import DataLoader

from transformers.data.data_collator import DataCollator, default_data_collator
from transformers import AutoConfig, pipeline
from transformers import RobertaForMaskedLM, RobertaTokenizer, GPT2LMHeadModel, GPT2Tokenizer, GPTNeoForCausalLM, AutoModelForCausalLM

from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPTNeoForCausalLM, GPTJForCausalLM
import pysvelte as ps 
import unseal
import string
from itertools import product, chain
import math
import json
from IPython.display import HTML, display

import torch
from unseal import transformers_util as tutil
from unseal import hooks
import unseal.visuals.utils as utils
from unseal.hooks import HookedModel
import npm
from contextlib import suppress
from typing import Optional
import random
import pysvelte as ps 


from utils import *
from child_utils import *
from common_utils import *
from model_utils import *
from weight_analysis import *

def _choose_date(name):
    if(name == "1"):
        sentences_Winograd = _read_Winograd("/nas/xd/data/circuits_datasets/winogrande_1.1/winogrande_1.1/train_l.jsonl")
        textsent = sentences_Winograd
    elif(name == "2"):
        sentences_commonsenseqa = _read_commonsenseqa("/nas/xd/data/circuits_datasets/commonsenseqa/train_rand_split.jsonl")
        textsent = sentences_commonsenseqa
    elif(name == "3"):
        sentences_CSQA2 = _read_CSQA2("/nas/xd/data/circuits_datasets/CSQA2/CSQA2_train.json")   
        textsent = sentences_CSQA2
    elif(name == "4"):
        sentences_anli = _read_anli("/nas/xd/data/circuits_datasets/anli_v1.0/anli_v1.0/R1/train.jsonl")   
        textsent = sentences_anli
    else:
        sentences_RACE = _read_RACE("/nas/xd/data/circuits_datasets/RACE/train/middle")
        textsent = sentences_RACE
    return textsent

# 获取file_path路径下的所有TXT文本内容和文件名 用于RACE Dataset
def get_text_list(file_path):
    files = os.listdir(file_path)
    text_list = []
    for file in files:
        with open(os.path.join(file_path, file), "r") as f:
            text_list.append(f.read())
    return text_list


#经典Winograd进阶版，短   （原文+嵌入）
def _read_Winograd(file_path: str):
    label_dict = {'1':"option1", '2':"option2"}
    sentences = []
    with open(file_path, "r") as f:
        for line_num, line in enumerate(f):
            line = line.strip("\n")
            line = json.loads(line)
            if not line:
                continue
            question = line['sentence']
            choices = line['answer']
            replace = line[label_dict[choices]]
            sentence = question.replace("_",replace)
            sentences.append(sentence)
    
#     random.shuffle(sentences)
    return sentences[:20]

# CSQA   （问题+答案）
def _read_commonsenseqa(file_path: str):
    label_dict = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4}
    sentences = []
    with open(file_path, "r") as f:
        for line_num, line in enumerate(f):
            line = line.strip("\n")
            line = json.loads(line)
            if not line:
                continue
            question = line['question']['stem']
            choices = [c['text'] for c in line['question']['choices']]
            label = label_dict[line['answerKey']] if 'answerKey' in line else None
            sentence = _connect(question,choices[label])
            sentences.append(sentence)
#     random.shuffle(sentences)
    return sentences[:20]

#  CSQA2 常识问答，短     （问题+答案）
def _read_CSQA2(file_path: str):
    sentences = []
    with open(file_path, "r") as f:
        for line_num, line in enumerate(f):
            line = line.strip("\n")
            line = json.loads(line)
            if not line:
                continue
            question = line['question']
            ans = line['answer']
            sentence = _connect(question,ans)
            sentences.append(sentence)
#     random.shuffle(sentences)
    return sentences[:20]

#NLI/MNLI进阶版，中    （前提+假设）
def _read_anli(file_path: str):
    sentences = []
    with open(file_path, "r") as f:
        for line_num, line in enumerate(f):
            line = line.strip("\n")
            line = json.loads(line)
            if not line:
                continue
            context = line['context']
            hypothesis = line['hypothesis']
            sentence = _connect(context,hypothesis)
            sentences.append(sentence)
#     random.shuffle(sentences)
    return sentences[:20]

#RACE Dataset  中学英语阅读理解，长    （原文+问题+选项）
def _read_RACE(file_path: str):
    sentences = []
    label_dict = {'A':0, 'B':1, 'C':2, 'D':3}
    text_list = get_text_list(file_path)
    for i in range(len(text_list)):
        text_list[i] = eval(text_list[i])
        context = text_list[i]['article']
#         label = text_list[i]['id']
        for j in range(len(text_list[i]['answers'])):
            que = text_list[i]['questions'][j]
            ans = text_list[i]['options'][j][label_dict[text_list[i]['answers'][j]]]
            sentence = _connect(context,que)
            sentence = _connect(sentence,ans)
#             sentence = _connect(sentence,label)
            sentences.append(sentence)
#     random.shuffle(sentences)
    return sentences[:10]

#前传 输出attention矩阵
def forward(model, tokenizer, text):
    inputs1 = tokenizer.encode_plus(text, return_tensors='pt')
    inputs = prepare_inputs(inputs1, model.device)
    outputs = model(**inputs, output_attentions=True)
    return outputs.attentions

# 查找激活的head，但是感觉还是不好，可以改成如果头的注意力分数不是最大的，就加入
def _create_attention(outputs, heads, head_dict): #任何一个开头的注意力分数<0.5，加入该head
    head = []
    for i in heads:
        flag = False
        for j in range(2,len(outputs[0][0][0])):
            if(outputs[i[1]][0][i[0]][j][0] != max(outputs[i[1]][0][i[0]][j])):
                flag = True
                break
        if(flag == True):
            head.append(i)
    head_dict["heads"].append(head)

#随机选取几个句子
def choose(k,head_dict): 
    textlist = []
    for i,j in zip(head_dict["text"],head_dict["heads"]):
        if(j!=[]): textlist.append(i)
    text_choice = random.choices(textlist,k=k)
    return text_choice

#激活head拼接
def _concat(attentions, heads):
    if(heads == []):
        return
    output = attentions[heads[0][1]][0][heads[0][0]]
    if(len(heads)>1):
        for i in heads[1:]:
            try:
                output= torch.stack([output, attentions[i[1]][0][i[0]]],0);
            except:
                output= torch.vstack((output,attentions[i[1]][0][i[0]][None]));
    val= torch.tensor([item.cpu().detach().numpy() for item in output]).cuda()
    return val
    
#绘图函数
def draw(attentions, text, heads):
    html_storage = {}
    val = _concat(attentions, heads)
    try:
        vall = einops.rearrange(val[:,:,:], 'h n1 n2 -> n1 n2 h ')
    except:
        val = val[None]
        vall = einops.rearrange(val[:,:,:], 'h n1 n2 -> n1 n2 h ')
    sent = tokenizer.tokenize(text)
    tokenized_text = list(map(tokenizer.convert_tokens_to_string, map(lambda x: [x], sent))) 
    html_object = ps.AttentionLogits(tokens=tokenized_text, attention=vall, pos_logits=vall, neg_logits=vall, head_labels=[f'{i}:{j}' for i,j in heads])
    html_object = html_object.update_meta(suppress_title=True)
    html_str = html_object.html_page_str()

    html_storage[f'{text}'] = html_str
    html_objects = {key: HTML(val) for (key, val) in html_storage.items()}
    print(f'\n')
    display(html_objects[f'{text}'], display_id=text)

#展示部分
def show_activations(model, tokenizer, head_dict, heads, k): 
#     texts = choose(k,head_dict) #重写函数，随机找k个文本输入
#     head_dict = {"text":[],"heads":[]}
    head_dict = {"text":[" A is bigger than B, so B is smaller."],"heads":[]}
    for i in range(k):
#         head_dict["text"].append(texts[i])
#         outputs = forward(model,tokenizer, texts[i])#outputs attention矩阵
        outputs = forward(model,tokenizer, " A is bigger than B, so B is smaller.")
        _create_attention(outputs, heads, head_dict)
        draw(outputs, head_dict["text"][i], head_dict["heads"][i]) #绘图函数