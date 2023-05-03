import sys
import os
import torch
sys.path.insert(0, '/nas/xd/projects/transformers/src')
os.environ['HF_HOME'] = '/raid/xd/.cache/torch'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# device = torch.device('cuda:4')

from IPython.display import HTML, display
import copy
from math import log, exp

import random
import string
from itertools import product, chain
import math
from functools import reduce
import json
from tqdm import tqdm
from contextlib import suppress
from typing import Optional
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import einops



from transformers.data.data_collator import DataCollator, default_data_collator
from transformers import AutoConfig, pipeline
from transformers import RobertaForMaskedLM, RobertaTokenizer, GPT2LMHeadModel, GPT2Tokenizer, GPTNeoForCausalLM, AutoModelForCausalLM
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPTNeoForCausalLM, GPTJForCausalLM

import npm
import pysvelte as ps 
import unseal
from unseal import transformers_util as tutil
from unseal import hooks
import unseal.visuals.utils as utils
from unseal.hooks import HookedModel

from common_utils import *
from utils import *
# from child_utils import *
# from model_utils import *



###读数据

def choose_date(name):
    if name == "1":
        sentences_Winograd = read_Winograd("../../../data/circuits_datasets/winogrande_1.1/winogrande_1.1/dev.jsonl")
        textsent = sentences_Winograd
    elif name == "2":
        sentences_commonsenseqa = read_commonsenseqa("../../../data/circuits_datasets/commonsenseqa/dev_rand_split.jsonl")
        textsent = sentences_commonsenseqa
    elif name == "3":
        sentences_CSQA2 = read_CSQA2("../../../data/circuits_datasets/CSQA2/CSQA2_dev.json")   
        textsent = sentences_CSQA2
    elif name == "4":
        sentences_anli = read_anli("../../../data/circuits_datasets/anli_v1.0/anli_v1.0/R1/dev.jsonl")   
        textsent = sentences_anli[:10]
    else:
        sentences_RACE = read_RACE("../../../data/circuits_datasets/RACE/dev/middle")
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
def read_Winograd(file_path: str):
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
    return sentences

# CSQA   （问题+答案）
def read_commonsenseqa(file_path: str):
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
            sentence = connect(question,choices[label])
            sentences.append(sentence)
#     random.shuffle(sentences)
    return sentences

#  CSQA2 常识问答，短     （问题+答案）
def read_CSQA2(file_path: str):
    sentences = []
    with open(file_path, "r") as f:
        for line_num, line in enumerate(f):
            line = line.strip("\n")
            line = json.loads(line)
            if not line:
                continue
            question = line['question']
            ans = line['answer']
            sentence = connect(question,ans)
            sentences.append(sentence)
#     random.shuffle(sentences)
    return sentences

#NLI/MNLI进阶版，中    （前提+假设）
def read_anli(file_path: str):
    sentences = []
    with open(file_path, "r") as f:
        for line_num, line in enumerate(f):
            line = line.strip("\n")
            line = json.loads(line)
            if not line:
                continue
            context = line['context']
            hypothesis = line['hypothesis']
            sentence = connect(context,hypothesis)
            sentences.append(sentence)
#     random.shuffle(sentences)
    return sentences[:10]

#RACE Dataset  中学英语阅读理解，长    （原文+问题+选项）
def read_RACE(file_path: str):
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
            sentence = connect(context,que)
            sentence = connect(sentence,ans)
#             sentence = connect(sentence,label)
            sentences.append(sentence)
#     random.shuffle(sentences)
    return sentences

###数据处理

#前传 输出attention矩阵
def forward(model, tokenizer, text):
    inputs_en = tokenizer.encode_plus(text, return_tensors='pt')
    inputs = prepare_inputs(inputs_en, model.device)
    outputs = model(**inputs, output_attentions=True)
    return outputs.attentions

###### 

#构造输出的格式
def create_form(act_loc,text_id,value):
    src_id = int(text_id[act_loc[0]])
    tar_id = int(text_id[act_loc[1]])
    tuple_head = (act_loc,(src_id,tar_id),value)
    return tuple_head

#找到激活的head
# def create_attention1(outputs, heads, text, tokenizer): 
#     head = {}
#     head_list = []
#     head_list.append(text)
#     text_id = tokenizer.encode(text)
#     for i in heads:
#         tuple_head = [] #格式为[（（src，tar），（sid，tid），value）]
#         for j,k in enumerate(outputs[i[0]][0][i[1]]):
#             if j<=2: continue
#             temp = torch.squeeze((k==torch.max(k)).nonzero(),0)
#             if temp[0]!=0:
#                 value = float(max(k))
#                 tuple_head.append(create_form((j,int(temp[0])),text_id,value))
#         tuple_head = sort_form(tuple_head)
#         head[i] = tuple_head
#     head_list.append(head)
#     return head_list

def create_attention(outputs, heads, text, tokenizer): 
    head = {}
    head_list = []
    head_list.append(text)
    text_id = tokenizer.encode(text)
    for i in heads:
        tuple_head = []
        Max = torch.max(outputs[i[0]][0][i[1]], 1)
        temp = (0 != Max.indices).nonzero().tolist()
        for j in temp: 
            tuple_head.append(create_form((j[0], Max.indices[j[0]].item()), text_id, Max.values[j[0]].item()))
        tuple_head = sort_form(tuple_head)
        head[i] = tuple_head
    head_list.append(head)
    return head_list

def get_value(t):
    return t[-1]

def sort_form(tuple_head, k = 5):
    tuple_head_sort = sorted(tuple_head, key=get_value, reverse=True)
    return tuple_head_sort[:k]

#########


def connect(sentence,ans):
    return sentence+" "+ans.capitalize()

#随机选取几个句子
def choose(k,head_list): 
    textlist = []
    for i in head_list:
        if len(i[1])>0: textlist.append(i[0])
    text_choice = random.choices(textlist,k=k)
    return text_choice

#激活head注意力拼接
def concat_attn(attentions, heads):
    if heads == []:
        return
    
    output = attentions[heads[0][0]][0][heads[0][1]]
    if len(heads)>1:
        for i in heads[1:]:
            try:
                output= torch.stack([output, attentions[i[0]][0][i[1]]],0);
            except:
                output= torch.vstack((output,attentions[i[0]][0][i[1]][None]));
    return output.detach()
    
#预测概率值拼接（稀疏矩阵）
def concat_prob(matrix1,matrix2):
    if type(matrix1) == int:
        return matrix2
    else:
        try:
            output= torch.stack([matrix1, matrix2],0);
        except:
            output= torch.vstack((matrix1,matrix2[None]));
        return output

#绘图函数
def draw(attentions,tokenizer, text, heads):
    html_storage = {}
    val = concat_attn(attentions, heads)
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


#output为对于文本的list
def read_texts(num):
    texts = choose_date(num)
    return texts

#找到文本（texts）中 所有激活的 已给出的head（heads）
def find_activations(model, tokenizer, texts, heads):  # long run
    head_list = []
    for text in tqdm(texts):
        outputs = forward(model, tokenizer, text)#outputs attention矩阵
        head_list.append(create_attention(outputs, heads, text,tokenizer))
    return head_list

def tuple2str(head_list):
    head_list1 = []
    for i in head_list:
        hlist = [i[0],{str(j) : i[1][j] for j in i[1]}]
        head_list1.append(hlist)
    return head_list1

def str2tuple(head_list):
    head_list1 = []
    for i in head_list:
        hlist = [i[0],{eval(j) : i[1][j] for j in i[1]}]
        head_list1.append(hlist)
    return head_list1
    


#pickle dump head_dict #存json
def dump_list(head_list,name,num):
    with open(f'nrk/{name}_{num}.json', 'w',encoding='utf-8') as fp:   #覆写
    # with open(f'nrk/{name}_{num}.json', 'a',encoding='utf-8') as fp:   #添加
        head_list = tuple2str(head_list)
        json.dump(head_list, fp)

#pickle load head_dict #读json
def load_list(name, num):
    head_list1 = []
    with open(f'nrk/{name}_{num}.json', 'r',encoding='utf-8') as fp:
        head_list = json.load(fp)
    head_list = str2tuple(head_list)
    return head_list

def show_activation(model, tokenizer, head_list, heads, num): 
    texts = head_list[num][0] #重写函数，随机找k个文本输入
    head_list = []
    outputs = forward(model,tokenizer, texts)#outputs attention矩阵
    attention = create_attention(outputs, heads,  texts,tokenizer)
    head_list.append(attention)
    draw(outputs,tokenizer, attention[0], list(attention[1].keys())) #绘图函数
    
#展示部分
def show_activations(model, tokenizer, head_list, heads, k): 
    texts = choose(k,head_list) #重写函数，随机找k个文本输入
    head_list = []
    for i in range(k):
        outputs = forward(model,tokenizer, texts[i])#outputs attention矩阵
        attention = create_attention(outputs, heads,  texts[i],tokenizer)
        head_list.append(attention)
        draw(outputs,tokenizer, attention[0], list(attention[1].keys())) #绘图函数

# 统计head激活次数与频率     
def count_head_frequency(head_list,heads):
    head_dict ={}
    for i in heads:
        head_dict[i] = 0
    length = len(head_list)
    print("数据量：",length)
    for i in head_list:
        for j,k in i[1].items():
            if len(k)>0:   
                head_dict[j] += 1
    
    head_frequency = sorted(head_dict.items(), key=lambda x:x[1])
  
    data = [[i[1], i[1]/length] for i in head_frequency]
    heads = [i[0] for i in head_frequency]
    return (heads,data)

#展示head激活频率
def show_rate(head_list,heads):
    data = count_head_frequency(head_list,heads)
    rate = pd.DataFrame(data[1], index = data[0], columns = ["出现次数", "出现频率"])
    return rate

#计算token的关注距离
def count_distence(head_list, head, tokenizer):
    text_chose = head_list
    data = []
    dis = []
    for num,i in enumerate(text_chose):
        for j in i[1].items():
            if j[0] == head:
                for k in j[1]:
                    distence = k[0][0]-k[0][1]
                    dis.append(distence)
                    src_token = tokenizer.convert_ids_to_tokens(k[1][0])
                    tar_token = tokenizer.convert_ids_to_tokens(k[1][1])
                    Data = [tar_token,src_token,distence,num,k[2]]
                    data.append(Data)
            else:
                pass
    return dis,data

#展示关注距离  head_list为load_list函数读取的所有值  head为你想看的名称 如（3，12）
# def show_loc(head_list, k, head):
#     text_chose = choices(head_list,k=k)
def show_loc(head_list, head,tokenizer,k=100):
    dis,data = count_distence(head_list, head,tokenizer)
    data = sort_form(data,k)
    distence = pd.DataFrame(data, columns = ["被关注token", "关注token", "关注距离", "对应文本编号","注意力分数"])   
    pd.set_option('display.max_rows', None)      
    return distence

def which_sentence(head_list,num):
    print(head_list[num][0])

#计算每个距离（list）的激活数
# 激活次数  [1,5,6,3,2]
# 激活距离   0 1 2 3 4
def only_count_distence(dis):
    Max = max(dis)
    num = Max+10-Max%10
    distence = np.zeros(num)
    for i in dis:
        distence[i] += 1
    return distence


#展示每个距离激活次数的柱状图
def show_times_distence(head_list, heads,tokenizer):
    for i in heads:
        dis,data = count_distence(head_list, i,tokenizer)
        # distence = only_count_distence(dis)
        # x_label = [int(i) for i in range(len(distence))]
        # plt.bar(x_label,distence,0.6)
        plt.title(i)
        plt.xlabel("active_distence")
        plt.ylabel("active_time")
        plt.hist(dis, bins = 20, 
                            density = 1, 
                            color ='green',
                            alpha = 0.7)
        # plt.xticks(range(0,len(x_label),10), [i for i in range(0,len(x_label),10)], fontsize=7)
        plt.show()

####清除预测概率部分
def find_position(tokenizer,head_list, num): #num查询第几句话,查询全部可以用循环
    pos = {}
    text = head_list[num][0]
    for k,v in head_list[num][1].items():
        pos[k] = [(l[0][0],l[0][1]) for l in v]
    return text, pos

#将需要清除注意力的位置置零
def create_mask_row(head, position, length):
    mask = torch.ones(16,length)
    mask[head, position] = 0
    return mask

def create_mask_dot(position, length):
    mask = torch.ones(length,length)
    mask[position[0],position[1]] = 0
    return mask

#筛选数据，仅保存两次均预测正确且清除后预测概率发生变化的值
def pred_right(model, Data, input_ids, out, pos):
    data = copy.deepcopy(Data)
    result = input_ids[0][pos+1:] == out.logits[0].argmax(1)[pos:-1]
    for i in range(len(data)):
        data[i]["prob_2"] = None
    
    for i in range(pos+1,len(data)):
        if result[i-pos-1] == True:
            data[i]["prob_2"] = out.logits[0, i-1].softmax(-1)[input_ids[0][i]].item()
            
    for i in range(len(data)):
        data[i]["Prob_diff"] = 0
        data[i]["cut_pos"] = pos
        
    for i in range(pos+1,len(data)):
        if data[i]["prob_2"] != None:
            data[i]["Prob_diff"] = log(data[i]["prob_2"]) - log(data[i]["prob_1"])
            
    data1 = []        
    for i in data:
        if i["Prob_diff"] != 0:
            data1.append(i)
    return data1

def show_difference_of_probability(data):
    difference = pd.DataFrame(data, columns = ["文本编号","token位置", "token_id", "是否预测正确","清除注意力前概率","清除注意力后概率","log概率差","被注意力清除的token位置"])   
    pd.set_option('display.max_rows', None)      
    return difference


#前传两次，获得全部值
def make(model,tokenizer, text, pos, num, device, pattern):
    inputs = tokenizer.encode_plus(text, return_tensors='pt')
    inputs = prepare_inputs(inputs, device)
    Data = []
    json_dict = {}
    data ={}
    #前传第一次，不清除注意力
    out = model(**inputs, output_attentions=True)
    
    data["sentence"] = num
    data["pos"] = 0
    data["token_id"] = inputs.input_ids[0][0].item()
    data["is_right"] = 0
    data["prob_1"] = 0
    Data.append(data) #开头第一个token，不预测
    for i in range(1,len(inputs.input_ids[0])):
        data = {}
        token = inputs.input_ids[0][i].item()
        right = 1 if out.logits[0].argmax(1)[i-1] == token else 0
        prob = out.logits[0, i-1].softmax(-1)[token].item()
        data["sentence"] = num
        data["pos"] = i
        data["token_id"] = token
        data["is_right"] = right
        data["prob_1"] = prob
        Data.append(data)
    
    #前传第二次，清除注意力
    for h,p in pos.items():
        data_dict = {}
        layer,head = h
        if(pattern == "row"):
            self = model.transformer.h[layer].attn
            for position in p:
                self.pattern = pattern
                self.mask = create_mask_row(head, position[0], len(inputs.input_ids[0]))
                self.mask = self.mask.to(device)
                out = model(**inputs, output_attentions=True)
                data = pred_right(model, Data, inputs.input_ids, out, position[0])
                data_dict[position[0]] = data
                json_dict[(layer,head)] = data_dict
                del self.mask
                del self.pattern
            del self
        else:
            for position in p:
                for l in range(28):
                    self = model.transformer.h[l].attn
                    self.pattern = pattern
                    self.mask = create_mask_dot(position, len(inputs.input_ids[0]))
                    self.mask = self.mask.to(device)
                out = model(**inputs, output_attentions=True)
                data = pred_right(model, Data, inputs.input_ids, out, position[0])
                data_dict[position[0]] = data
                json_dict[(layer,head)] = data_dict
                for l in range(28):
                    self = model.transformer.h[l].attn
                    # self.pattern = pattern
                    # self.mask = create_mask_dot(position, len(inputs.input_ids[0]))
                    # self.mask = self.mask.to(device)
                    del self.mask
                    del self.pattern
                    # del self
    return [text,json_dict]

#构造可能性矩阵    
def Constructing_probability_matrix(tokenizer,data, heads, pos): #pos：（行，列）,length:句子长度， out.logits[1,length,50400]
    prosort = []
    TF = [len(i[1]) > 0 for i in pos.items()]
    heads = [heads[i] for i in range(len(TF)) if TF[i] == True]
    output = 0

    length = len(tokenizer.encode_plus(data[0]).input_ids)
    for h in heads:
        for j,k in data[1].items():
            if j == h:
                zero = torch.zeros(length,length)
                for column,r in k.items():
                    for row in r:
                        zero[row["pos"]][int(column)] = row["Prob_diff"] * -1  * 10
                output = concat_prob(output,zero)
    return output,heads

#画可能性矩阵关系图
def draw_prob(tokenizer, text, heads, matrix):
    html_storage = {}
    val = matrix
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

def Filter_head(data, head):
    data_head = []
    for i in data:
        try:
            data_head.append(i[1][head])
        except:
            continue
    return data_head


def sort_prob_diff(data, head):
    data_head = Filter_head(data, head)
    data_head_dict = []
    for i in data_head:
        for j in list(i.values())[:]:
            for k in j:
                data_head_dict.append(k)
    data_head = sorted(data_head_dict, key=lambda x:x['Prob_diff'])
    
    sorted_list = [list(i.values())[:] for i in data_head if list(i.values())[-2] > 1e-2 or list(i.values())[-2] < -1e-2]
    
    return data_head, sorted_list


def main():
    
    models = {}
    cache_dir = '/nas/xd/.cache/torch/transformers/'  # for models besides t5-3b/11b
    # cache_dir = '/mnt/nvme1/xd/.cache/torch/transformers/'  # for gpt-j-6B on elderberry
    proxies = {'http': '192.168.50.1:1081'} 
    model_name = "EleutherAI/gpt-j-6B"
    device = torch.device('cuda:7')
    model = GPTJForCausalLM.from_pretrained(model_name, proxies=proxies, cache_dir=cache_dir, revision="float16", torch_dtype=torch.float16, low_cpu_mem_usage=True).to(device)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', cache_dir=cache_dir)
    models[model_name] = model, tokenizer

    # heads = [(5,12), (7,9) , (7,2) , (6,5) , (3,7) , (8,7) , (6,2) , (3,12)]
    # for num in ["1","2","3","4","5"]:
    #     print('processing dataset', num)
    #     texts = read_texts(num)
    #     head_list = find_activations(model, tokenizer, texts, heads)
    #     dump_list(head_list,num)
    for num in ["1"]:
        head_list = load_list("head_list",num)
        data = []
        for i in tqdm(range(len(head_list))):
            text,pos = find_position(tokenizer, head_list, i)
            a = tokenizer.encode(text)
            for j in pos.items():
                Pos = j[1]  
                for k in Pos:
                    if(a[k[0]] != a[k[1]] or k[0]-k[1]==1):
                        Pos.pop(Pos.index(k))
            print(pos)
            return 0           
            Data = make(model, tokenizer, text, pos, i, device, "dot")
            data.append(Data)
        
        dump_list(data,"probability_dot_without_same",int(num))


if __name__== "__main__" :
    main()
