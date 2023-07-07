import logging
import random
import os
import sys
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,2,3,8"
# sys.path.insert(0, '/nas/xd/projects/transformers/src')
import evaluate
import numpy as np
import torch
from data_args import DataTrainingArguments
from model_args import ModelArguments
from task_dataset import TasksDataset
import math
import transformers
from transformers.deepspeed import deepspeed_init
from transformers.trainer_utils import get_last_checkpoint, is_main_process  #is_main_process不是应用于TPU训练吗
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments, default_data_collator, set_seed,
    HfArgumentParser, TrainingArguments, TransfoXLConfig)


# Setup logging
logger = logging.getLogger(__name__)



def main():
    # load arguments  
    # 类ModelArguments中包含的是关于模型的属性，如model_name，config_name，tokenizer_name等，类在run.py文件中定义；
    # 类DataTrainingArguments中包含的是关于微调数据的属性，如task_name，data_dir等，类在transformers/data/datasets/glue.py文件中定义；
    # TrainingArguments中包含的是关于微调过程的参数，如batch_size，learning_rate等参数，类在transformers/training_args.py中定义。
    parser = HfArgumentParser(   
        (ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses(
        )
    training_args.sampler = None

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(                   #os.path.isdir()用于判断对象是否为一个目录。 
            training_args.output_dir    #os.path.isfile()用于判断对象是否为一个文件。
    ) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(  #os.listdir()用于返回一个由文件名和目录名组成的列表，需要注意的是它接收的参数需要是一个绝对的路径。
                training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome.")
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )    

    # Setup logging
    logging.basicConfig(
        format='[%(levelname)s][%(asctime)s][%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank
                                                    ) else logging.WARN)


    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        +
        f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    # 记录参数是否有误    
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info(f"Model_settings: {model_args}")
    logger.info(f"data_settings: {data_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load model and tokenizer
    
    cache_dir = '/nas/xd/.cache/torch/transformers/'
    proxies = {'http': '192.168.50.1:1081'} 

    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, cache_dir=cache_dir, proxies = proxies)#, low_cpu_mem_usage=True)
    # model = AutoModelForCausalLM.from_pretrained('/nas/xd/projects/transformers/notebooks/nrk_train/code/train_tasks_0701_28',cache_dir=cache_dir,proxies = proxies)
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, cache_dir=cache_dir)
    
    tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer))
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id
    # print(model.config.use_cache)
    model.config.use_cache = False
    # print(tokenizer.eos_token, tokenizer.eos_token_id)

    if data_args.block_size is None:
        logger.info("Setting `block_size` 1024 since it was not set")
        tokenizer.model_max_length = 1024   #这一步没用
    else:
        logger.info("Setting `block_size` to %d", data_args.block_size)
        tokenizer.model_max_length = data_args.block_size
    


    # Load train datasets
    train_dataset = TasksDataset(
        data_args.train_file,
        tokenizer,
        'train',
        data_args.validation_split_percentage,
        max_length=data_args.max_input_length,
        k_shot = data_args.k_shot,
        seed = training_args.seed,
        local_rank = training_args.local_rank,
        eval_task_file = data_args.eval_task_file
    )
    # 传入两个数据集
    train_eval_dataset = TasksDataset(
        data_args.train_file,
        tokenizer,
        'train',
        data_args.validation_split_percentage,
        max_length=data_args.max_input_length,
        k_shot = data_args.k_shot,
        seed = training_args.seed,
        local_rank = training_args.local_rank,
        eval_task_file = data_args.eval_task_file
    )
    eval_dataset = TasksDataset(
        data_args.train_file,
        tokenizer,
        'eval',
        data_args.validation_split_percentage,
        max_length=data_args.max_input_length,
        k_shot = data_args.k_shot,
        seed = training_args.seed,
        local_rank = training_args.local_rank,
        eval_task_file = data_args.eval_task_file
    )

    # def compute_metrics1(eval_preds):
    #     pred_ids = torch.tensor(eval_preds.predictions[:, :-1])
    #     labels_ids = torch.tensor(eval_preds.label_ids[:, 1:])
    #     assert labels_ids.size() == pred_ids.size(), f'{labels_ids.size()} != {pred_ids.size()}'
        
    #     label_mask = (labels_ids != -100)   

    #     accuracy = torch.einsum('bi->b', (pred_ids == labels_ids ) * label_mask) \
    #         / torch.einsum('bi->b', label_mask)
        
    #     return {#'example_acc': accuracy,
    #             #'task_acc':accuracy.view(-1, 16).mean(-1).tolist(),
    #             'all_acc': accuracy.mean()}
    
    # Set up the metric
    # 评估在这里计算准确率
    def compute_metrics(eval_preds):
        # pred_ids 是 logits经过argmax后的结果 维度【batch, seq_len】
        pred_ids = eval_preds.predictions[:, :-1]
        labels_ids = eval_preds.label_ids[:, 1:]      #在这里偏移？
        # 保持维度一致。
        assert labels_ids.shape == pred_ids.shape, f'{labels_ids.shape} != {pred_ids.shape}'
        # 将label不是-100的地方mask掉，不计算准确率。
        label_mask = (labels_ids != -100)    #tensor([False, False,  True, False, False,  True])
        # 计算总的准确率。
        accuracy = ((pred_ids == labels_ids ) * label_mask).sum(-1) / label_mask.sum(-1)  
       
        return {'acc1': accuracy.tolist(), #每个样本的准确率，保留3/4位小数
                # 'task_acc':np.around(accuracy.reshape((-1, 16)).mean(-1), 3), # 想法是每个task准确率，但是结果不对。
                'all_acc': np.around(accuracy.mean(),4)}

    # Create a preprocessing function to extract out the proper logits from the model output
    def preprocess_logits_for_metrics(logits, labels):   #logits转换为各位置概率最高的值
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits.argmax(dim=-1)    

    # 设置是否评估几个集合，这里默认两个都进行评估。
    # merge_dataset = {'train_eval_dataset': train_eval_dataset, 'eval_dataset': eval_dataset}
    merge_dataset = {"eval_dataset": eval_dataset}
    # 构造Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=merge_dataset,
        compute_metrics=compute_metrics,
        data_collator=default_data_collator,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,  #返回修改后的logits，这里对logits做了softmax
    )

    # 模型训练
    if training_args.do_train:
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif model_args.model_name_or_path is not None and os.path.isdir(
                model_args.model_name_or_path):
            checkpoint = model_args.model_name_or_path
        else:
            checkpoint = None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)  #查看内存分配
        # perplexity = math.exp(metrics["eval_loss"])
        # metrics["perplexity"] = perplexity
        trainer.save_metrics("train", metrics)
        
    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        for eval_dataset_name, eval_dataset in merge_dataset.items():
            metrics = trainer.evaluate(
                eval_dataset=eval_dataset,
                metric_key_prefix=f"eval_{eval_dataset_name}",   #重命名
            )
            if is_main_process(training_args.local_rank):
                print("eval", metrics)
            # logger.info("eval", metrics)
            trainer.log_metrics("eval", metrics)
            # trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
    
