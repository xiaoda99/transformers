import logging
import random
import os
import sys
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
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments, default_data_collator, set_seed,
    HfArgumentParser, TrainingArguments, TransfoXLConfig)


# Setup logging
logger = logging.getLogger(__name__)


# os.environ["WANDB_DISABLED"] = "true"
def main():
    # load arguments
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses(
        )

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(
            training_args.output_dir
    ) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(
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
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info(f"Model_settings: {model_args}")
    logger.info(f"data_settings: {data_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load model and tokenizer
    cache_dir = '/nas/lxy/.cache/torch/transformers/'
    # proxies = {'http': '192.168.50.1:1081'} 

    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, cache_dir=cache_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, cache_dir=cache_dir)
    
    tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer))
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id

    print(tokenizer.eos_token, tokenizer.eos_token_id)

    if data_args.block_size is None:
        logger.info("Setting `block_size` 1024 since it was not set")
        tokenizer.model_max_length = 1024
    else:
        logger.info("Setting `block_size` to %d", data_args.block_size)
        tokenizer.model_max_length = data_args.block_size
    


    # Load train datasets
    train_dataset = TasksDataset(
        data_args.train_file,
        tokenizer,
        'train',
        0.8,
        max_length=data_args.max_input_length,
        k_shot = data_args.k_shot
    )
    # 传入两个数据集
    train_eval_dataset = TasksDataset(
        data_args.train_file,
        tokenizer,
        'train',
        0.8,
        max_length=data_args.max_input_length,
        k_shot = data_args.k_shot
    )
    eval_dataset = TasksDataset(
        data_args.train_file,
        tokenizer,
        'eval',
        0.8,
        max_length=data_args.max_input_length,
        k_shot = data_args.k_shot
    )
    # # Set up the metric
    # rouge = evaluate.load("rouge")
    # 这里可以计算准确率
    def compute_metrics(eval_preds):
        labels_ids = eval_preds.label_ids[:, -1].contiguous()
        pred_ids = eval_preds.predictions[:, 1:].contiguous()

        assert labels_ids.size() == pred_ids.size(), f'{labels_ids.size()} != {pred_ids.size()}'
        
        label_mask = (labels_ids != -100)

        accuracy = torch.einsum('bi->b', (pred_ids == labels_ids ) * label_mask) \
            / torch.einsum('bi->b', label_mask)

    
        # pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        # label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
        # result = rouge.compute(predictions=pred_str, references=label_str)
        return accuracy, accuracy.mean()

    # Create a preprocessing function to extract out the proper logits from the model output
    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits.argmax(dim=-1)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset={'train_eval_dataset': train_eval_dataset, 'eval_dataset': eval_dataset},
        compute_metrics=compute_metrics,
        data_collator=default_data_collator,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
    # deepspeed_engine, optimizer, lr_scheduler = deepspeed_init(
    #                 trainer,
    #                 num_training_steps=training_args.max_steps,
    #                 resume_from_checkpoint=trainer.state.best_model_checkpoint,
    #         )
    # trainer.model = deepspeed_engine.module
    # trainer.model_wrapped = deepspeed_engine
    # trainer.deepspeed = deepspeed_engine
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
        trainer.log_metrics("train", metrics)
        # perplexity = math.exp(metrics["eval_loss"])
        # metrics["perplexity"] = perplexity
        trainer.save_metrics("train", metrics)
        
    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()
        perplexity = math.exp(metrics["eval_loss"])
        metrics["perplexity"] = perplexity
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
    
