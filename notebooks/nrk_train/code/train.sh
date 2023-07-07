#!/bin/bash
# export CUDA_VISIBLE_DEVICES=5,6 && deepspeed --num_gpus=2 train_gptj_summarize.py
export CUDA_VISIBLE_DEVICES=0,2,3,8
# deepspeed train_gptj_summarize.py
# deepspeed --num_gpus=2 
#--overwrite_output_dir    --do_train \     # --load_best_model_at_end \     --half_precision_backend \     --gradient_checkpointing \ --do_train \ 
# --overwrite_output_dir \
#--overwrite_output_dir \
deepspeed   train_model_tasks.py \
    --output_dir "train_tasks_0701_28_2" \
    --do_train \
    --do_eval \
    --evaluation_strategy "steps" \
    --eval_steps 200 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --eval_accumulation_steps 1 \
    --learning_rate 1e-5 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --warmup_steps 100 \
    --num_train_epochs 2 \
    --gradient_checkpointing \
    --logging_steps 5 \
    --save_strategy "steps" \
    --save_steps 200 \
    --fp16 \
    --deepspeed /nas/xd/projects/transformers/notebooks/nrk_train/code/ds_config_gptj.json \
    --train_file /nas/xd/projects/transformers/notebooks/nrk_train/data/task_datasets.json \
    --block_size 1024 \
    --max_input_length 256 \
    --validation_split_percentage 20 \
    --k_shot 1 \
    --model_name_or_path /nas/xd/projects/transformers/notebooks/nrk_train/code/train_tasks_0701_28 \
    --tokenizer_name EleutherAI/gpt-j-6B \
    --eval_task_file /nas/xd/projects/transformers/notebooks/nrk_train/data/eval_task_file.json \
    --overwrite_output_dir \
    | tee train0701_28_2.log
