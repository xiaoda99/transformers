#!/bin/bash
# export CUDA_VISIBLE_DEVICES=5,6 && deepspeed --num_gpus=2 train_gptj_summarize.py
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# deepspeed train_gptj_summarize.py
# deepspeed --num_gpus=2 
#--overwrite_output_dir    --do_train \     # --load_best_model_at_end \     --half_precision_backend \     --gradient_checkpointing \ --do_train \ 
# --overwrite_output_dir \
#--overwrite_output_dir \
deepspeed   train_model_tasks.py \
    --output_dir "tasks_test" \
    --do_train \
    --do_eval \
    --evaluation_strategy "steps" \
    --eval_steps 2 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --eval_accumulation_steps 1 \
    --learning_rate 1e-5 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --warmup_steps 100 \
    --num_train_epochs 3 \
    --logging_steps 1 \
    --save_strategy "steps" \
    --save_steps 200 \
    --fp16 \
    --deepspeed /nas/xd/projects/transformers/notebooks/lxy_train/ds_config_gptj.json \
    --train_file /nas/xd/projects/transformers/notebooks/lxy_train/task_datasets.pickle \
    --block_size 1024 \
    --max_input_length 512 \
    --validation_split_percentage 80 \
    --k_shot 1 \
    --model_name_or_path EleutherAI/gpt-j-6B \
    --tokenizer_name EleutherAI/gpt-j-6B \
    --overwrite_output_dir \
    | tee train0511.log
