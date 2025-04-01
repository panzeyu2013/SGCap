#!/bin/bash

#SBATCH --job-name="Zero-Shot"
#SBATCH --cpus-per-task=16

eval "$(conda shell.bash hook)"
conda activate GCap

export MODEL_LOAD='../checkpoints'
export CONFIG_LOAD='../checkpoints'

export TOKENIZER_LOAD='gpt2'
export LANGUAGE_MODEL='gpt2'

export CACHE_DIR='~/.cache/huggingface/hub/'

current_time=$(date +"%Y-%m-%d_%H-%M-%S")
output_dir="../checkpoints/$current_time"
mkdir -p "$output_dir"

SCRIPT_PATH=$(realpath "$0")
cp "$SCRIPT_PATH" "$output_dir"

export DATASET_NAME=""
export topk=5

python ../train.py \
    --load_from_config False \
    --load_from_pretrained False \
    --model_name_or_path $MODEL_LOAD \
    --config_name_or_path $CONFIG_LOAD \
    --language_model $LANGUAGE_MODEL \
    --freeze_language False \
    \
    --need_tokenizer True \
    --tokenizer_from_pretrained False \
    --tokenizer_name_or_path $TOKENIZER_LOAD \
    --tokenizer_max_length 512 \
    --cache_dir $CACHE_DIR \
    --version v1 \
    \
    --dataset_name $DATASET_NAME \
    --sigma 0.5\
    --noise 1 \
    --p 1 \
    --neighbor_noise 1 \
    --data_root "../data/extract" \
    --data_split "all" \
    --k_neighbors $topk \
    --max_length 48 \
    --percent 1 \
    \
    --optim "adamw_torch" \
    --load_best_model_at_end False \
    --adam_beta1 0.9 \
    --adam_beta2 0.999 \
    --adam_epsilon 1e-08 \
    --bf16 True \
    --dataloader_drop_last False \
    --dataloader_num_workers 16 \
    --dataloader_pin_memory True \
    --evaluation_strategy "epoch" \
    --eval_steps 1 \
    --gradient_checkpointing False \
    --group_by_length False \
    --learning_rate 1e-4 \
    --logging_steps 1 \
    --logging_strategy "steps" \
    --lr_scheduler_type "cosine" \
    --num_train_epochs 5 \
    --output_dir $output_dir \
    --report_to wandb \
    --resume_from_checkpoint None \
    --save_strategy "epoch" \
    --save_steps 30 \
    --save_total_limit 1 \
    --metric_for_best_model "loss" \
    --seed 42 \
    --gradient_accumulation_steps 1 \
    --per_device_train_batch_size 256 \
    --per_device_eval_batch_size 64 \
    --eval_accumulation_steps 8 \
    --weight_decay 0 \
    --warmup_ratio 0.03

python ../evaluate_caption.py \
    --data_root ../data \
    --dataset_name $DATASET_NAME \
    --caption_from $DATASET_NAME \
    --checkpoint $output_dir \
    --result_file "$output_dir/result.json" \
    --topk $topk \
    --sampling "uniform"