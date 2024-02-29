#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --mem=62gb
#SBATCH --output=log/%j.out                              
#SBATCH --error=log/%j.out
#SBATCH --job-name=dpo
#SBATCH --nodes=1
#SBATCH --requeue
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=mhong
module list

# Benchmark info
echo "TIMING - Starting jupyter at: $(date)"

source activate dpo
wandb login 3dbaa1026adab988dca53f5bebe8eff91ed0d378

nvidia-smi
unset CUDA_VISIBLE_DEVICES
which python3
which wandb
echo "Job is starting on $(hostname)"

cd ~/jobsubmit/dpo_trl || exit

# python test.py
accelerate launch --config_file=deepspeed_zero3.yaml dpo.py \
    --model_name_or_path=daryl149/llama-2-7b-chat-hf \
    --per_device_train_batch_size 1 \
    --max_steps 1000 \
    --learning_rate 1e-3 \
    --gradient_accumulation_steps 1 \
    --logging_steps 10 \
    --eval_steps 500 \
    --output_dir="dpo_anthropic_hh" \
    --optim rmsprop \
    --warmup_steps 150 \
    --report_to wandb \
    --logging_first_step \
    --no_remove_unused_columns \
    --use_peft \
    --lora_r=16 \
    --lora_alpha=16 \
    --bf16

exit