#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --mem=62gb
#SBATCH --output=log/%j.out                              
#SBATCH --error=log/%j.out
#SBATCH --job-name=spin
#SBATCH --nodes=1
#SBATCH --requeue
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=mhong
module list

# Benchmark info
echo "TIMING - Starting jupyter at: $(date)"

source activate spin
wandb login 3dbaa1026adab988dca53f5bebe8eff91ed0d378
export 'WANDB_ENTITY=jasonljx96'
export 'WANDB_PROJECT=self_play'

nvidia-smi
unset CUDA_VISIBLE_DEVICES
which python3
which wandb
echo "Job is starting on $(hostname)"

cd ~/jobsubmit/dpo_trl || exit

# python test.py
accelerate launch --main_process_port 29505 \
    --config_file=deepspeed_zero3.yaml dpo.py \
    --model_name_or_path=alignment-handbook/zephyr-7b-sft-full \
    --per_device_train_batch_size 1 \
    --max_steps 1000 \
    --learning_rate 5e-7 \
    --gradient_accumulation_steps 1 \
    --logging_steps 10 \
    --eval_steps 500 \
    --output_dir="./model_llama/huggyllama_test_5e-7" \
    --dataset "spin"\
    --optim adamw \
    --warmup_steps 150 \
    --report_to wandb \
    --logging_first_step \
    --no_remove_unused_columns \
    --use_peft \
    --lora_r=16 \
    --lora_alpha=16 \
    --bf16

exit