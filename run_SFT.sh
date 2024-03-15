#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --mem=62gb
#SBATCH --output=log/%j.out                              
#SBATCH --error=log/%j.out
#SBATCH --job-name=dpo
#SBATCH --requeue
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=mhong
module list

# Benchmark info
echo "TIMING - Starting jupyter at: $(date)"

source activate dpo
wandb login 473d9c02a8828637658f0cad6a187faf250ab481

nvidia-smi
unset CUDA_VISIBLE_DEVICES
which python3
which wandb
echo "Job is starting on $(hostname)"

cd ~/dpo_trl || exit

# python test.py
accelerate launch --main_process_port 29506 \
    --config_file=deepspeed_zero3.yaml sft.py \
    --model_name_or_path=mistralai/Mistral-7B-v0.1 \
    --report_to="wandb" \
    --learning_rate=1.41e-5 \
    --per_device_train_batch_size=16 \ #16 is ok for 7B model
    --gradient_accumulation_steps=1 \
    --output_dir="model_Mistral/ultra_hh_sft_2" \
    --logging_steps=1 \
    --num_train_epochs=2 \
    --max_steps=-1 \
    --gradient_checkpointing \
    --use_peft \
    --lora_r=64 \
    --bf16 \
    --lora_alpha=16

exit