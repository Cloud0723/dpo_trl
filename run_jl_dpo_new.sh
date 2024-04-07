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
export 'WANDB_PROJECT=self_play_dpo'

nvidia-smi
unset CUDA_VISIBLE_DEVICES
which python3
which wandb
echo "Job is starting on $(hostname)"

cd ~/jobsubmit/dpo_trl || exit

# Possible model names:
# mistralai/Mistral-7B-v0.1
# alignment-handbook/zephyr-7b-sft-full
# huggyllama/llama-7b
# meta-llama/Llama-2-7b
# facebook/opt-1.3b

# --use_peft \
# --lora_r=16 \
# --lora_alpha=16

accelerate launch --main_process_port 29502 \
    --config_file=deepspeed_zero3.yaml dpo.py \
    --model_name_or_path=facebook/opt-1.3b \
    --per_device_train_batch_size 2 \
    --max_steps 20000 \
    --learning_rate 5e-7 \
    --gradient_accumulation_steps 1 \
    --logging_steps 10 \
    --eval_steps 500 \
    --output_dir="./results/facebook/opt-1.3b" \
    --dataset "spin" \
    --trainer "new_dpo" \
    --optim rmsprop \
    --warmup_steps 150 \
    --report_to wandb \
    --logging_first_step \
    --no_remove_unused_columns \
    --bf16

exit