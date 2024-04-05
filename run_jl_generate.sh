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

source activate dpo
# wandb login 473d9c02a8828637658f0cad6a187faf250ab481

nvidia-smi
unset CUDA_VISIBLE_DEVICES
which python3
# which wandb
echo "Job is starting on $(hostname)"

cd ~/jobsubmit/dpo_trl || exit

CUDA_VISIBLE_DEVICES=0 python generation_spin.py \
    --model 'UCLA-AGI/zephyr-7b-sft-full-SPIN-iter0'\
    --input_dir 'UCLA-AGI/SPIN_iter0'\
    --output_dir 'data/spin_generated/iter0.json'\
    --frac_len 800

exit