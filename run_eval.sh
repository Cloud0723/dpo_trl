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

# source activate dpo
source activate eval
# wandb login 473d9c02a8828637658f0cad6a187faf250ab481

nvidia-smi
unset CUDA_VISIBLE_DEVICES
which python3
# which wandb
echo "Job is starting on $(hostname)"

cd ~/dpo_trl || exit

# CUDA_VISIBLE_DEVICES=0 python generation.py
set -x
set -e
epoch=40
file_name="IRLHF_1B_5epoch"
# --model ./model_llama/huggyllama_test_5e-7/checkpoint-${j} \
####eval part########
# for ((i=0; i<epoch; i++))
# do
#     j=$(((i+1) * 500))
#     accelerate launch --main_process_port=2950 generate.py \
#         --model ./model_llama/ultrahh_sft_1e-5/checkpoint-${j} \
#         --input_dir ./data/ultra_hh.json \
#         --batch_size 2 \
#         --frac_len 800 \
#         --data_frac 0 \
#         --output_dir ./eval_data/ultrahh_sft_1e-5_${j}.json
# done

#mistralai/Mistral-7B-v0.1 \

# for ((i=0; i<epoch; i++))
# do
#     j=$(((i+1) * 500))
#     accelerate launch --main_process_port=2950 eval_openmb.py \
#         --eval_data_file ./eval_data/llama_7B_${j}.json \
#         --output_dir ./eval_data/llama_7B_${j}_reward.json
# done

for ((i=0; i<epoch; i++))
do
    j=$(((i+1) * 500))
    accelerate launch --main_process_port=2950 eval_openmb.py \
        --eval_data_file ./eval_data/ultrahh_sft_1e-5_${j}.json \
        --output_dir ./eval_data/ultrahh_sft_1e-5_${j}_reward.json
done

exit