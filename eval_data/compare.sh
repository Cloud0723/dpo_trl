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
# wandb login 473d9c02a8828637658f0cad6a187faf250ab481

nvidia-smi
unset CUDA_VISIBLE_DEVICES
which python3
# which wandb
echo "Job is starting on $(hostname)"

cd ~/dpo_trl/eval_data || exit



export OPENAI_API_KEY=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  # set the OpenAI API key
alpaca_eval --model_outputs './Base_output.json' \
  --reference_outputs './DPO_output.json' 
#   --annotators_config 'alpaca_eval_gpt4_turbo_fn' 

exit
