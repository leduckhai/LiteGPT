#!/bin/bash

#SBATCH --job-name=testcode       # create a short name for your job
#SBATCH --output=./examples/litegpt/litegpt_train_output/Eval-audio-whisper-biomedclip_llama-vin-JOB_ID_%j-%N.log # create output file

#SBATCH --nodes=1                  # node count
#SBATCH --ntasks-per-node=1        #CPUS per node (how many cpu to use withinin 1 node)
#SBATCH --mem=100G
#SBATCH --time=500:00:00               # total run time limit (HH:MM:SS)
#SBATCH --partition=gpu2 --gres=gpu  # number of gpus per node

echo "Job ID: $SLURM_JOBID" 
echo "Node names: $SLURM_JOB_NODELIST"
echo "Notes: Test code"
# pip install --user iopath nltk
# export FORCE_CUDA="1"
torchrun --nproc_per_node 2 setup.py examples/litegpt/evaluate.py \
      --cfg-path examples/litegpt/eval_configs/eval_audio.yaml\
      --eval-dataset audio_val
