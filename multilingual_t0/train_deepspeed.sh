#!/bin/bash

# Request half an hour of runtime:
#SBATCH --time=14-23:59:00

# Ask for the GPU partition and 1 GPU
#SBATCH --partition=gpu-he --gres=gpu:2
#SBATCH --constraint=v100

# Default resources are 1 core with 2.8GB of memory.
#SBATCH --ntasks=2

# Use more memory (10GB) (CPU RAM):
#SBATCH --mem=200g

# Specify a job name:
#SBATCH -J exp-001-mt0_deepspeed_lm_adapt

# Specify an output file
#SBATCH -o /users/zyong2/data/zyong2/mt0/logs/log-001/mt0_deepspeed_lm_adapt.out
#SBATCH -e /users/zyong2/data/zyong2/mt0/logs/log-001/mt0_deepspeed_lm_adapt.err

set -a # automatically export all variables
source ~/.env
set +a

module load python/3.7.4
source $MT0/env_mT0/bin/activate

module load cuda/11.1.1
module load gcc/10.2
nvcc --version
# python3 -m deepspeed.env_report

# https://github.com/huggingface/transformers/issues/8771#issuecomment-886233087

GPU_NODES=2
DATASET_CACHE_DIR="/users/zyong2/data/zyong2/mt0/data/external/mt0/mC4_download/data"
MODEL_NAME="google/mt5-xl"
CACHE_DIR="/users/zyong2/data/zyong2/huggingface/mt5_xl"
TRAIN_BSZ=1
GRAD_ACC=16
OUTPUT_DIR="/users/zyong2/data/zyong2/mt0/data/processed/001/mt5_xl"
MAX_STEPS=$((100000*1024/($GPU_NODES*$TRAIN_BSZ*$GRAD_ACC)))
LOGGING_DIR="/users/zyong2/data/zyong2/mt0/data/processed/001/runs/mt5_xl_lm_adaptation"
LOGGING_STEPS=1000
SAVE_STEPS=1000
DS_CONFIG="/users/zyong2/data/zyong2/mt0/data/external/mt0/multilingual_t0/ds_config_zero3.json"

deepspeed \
/users/zyong2/data/zyong2/mt0/data/external/mt0/multilingual_t0/main.py \
--model_name_or_path $MODEL_NAME \
--dataset_cache_dir $DATASET_CACHE_DIR \
--cache_dir $CACHE_DIR \
--dataset_name "mc4" \
--max_input_length 1024 \
--max_target_length 256 \
--do_train \
--preprocessing_num_workers 4 \
--per_device_train_batch_size $TRAIN_BSZ \
--gradient_accumulation $GRAD_ACC \
--overwrite_output_dir \
--output_dir $OUTPUT_DIR \
--max_steps $MAX_STEPS \
--logging_dir $LOGGING_DIR \
--logging_strategy "steps" \
--logging_steps $LOGGING_STEPS \
--save_strategy "steps" \
--save_steps $SAVE_STEPS \
--report_to "tensorboard" \
--deepspeed $DS_CONFIG