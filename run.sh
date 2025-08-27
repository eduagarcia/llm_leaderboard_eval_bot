#!/bin/bash

export CUDA_VISIBLE_DEVICES=4
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_CACHE=/workspace/datasets/hf_cache/
export HF_CACHE=/workspace/datasets/hf_cache/
export HF_HOME=/workspace/datasets/hf_cache/
export TRANSFORMERS_CACHE=/workspace/datasets/hf_cache/
export NUMEXPR_MAX_THREADS=32
export TRUST_REMOTE_CODE="True"

#PYTHON_BIN=/root/miniconda3/envs/torch21/bin/
PYTHON_BIN=python

cd /workspace/repos/llm_leaderboard/llm_leaderboard_eval_bot
#$PYTHON_BIN evaluate_llms.py --download_queue_size 1 --max_model_size 50 --eval_engine vllm
$PYTHON_BIN evaluate_llms.py --download_queue_size 3 --max_model_size 40
#/root/miniconda3/envs/torch21/bin/python evaluate_llms.py --gpu_ids=0,1 --download_queue_size 3 --parallelize True --not_allow_4bit True

#python evaluate_llms.py --download_queue_size 1 --max_model_size 50