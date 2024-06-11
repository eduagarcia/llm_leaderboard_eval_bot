#!/bin/bash

export CUDA_VISIBLE_DEVICES=2,5
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_CACHE=/workspace/datasets/hf_cache/
export HF_CACHE=/workspace/datasets/hf_cache/
export HF_HOME=/workspace/datasets/hf_cache/
export TRANSFORMERS_CACHE=/workspace/datasets/hf_cache/
export NUMEXPR_MAX_THREADS=32
export TRUST_REMOTE_CODE="True"

cd /workspace/repos/llm_leaderboard/llm_leaderboard_eval_bot
/root/miniconda3/envs/torch21/bin/python evaluate_llms.py --gpu_ids=0,1 --parallelize True