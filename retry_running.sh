#!/bin/bash
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_CACHE=/workspace/datasets/hf_cache/
export HF_CACHE=/workspace/datasets/hf_cache/
export HF_HOME=/workspace/datasets/hf_cache/
export TRANSFORMERS_CACHE=/workspace/datasets/hf_cache/

cd /workspace/repos/llm_leaderboard/llm_leaderboard_eval_bot
/root/miniconda3/envs/torch21/bin/python retry_failed.py RUNNING24