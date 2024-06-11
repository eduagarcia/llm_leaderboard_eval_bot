#!/bin/bash

gpu_id=$1

if [ -z "$gpu_id" ]; then
    echo "Usage: $0 <GPU_ID>"
    exit 1
fi

for i in {1..5}
do
    gpu_info=$(nvidia-smi --query-gpu=index,memory.total,memory.used,utilization.gpu --format=csv,noheader | grep "$gpu_id,")

    total_mem=$(echo "$gpu_info" | awk -F ',' '{print $2}' | sed 's/ MiB//')
    used_mem=$(echo "$gpu_info" | awk -F ',' '{print $3}' | sed 's/ MiB//')
    gpu_usage=$(echo "$gpu_info" | awk -F ',' '{print $4}' | sed 's/ %//')

    if [ "$used_mem" -lt 40000 ] && [ "$gpu_usage" -lt 3 ]
    then
        echo "GPU $gpu_id memory usage less than 40GB and GPU usage less than 3%."
    else
        echo "GPU $gpu_id does not meet the specified conditions. Mem $used_mem/$total_mem usage $gpu_usage"
        exit 1
    fi

    if [ $i -ne 5 ]
    then
        sleep 60
    fi
done

max_model_size=11
memory_fraction=0.5

if [ "$used_mem" -lt 15000 ]
then
    max_model_size=21
    memory_fraction=0.8
fi

if [ "$used_mem" -lt 10000 ]
then
    max_model_size=29
    memory_fraction=0.85
fi

export CUDA_VISIBLE_DEVICES=$gpu_id
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_CACHE=/workspace/datasets/hf_cache/
export HF_CACHE=/workspace/datasets/hf_cache/
export HF_HOME=/workspace/datasets/hf_cache/
export TRANSFORMERS_CACHE=/workspace/datasets/hf_cache/
export NUMEXPR_MAX_THREADS=32
export TRUST_REMOTE_CODE="True"

cd /workspace/repos/llm_leaderboard/llm_leaderboard_eval_bot
/root/miniconda3/envs/torch21/bin/python evaluate_llms.py --download_queue_size 0 --memory_fraction_per_gpu $memory_fraction --force_kill_time 07:00 --max_model_size $max_model_size --not_allow_4bits True