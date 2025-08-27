#!/bin/bash

# --- Load Saved Environment ---
ENV_FILE="/workspace/torch21.env"

if [ ! -f "$ENV_FILE" ]; then
    echo "Error: Environment file not found at $ENV_FILE"
    echo "Please create it first in a working interactive shell:"
    echo "conda activate torch21 && env | sort > $ENV_FILE"
    exit 1
fi

echo "Loading environment variables from $ENV_FILE"
# Read each line (KEY=VALUE) from the file and export it
# Using process substitution and 'export -p' might be slightly safer,
# but simple export should work for typical 'env' output.
while IFS= read -r line; do
    # Skip empty lines if any
    if [ -n "$line" ]; then
        export "$line"
    fi
done < "$ENV_FILE"
echo "Environment loaded."
# Optional: Log a few key variables to confirm they were loaded
echo "PATH after load: $PATH"
echo "LD_LIBRARY_PATH after load: $LD_LIBRARY_PATH"
# --- End Load Environment ---

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

    if [ "$used_mem" -lt 25000 ] && [ "$gpu_usage" -lt 15 ]
    then
        echo "GPU $gpu_id memory usage less than 25GB and GPU usage less than 15%."
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

echo "Current PATH before python run: $PATH" # Log for debugging
echo "Current LD_LIBRARY_PATH before python run: $LD_LIBRARY_PATH" # Log for debugging

cd /workspace/repos/llm_leaderboard/llm_leaderboard_eval_bot
/root/miniconda3/envs/torch21/bin/python evaluate_llms.py --download_queue_size 0 --memory_fraction_per_gpu $memory_fraction --force_kill_time 07:00 --max_model_size $max_model_size --not_allow_4bits True

status=$? # Capture exit code

if [ $status -eq 0 ]; then
    echo "Python script finished successfully via conda run."
else
    echo "Python script failed with exit code $status when run via conda run."
fi

exit $status