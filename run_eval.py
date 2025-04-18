from tasks import Tasks
from lm_eval_util import evaluate
import os
from envs import EVAL_RESULTS_PATH
import time
import json
from collections import defaultdict
from datetime import datetime, timezone
import gc
import torch

def run_eval_on_model(
        model="huggingface",
        model_args="pretrained=EleutherAI/pythia-14m",
        output_path=os.path.join(EVAL_RESULTS_PATH+"_test", "default"),
        start_time=None,
        batch_size=None
):  
    if start_time is None:
        start_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S.%f")
    #os.makedirs(os.path.join(output_path, "raw"), exist_ok=True)
    result_tasks = {
        "config_general": {
            "start_date": start_time,
            "start_time": time.time()
        },
        "results": {
            "all_grouped_average": 0,
            "all_grouped_npm": 0,
            "all_grouped": {},
            "all": {},
            
        },
        "config_tasks": {},
        "versions": {
            "all": 0
        },
        "summary_tasks": {},
        "summary_general": {}
    }
    all_results = defaultdict(list)
    task_list = []
    few_shot_list = []
    limit_list = []
    task_order = []
    for task in Tasks:
        task = task.value
        for subtask in task.task_list:
            task_order.append(task)
            task_list.append(subtask)
            few_shot_list.append(str(task.few_shot))
            limit_list.append(str(task.limit))

    output_raw_path = os.path.join(output_path, f"raw_{start_time}")

    max_batch_size = 64
    if batch_size is not None:
        max_batch_size = batch_size

    if ",max_length=" not in model_args:
        model_args = model_args + ",starting_max_length=2560"

    result = evaluate(
        model=model,
        model_args=model_args,
        tasks=",".join(task_list),
        num_fewshot=",".join(few_shot_list),
        limit=",".join(limit_list),
        batch_size='auto',
        max_batch_size=max_batch_size,
        log_samples=True,
        show_config=True,
        output_path=output_raw_path,
        bootstrap_iters=0,
    )
    
    scores = result["results"]
    npm = []
    for task in Tasks:
        task = task.value
        scores_grouped = []
        for subtask in task.task_list:
            new_task_scores = {k:v for k,v in scores[subtask].items() if k != 'alias'}
            main_score = new_task_scores[task.metric + ',all']

            new_task_scores['main_score'] = main_score
            scores_grouped.append(main_score)

            subtask_name = f"harness|{task.benchmark}|{subtask}"
            subtask_full_name = f"{subtask_name}|{task.limit}|{task.few_shot}"

            result_tasks['results'][subtask_full_name] = new_task_scores
            result_tasks['config_tasks'][subtask_name] = "LM Harness task"
            result_tasks['versions'][subtask_name] = result["configs"][subtask]["metadata"]["version"]
            result_tasks['summary_tasks'][subtask_full_name] = result["task_model_meta"][subtask]
            result_tasks['results']['all'][subtask_full_name] = main_score
        grouped_score = sum(scores_grouped)/len(scores_grouped)
        result_tasks['results']['all_grouped'][task.benchmark] = grouped_score
        npm.append((grouped_score-(task.baseline/100)) / (1.0-(task.baseline/100)))

    result_tasks['results']['all_grouped_average'] = sum(result_tasks['results']['all_grouped'].values())/len(result_tasks['results']['all_grouped'].values())
    result_tasks['results']['all_grouped_npm'] = sum(npm)/len(npm)
    
    result_tasks['config_general']['end_time'] = time.time()
    result_tasks['config_general']['total_evaluation_time_seconds'] = result_tasks['config_general']['end_time'] - result_tasks['config_general']['start_time']

    keys = ["truncated", "non_truncated", "padded", "non_padded", "fewshots_truncated"]
    for k in keys:
        if k in result['model_meta']:
            result_tasks["summary_general"][k] = result['model_meta'].pop(k)

    result_tasks['config_general'].update(result['model_meta'])

    with open(os.path.join(output_path, f"results_{start_time}.json"), "w", encoding='utf-8') as f:
        json.dump(result_tasks, f, indent=4, ensure_ascii=False)

    #gc.collect()
    #torch.cuda.empty_cache()
    
    return result_tasks

if __name__ == "__main__":
    print(run_eval_on_model(
        model_args="pretrained=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,max_length=4098,enforce_eager=True",
        model="vllm"
    ))
    #print(run_eval_on_model(
    #    model_args="pretrained=gpt2,revision=main,dtype=float16,trust_remote_code=True"
    #))