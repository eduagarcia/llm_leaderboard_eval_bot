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
        start_time=None
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
            "all": {}
        },
        "config_tasks": {},
        "versions": {
            "all": 0
        },
        "summary_tasks": {},
        "summary_general": {}
    }
    all_results = defaultdict(list)
    for task in Tasks:
        task = task.value
        result = evaluate(
            model=model,
            model_args=model_args,
            tasks=",".join(task.task_list),
            num_fewshot=task.few_shot,
            limit=task.limit,
            batch_size='auto',
            max_batch_size=128,
            log_samples=True,
            show_config=True,
            output_path=os.path.join(output_path, f"raw_{start_time}", task.benchmark)
        )
        scores = result["results"]
        for subtask in scores:
            new_task_scores = {k.split(',')[0]:v for k,v in scores[subtask].items() if 'k' != 'alias'}
            for score_name in new_task_scores:
                score = new_task_scores[score_name]
                if not (isinstance(score, float) or isinstance(score, int)):
                    continue
                all_results[score_name].append(score)
            subtask_name = f"harness|{task.benchmark}|{subtask}"
            subtask_full_name = f"{subtask_name}|{task.limit}|{task.few_shot}"
            result_tasks['results'][subtask_full_name] = new_task_scores
            result_tasks['config_tasks'][subtask_name] = "LM Harness task"
            result_tasks['versions'][subtask_full_name] = result["configs"][subtask]["metadata"]["version"]
            result_tasks['summary_tasks'][subtask_full_name] = {}
        gc.collect()
        torch.cuda.empty_cache()

    for score_name in all_results:
        result_tasks['results']['all'][score_name] = sum(all_results[score_name])/len(all_results[score_name])

    result_tasks['config_general']['end_time'] = time.time()
    result_tasks['config_general']['total_evaluation_time_seconds'] = result_tasks['config_general']['end_time'] - result_tasks['config_general']['start_time']
    with open(os.path.join(output_path, f"results_{start_time}.json"), "w", encoding='utf-8') as f:
        json.dump(result_tasks, f, indent=4, ensure_ascii=False)
    return result_tasks

if __name__ == "__main__":
    #print(run_eval_on_model(
    #    model_args="pretrained=meta-llama/Llama-2-7b-hf,revision=main,dtype=float16"
    #))
    print(run_eval_on_model(
        model_args="pretrained=gpt2,revision=main,dtype=float16,trust_remote_code=True"
    ))