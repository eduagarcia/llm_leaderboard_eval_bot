from hf_util import download_requests_repo, update_status_requests, upload_results, free_up_cache, download_model, upload_raw_results, delete_model_from_cache
from run_eval import run_eval_on_model
from datetime import datetime, timezone
from eval_queue import get_eval_results_df, update_eval_version
import os
from envs import EVAL_REQUESTS_PATH, EVAL_RESULTS_PATH, EVAL_VERSION, TRUST_REMOTE_CODE, RAW_RESULTS_REPO
from tasks import Tasks
import traceback
import json
import time
import logging
from threading import Thread, Lock, RLock
import torch
import gc
import argparse
from queue import Queue
from itertools import count

def run_request(
        model
    ):
    commit_hash = None
    print(model)
    download_requests_repo()
    requests_df = get_eval_results_df()
    request = requests_df[requests_df['model'] == model].iloc[0]
    with open(request["filepath"], encoding='utf-8') as fp:
        request_data = json.load(fp)
    model_id = request['model_id']
    print(model_id)
    
    print(request_data)

    result_folder_path = os.path.join(EVAL_RESULTS_PATH, request_data['model'])
    if not os.path.exists(result_folder_path):
        print(f"{result_folder_path} folder does not exists")
        return
    filenames = sorted([i for i in os.listdir(result_folder_path) if i.endswith('.json')])
    if len(filenames) == 0:
        print(f"result file in {result_folder_path} does not exists")
        return
    filename = filenames[-1]
    result_path = os.path.join(result_folder_path, filename)
    print(result_path)

    with open(result_path, 'r', encoding='utf8') as f:
        results = json.load(f)

    results["config_general"]["model_name"] = request_data["model"]
    results["config_general"]["model_dtype"] = request_data["precision"]
    results["config_general"]["job_id"] = request_data["job_id"]
    results["config_general"]["model_id"] = model_id
    results["config_general"]["model_base_model"] = request_data["base_model"]
    results["config_general"]["model_weight_type"] = request_data["weight_type"]
    results["config_general"]["model_revision"] = request_data["revision"]
    results["config_general"]["model_private"] = request_data["private"]
    results["config_general"]["model_type"] = request_data["model_type"]
    results["config_general"]["model_architectures"] = request_data["architectures"]
    results["config_general"]["submitted_time"] = request_data["submitted_time"]
    results["config_general"]["lm_eval_model_type"] = "huggingface"
    results["config_general"]["eval_version"] = EVAL_VERSION

    upload_results(request_data["model"], results)

    request_data["status"] = "FINISHED"
    request_data["eval_version"] = EVAL_VERSION

    #update new results
    if "result_metrics" not in request_data:
        request_data["result_metrics"] = {}
    for benchmark in results["results"]["all_grouped"]:
        request_data["result_metrics"][benchmark] = results["results"]["all_grouped"][benchmark]
    
    #clean up old benchmarks
    valid_task_list = [task.value.benchmark for task in Tasks]
    for benchmark in list(request_data["result_metrics"].keys()):
        if benchmark not in valid_task_list:
            del request_data["result_metrics"][benchmark]

    #calculate new average
    request_data["result_metrics_average"]= sum(request_data["result_metrics"].values())/len(request_data["result_metrics"])
    npm = []
    for task in Tasks:
        npm.append((request_data["result_metrics"][task.value.benchmark]-(task.value.baseline/100)) / (1.0-(task.value.baseline/100)))
    request_data["result_metrics_npm"] = sum(npm)/len(npm)
    update_status_requests(model_id, request_data)

    if RAW_RESULTS_REPO is not None:
        upload_raw_results(request_data['model'])
    
    if commit_hash is None:
        commit_hash = results["config_general"]["model_sha"]
    delete_model_from_cache(commit_hash)

if __name__ == "__main__":
    import sys
    run_request(sys.argv[-1])