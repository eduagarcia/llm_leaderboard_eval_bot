from hf_util import download_requests_repo, update_status_requests, upload_results, free_up_cache, download_model
from run_eval import run_eval_on_model
from datetime import datetime, timezone
from eval_queue import get_eval_results_df, update_eval_version
import os
from envs import EVAL_REQUESTS_PATH, EVAL_RESULTS_PATH, EVAL_VERSION, TRUST_REMOTE_CODE
import traceback
import json
import time
import logging
from threading import Thread, RLock
import torch
import gc

def run_request(model_id, request_data, job_id=None, commit_hash=None):
    start_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S.%f")
    request_data["status"] = "RUNNING"
    request_data["job_id"] = job_id
    request_data["job_start_time"] = start_time

    update_status_requests(model_id, request_data)

    model_args = request_data['model']
    lm_eval_model_type = "huggingface" if "lm_eval_model_type" not in request_data else request_data["lm_eval_model_type"]
    
    if lm_eval_model_type == "huggingface":
        model_args = f"pretrained={request_data['model']}"
        if request_data["weight_type"] == "Adapter":
            model_args = f"pretrained={request_data['base_model']},peft={request_data['model']}"
        elif request_data["weight_type"] == "Delta":
            raise Exception("Delta weights are not supported yet")
        
        extra_args=",dtype=float16"
        if request_data["precision"] == "bfloat16":
            extra_args = ",dtype=bfloat16"

        if request_data["precision"] == "8bit":
            extra_args += ",load_in_8bit=True"
        elif request_data["precision"] == "4bit":
            extra_args += ",load_in_4bit=True"
        elif request_data["precision"] == "GPTQ":
            extra_args += ",autogptq=True"
        
        model_args += extra_args

        model_args += f",revision={request_data['revision']},trust_remote_code={str(TRUST_REMOTE_CODE)},parallelize=True,device_map=auto"

    results = run_eval_on_model(
        model=lm_eval_model_type,
        model_args=model_args,
        output_path=os.path.join(EVAL_RESULTS_PATH, request_data['model']),
        start_time=start_time
    )

    results["config_general"]["model_name"] = request_data["model"]
    results["config_general"]["model_dtype"] = request_data["precision"]
    results["config_general"]["model_size"] = None
    results["config_general"]["job_id"] = job_id
    results["config_general"]["model_id"] = model_id
    results["config_general"]["model_base_model"] = request_data["base_model"]
    results["config_general"]["model_weight_type"] = request_data["weight_type"]
    results["config_general"]["model_revision"] = request_data["revision"]
    results["config_general"]["model_private"] = request_data["private"]
    results["config_general"]["model_params"] = request_data["params"]
    results["config_general"]["model_type"] = request_data["model_type"]
    results["config_general"]["model_architectures"] = request_data["architectures"]
    results["config_general"]["submitted_time"] = request_data["submitted_time"]
    results["config_general"]["lm_eval_model_type"] = lm_eval_model_type
    results["config_general"]["eval_version"] = EVAL_VERSION
    results["config_general"]["model_sha"] = commit_hash

    upload_results(request_data["model"], results)

    request_data["status"] = "FINISHED"
    request_data["eval_version"] = EVAL_VERSION
    update_status_requests(model_id, request_data)

lock = RLock()
MODELS_DOWNLOADED = {}
MODELS_DOWNLOADED_FAILED = {}
MAX_MODELS_DOWNLOADED_QUEUE_SIZE = 5
def download_all_models(pending_df):
    global MODELS_DOWNLOADED, MODELS_DOWNLOADED_FAILED
    for _, request in pending_df.iterrows():
        while len(MODELS_DOWNLOADED) >= MAX_MODELS_DOWNLOADED_QUEUE_SIZE:
            time.sleep(60)
        commit_hash = None
        if request["lm_eval_model_type"] == "huggingface":
            logging.info(f"Downloading of {request['model']} [{request['revision']}]...")
            retrys = 0
            quit_loop = False
            while not quit_loop:
                try:
                    model_to_download = request["model"]
                    revision = request["revision"]
                    if request["weight_type"] == "Adapter":
                        model_to_download = request["base_model"]
                        revision = "main"
                    commit_hash = download_model(model_to_download, revision, force=(retrys >= 2))
                    quit_loop = True
                except Exception as e:
                    print(e)
                    retrys += 1
                    if retrys >= 3:
                        quit_loop = True
                        MODELS_DOWNLOADED_FAILED[f"{request['model']}_{request['revision']}"] = str(e)
        with lock:
            MODELS_DOWNLOADED[f"{request['model']}_{request['revision']}")] = commit_hash
        logging.info(f"Download of {request['model']} [{request['revision']}] completed.")

MODELS_TO_PRIORITIZE = [
    "mistralai/Mixtral-8x7B-v0.1",
    "meta-llama/Llama-2-70b-hf",
    "huggyllama/llama-65b",
    "huggyllama/llama-30b",
    "01-ai/Yi-34B",
    "tiiuae/falcon-40b",
    "Qwen/Qwen-72B",
    "facebook/opt-66b",
    "facebook/opt-30b",
    "xverse/XVERSE-65B",
    "xverse/XVERSE-65B-2",
    "deepseek-ai/deepseek-llm-67b-base",
    "BAAI/Aquila2-34B",
    "AI-Sweden-Models/gpt-sw3-40b"
]

def main_loop():
    global MODELS_DOWNLOADED, MODELS_DOWNLOADED_FAILED
    logging.info("Running main loop")
    download_requests_repo()
    update_eval_version(get_eval_results_df(), EVAL_VERSION)
    requests_df = get_eval_results_df()
    last_job_id = int(requests_df["job_id"].max())
    pending_df = requests_df[requests_df["status"].isin(["PENDING", "RERUN", "PENDING_NEW_EVAL"])]
    
    priority_df = pending_df[requests_df["model"].isin(MODELS_TO_PRIORITIZE)]
    if len(priority_df) > 0:
        pending_df = priority_df
    
    pending_df = pending_df.sort_values('submitted_time')
    download_thread = Thread(target=download_all_models, args=(pending_df,))
    download_thread.daemon = True
    download_thread.start()
    for _, request in pending_df.iterrows():
        with open(request["filepath"], encoding='utf-8') as fp:
            request_dict = json.load(fp)
        last_job_id += 1
        logging.info(f"Starting job: {last_job_id} on model_id: {request['model_id']}")
        try:
            logging.info(f"Waiting download of {request['model']} [{request['revision']}]...")
            while f"{request['model']}_{request['revision']}" not in MODELS_DOWNLOADED:
                time.sleep(60)
            with lock:
                commit_hash = MODELS_DOWNLOADED[f"{request['model']}_{request['revision']}"]
                del MODELS_DOWNLOADED[f"{request['model']}_{request['revision']}"]
            if f"{request['model']}_{request['revision']}" in MODELS_DOWNLOADED_FAILED:
                exception_msg = MODELS_DOWNLOADED_FAILED[f"{request['model']}_{request['revision']}"]
                raise Exception(f"Failed to download and/or use the AutoModel class, trust_remote_code={TRUST_REMOTE_CODE} - Original Exception: {exception_msg}")
            run_request(request["model_id"], request_dict, job_id=last_job_id, commit_hash=commit_hash)
        except Exception as e:
            request_dict["status"] = "FAILED"
            request_dict["error_msg"] = str(e)
            request_dict["traceback"] = traceback.format_exc()
            logging.error(request_dict["traceback"])
            update_status_requests(request["model_id"], request_dict)
        gc.collect()
        torch.cuda.empty_cache()
    download_thread.join()
    #free_up_cache()

if __name__ == "__main__":
    while True:
        main_loop()
        print("sleeping")
        time.sleep(60)