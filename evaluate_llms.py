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
        model_id,
        request_data,
        job_id=None,
        commit_hash=None,
        gpu_id = 0,
        parallelize = True
    ):
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

        if parallelize:
            model_args += f",parallelize=True"
        else:
            model_args += f",device=cuda:{gpu_id}"

        model_args += f",revision={request_data['revision']},trust_remote_code={str(TRUST_REMOTE_CODE)}"

    results = run_eval_on_model(
        model=lm_eval_model_type,
        model_args=model_args,
        output_path=os.path.join(EVAL_RESULTS_PATH, request_data['model']),
        start_time=start_time
    )

    results["config_general"]["model_name"] = request_data["model"]
    results["config_general"]["model_dtype"] = request_data["precision"]
    results["config_general"]["job_id"] = job_id
    results["config_general"]["model_id"] = model_id
    results["config_general"]["model_base_model"] = request_data["base_model"]
    results["config_general"]["model_weight_type"] = request_data["weight_type"]
    results["config_general"]["model_revision"] = request_data["revision"]
    results["config_general"]["model_private"] = request_data["private"]
    results["config_general"]["model_type"] = request_data["model_type"]
    results["config_general"]["model_architectures"] = request_data["architectures"]
    results["config_general"]["submitted_time"] = request_data["submitted_time"]
    results["config_general"]["lm_eval_model_type"] = lm_eval_model_type
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

    #delete_model_from_cache(commit_hash)

lock = RLock()
MODELS_TO_DOWNLOAD = []
MODELS_DOWNLOADED = {}
MODELS_DOWNLOADED_FAILED = {}
def download_all_models(pending_df, max_queue_size=5):
    global MODELS_DOWNLOADED, MODELS_DOWNLOADED_FAILED, MODELS_TO_DOWNLOAD
    for _, request in pending_df.iterrows():
        MODELS_TO_DOWNLOAD.append(f"{request['model']}_{request['revision']}")
    for _, request in pending_df.iterrows():
        while len(MODELS_DOWNLOADED) >= max_queue_size:
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
                    traceback.print_exc()
                    print(f'Error on downloading model {model_to_download}:', e)
                    retrys += 1
                    if retrys >= 3:
                        quit_loop = True
                        MODELS_DOWNLOADED_FAILED[f"{request['model']}_{request['revision']}"] = str(e)
        with lock:
            MODELS_DOWNLOADED[f"{request['model']}_{request['revision']}"] = commit_hash
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


def wait_download_and_run_request(request, gpu_id, parallelize, job_id):
    global MODELS_DOWNLOADED, MODELS_DOWNLOADED_FAILED, MODELS_TO_DOWNLOAD
    with open(request["filepath"], encoding='utf-8') as fp:
        request_dict = json.load(fp)
    logging.info(f"Starting job: {job_id} on model_id: {request['model_id']}")
    try:
        logging.info(f"Waiting download of {request['model']} [{request['revision']}]...")
        if f"{request['model']}_{request['revision']}" in MODELS_TO_DOWNLOAD:
            while f"{request['model']}_{request['revision']}" not in MODELS_DOWNLOADED:
                time.sleep(60)
            with lock:
                commit_hash = MODELS_DOWNLOADED[f"{request['model']}_{request['revision']}"]
                del MODELS_DOWNLOADED[f"{request['model']}_{request['revision']}"]
            if f"{request['model']}_{request['revision']}" in MODELS_DOWNLOADED_FAILED:
                exception_msg = MODELS_DOWNLOADED_FAILED[f"{request['model']}_{request['revision']}"]
                raise Exception(f"Failed to download and/or use the AutoModel class, trust_remote_code={TRUST_REMOTE_CODE} - Original Exception: {exception_msg}")
        run_request(
            request["model_id"],
            request_dict,
            job_id=job_id,
            commit_hash=commit_hash,
            gpu_id=gpu_id,
            parallelize=parallelize
        )
    except Exception as e:
        request_dict["status"] = "FAILED"
        request_dict["error_msg"] = str(e)
        request_dict["traceback"] = traceback.format_exc()
        logging.error(request_dict["traceback"])
        update_status_requests(request["model_id"], request_dict)
    finally:
        gc.collect()
        if parallelize:
            torch.cuda.empty_cache()
        else:
            with torch.cuda.device(f"cuda:{gpu_id}"):
                torch.cuda.empty_cache()

def get_pending_df():
    requests_df = get_eval_results_df()
    pending_df = requests_df[requests_df["status"].isin(["PENDING", "RERUN", "PENDING_NEW_EVAL"])].copy()
    
    pending_df['priority'] = pending_df["model"].apply(lambda x: int(x not in MODELS_TO_PRIORITIZE))
    pending_df['source_priority'] = pending_df["source"].apply(lambda x: {"leaderboard": 0, "script": 1}.get(x, 2))
    
    pending_df = pending_df.sort_values(['priority', 'source_priority', 'submitted_time'])
    pending_df = pending_df.drop(['priority', 'source_priority'], axis=1)
    
    return pending_df

def main_loop(
        gpu_ids = [0],
        parallelize = False,
        download_queue_size = 5
    ):
    global MODELS_DOWNLOADED, MODELS_DOWNLOADED_FAILED
    logging.info("Running main loop")
    download_requests_repo()
    requests_df = get_eval_results_df()
    update_eval_version(requests_df, EVAL_VERSION)

    last_job_id = int(requests_df["job_id"].max())
    
    pending_df = get_pending_df()
    
    download_thread = Thread(target=download_all_models, args=(pending_df,download_queue_size))
    download_thread.daemon = True
    download_thread.start()

    if parallelize:
        gpu_ids = [gpu_ids[0]]
    
    if len(gpu_ids) == 1:
        while len(pending_df) > 0:
            pending_df = get_pending_df()
            request = pending_df.iloc[0]
            wait_download_and_run_request(request, gpu_ids[0], parallelize, last_job_id)
    else:
        # spawn threads of wait_download_and_run_request for each gpu
        max_queue_size = len(gpu_ids) + 1
        task_queue = Queue(maxsize=max_queue_size)
        job_id_counter = count(start=last_job_id)  # Thread-safe counter for job_id
        job_id_lock = Lock()

        def worker(task_queue, job_id_counter, job_id_lock, gpu_id):
            while True:
                task = task_queue.get()
                if task is None:  # Sentinel value to exit thread
                    task_queue.task_done()
                    break
                request = task
                try:
                    with job_id_lock:
                        job_id = next(job_id_counter)
                    wait_download_and_run_request(request, gpu_id, False, job_id)
                except Exception as e:
                    traceback.print_exc()
                    print(f"Error processing task with GPU {gpu_id}: {e}")
                finally:
                    task_queue.task_done()

        # Start a worker thread for each GPU
        threads = []
        for gpu_id in gpu_ids:
            thread = Thread(target=worker, args=(task_queue, job_id_counter, job_id_lock, gpu_id))
            thread.daemon = True
            thread.start()
            threads.append(thread)

        # Enqueue tasks
        models_queued = []
        while len(pending_df) > 0:
            pending_df = get_pending_df()
            pending_df = pending_df[~(pending_df["model_id"].isin(models_queued))]
            loop_size = len(gpu_ids) if len(gpu_ids) < len(pending_df) else len(pending_df)
            for i in range(loop_size):
                request = pending_df.iloc[i]
                task_queue.put(request)
                print(f"{request['model_id']} queued")
                models_queued.append(request["model_id"])
                
        for _ in gpu_ids:
            task_queue.put(None)  # Sentinel values to stop the worker threads

        # Wait for all of the tasks to finish
        for thread in threads:
            thread.join()

    download_thread.join()

def parse_eval_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default="0",
        help="GPU ids to utilize.",
    )
    parser.add_argument(
        "--parallelize",
        type=bool,
        default=False,
        help="Wether to paralleize model across all available GPU's with accelerator",
    )
    parser.add_argument(
        "--download_queue_size",
        type=int,
        default=5,
        help="Max download queue size",
    )
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_eval_args()
    gpu_ids= args.gpu_ids.split(',')
    while True:
        main_loop(
            gpu_ids=gpu_ids,
            parallelize=args.parallelize,
            download_queue_size=args.download_queue_size
        )
        print("sleeping")
        time.sleep(60)