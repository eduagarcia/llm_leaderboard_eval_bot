from huggingface_hub import snapshot_download, HfApi
from envs import EVAL_REQUESTS_PATH, EVAL_RESULTS_PATH, QUEUE_REPO, RESULTS_REPO
from huggingface_hub import scan_cache_dir
import json
import os
import shutil
from huggingface_

api = HfApi()

def download_requests_repo():
    """Download the eval requests repo"""
    #delete old folder
    #if os.path.exists(EVAL_REQUESTS_PATH):
    #    shutil.rmtree(EVAL_REQUESTS_PATH)
    snapshot_download(
        repo_id=QUEUE_REPO,
        local_dir=EVAL_REQUESTS_PATH,
        repo_type="dataset",
        tqdm_class=None,
        etag_timeout=30
    )

def update_status_requests(model_id, new_data):
    #update one file
    filepath = os.path.join(EVAL_REQUESTS_PATH, f"{model_id}.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(new_data, f, indent=4, ensure_ascii=False)
    api.upload_file(
        path_or_fileobj=filepath,
        path_in_repo=f"{model_id}.json",
        repo_id=QUEUE_REPO,
        repo_type="dataset",
        commit_message=f"Update status of {model_id} to {new_data['status']}",
    )

def upload_results(model_name, results_data):
    #upload results
    filename = f"results_{results_data['config_general']['start_date']}.json"
    filepath = os.path.join(EVAL_RESULTS_PATH, model_name, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=4, ensure_ascii=False)
    api.upload_file(
        path_or_fileobj=filepath,
        path_in_repo=f"{model_name}/{filename}",
        repo_id=RESULTS_REPO,
        repo_type="dataset",
        commit_message=f"Updating model {model_name}",
    )

def commit_requests_folder(commit_message):
    api.upload_folder(
        folder_path=EVAL_REQUESTS_PATH,
        repo_id=QUEUE_REPO,
        path_in_repo="./",
        allow_patterns="*.json",
        repo_type="dataset",
        commit_message=commit_message,
    )

def free_up_cache():
    hf_info = scan_cache_dir()
    revisions = []
    for repo in hf_info.repos:
        for revision in repo.revisions:
            revisions.append(revision.commit_hash)
    del_strategy = hf_info.delete_revisions(*revisions)
    del_strategy.execute()

if __name__ == "__main__":
    download_requests_repo()