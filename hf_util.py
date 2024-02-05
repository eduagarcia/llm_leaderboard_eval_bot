from huggingface_hub import snapshot_download, HfApi
from envs import EVAL_REQUESTS_PATH, EVAL_RESULTS_PATH, QUEUE_REPO, RESULTS_REPO, RAW_RESULTS_REPO, TRUST_REMOTE_CODE
from huggingface_hub import scan_cache_dir
import json
import os
import shutil
import time
from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from huggingface_hub.utils._errors import HfHubHTTPError

from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES,
)

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

def _update_status_requests(model_id, new_data):
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

def _upload_results(model_name, results_data):
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


def _upload_raw_results(model_name):
    #upload results
    api.upload_folder(
        folder_path=os.path.join(EVAL_REQUESTS_PATH, model_name),
        repo_id=RAW_RESULTS_REPO,
        path_in_repo=model_name,
        allow_patterns="*.json",
        ignore_patterns=["*/.ipynb_checkpoints/*", ".ipynb_checkpoints"],
        repo_type="dataset",
        commit_message=f"Uploading raw results for {model_name}",
    )

def _try_request_again(func, download_func, *args):
    for i in range(5):
        try:
            func(*args)
            break
        except HfHubHTTPError as e:
            download_func()
            pass
        except Exception as e:
            raise e

def update_status_requests(*args):
    return _try_request_again(_update_status_requests, download_requests_repo, *args)

def upload_results(*args):
    return _try_request_again(_upload_results, lambda: time.sleep(1), *args)

def upload_raw_results(*args):
    return _try_request_again(_upload_raw_results, lambda: time.sleep(1), *args)

def commit_requests_folder(commit_message):
    api.upload_folder(
        folder_path=EVAL_REQUESTS_PATH,
        repo_id=QUEUE_REPO,
        path_in_repo="./",
        allow_patterns="*.json",
        ignore_patterns=["*/.ipynb_checkpoints/*", ".ipynb_checkpoints"],
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

def delete_model_from_cache(commit_hash):
    hf_info = scan_cache_dir()
    revisions = []
    for repo in hf_info.repos:
        for revision in repo.revisions:
            if commit_hash == revision.commit_hash:
                revisions.append(revision.commit_hash)
    if len(revisions) == 0:
        return
    del_strategy = hf_info.delete_revisions(*revisions)
    del_strategy.execute()

def download_model(name, revision="main", force=False):
    config = AutoConfig.from_pretrained(
        name, revision=revision, trust_remote_code=TRUST_REMOTE_CODE, force_download=force, resume_download=not force
    )
    commit_hash = config._commit_hash

    # check if model is already downloaded
    # Not perfect, but works for most cases
    hf_info = scan_cache_dir()
    for repo in hf_info.repos:
        if repo.repo_id == name:
            for r in repo.revisions:
                if commit_hash == r.commit_hash and len(r.files) >= 4 and r.size_on_disk > 10*1024*1024: #10Mb
                    return commit_hash
                
    if (getattr(config, "model_type") in MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES):
        # first check if model type is listed under seq2seq models, since some
        # models like MBart are listed in both seq2seq and causal mistakenly in HF transformers.
        # these special cases should be treated as seq2seq models.
        AUTO_MODEL_CLASS = AutoModelForSeq2SeqLM
    elif (getattr(config, "model_type") in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES):
        AUTO_MODEL_CLASS = AutoModelForCausalLM
    else:
        if not TRUST_REMOTE_CODE:
            print(
                "HF model type is neither marked as CausalLM or Seq2SeqLM. \
            This is expected if your model requires `trust_remote_code=True` but may be an error otherwise."
            )
        # if model type is neither in HF transformers causal or seq2seq model registries
        # then we default to AutoModelForCausalLM
        AUTO_MODEL_CLASS = AutoModelForCausalLM
        
    model = AUTO_MODEL_CLASS.from_pretrained(
        name, revision=revision, device_map="cpu", trust_remote_code=TRUST_REMOTE_CODE, force_download=force, resume_download=not force
    )
    tokenizer = AutoTokenizer.from_pretrained(
        name, revision=revision, trust_remote_code=TRUST_REMOTE_CODE, force_download=force, resume_download=not force
    )
    del model, tokenizer
    return commit_hash

if __name__ == "__main__":
    #download_requests_repo()
    import sys
    download_model(sys.argv[-1], "main")