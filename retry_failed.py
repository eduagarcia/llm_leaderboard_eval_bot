from hf_util import download_requests_repo, commit_requests_folder
from eval_queue import get_eval_results_df
import json
import logging
import sys

def retry_failed(error_contains=None):
    failed = []
    download_requests_repo()
    requests_df = get_eval_results_df()
    status = ["FAILED", "RUNNING"]
    if error_contains == "RUNNING":
        status = ["RUNNING"]
        error_contains = None
    pending_df = requests_df[requests_df["status"].isin(status)]
    for _, request in pending_df.iterrows():
        with open(request["filepath"], encoding='utf-8') as fp:
            request_dict = json.load(fp)
        request_dict['status'] = "RERUN"
        #request_dict['job_id'] = -1
        #request_dict['job_start_time'] = None
        if 'error_msg' in request_dict:
            if error_contains is not None and error_contains not in request_dict['error_msg']:
                continue
            del request_dict['error_msg']
        if 'traceback' in request_dict:
            del request_dict['traceback']
        with open(request["filepath"], "w", encoding="utf-8") as f:
            json.dump(request_dict, f, indent=4, ensure_ascii=False)
        failed.append(request["model_id"])
    if len(failed) > 0:
        commit_requests_folder(f"Retry {len(failed)} FAILED models")

if __name__ == "__main__":
    if len(sys.argv) <= 1:
        retry_failed()
    else:
        retry_failed(error_contains=sys.argv[-1])