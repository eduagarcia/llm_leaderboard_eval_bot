from hf_util import download_requests_repo, commit_requests_folder
from eval_queue import get_eval_results_df
import json
import logging

def retry_failed():
    failed = []
    download_requests_repo()
    requests_df = get_eval_results_df()
    pending_df = requests_df[requests_df["status"].isin(["FAILED"])]
    for _, request in pending_df.iterrows():
        with open(request["filepath"], encoding='utf-8') as fp:
            request_dict = json.load(fp)
        request_dict['status'] = "RERUN"
        #request_dict['job_id'] = -1
        #request_dict['job_start_time'] = None
        del request_dict['error_msg']
        del request_dict['traceback']
        with open(request["filepath"], "w", encoding="utf-8") as f:
            json.dump(request_dict, f, indent=4, ensure_ascii=False)
        failed.append(request["model_id"])
    if len(failed) > 0:
        commit_requests_folder(f"Retry {len(failed)} FAILED models")

if __name__ == "__main__":
    retry_failed()