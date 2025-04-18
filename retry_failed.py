import os
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'
from hf_util import download_requests_repo, commit_requests_folder
from eval_queue import get_eval_results_df
import datetime as dt
import pandas as pd
import json
import logging
import sys

def retry_failed(error_contains=None):
    failed = []
    download_requests_repo()
    requests_df = get_eval_results_df()
    status = ["FAILED"]
    if "24" in error_contains:
        requests_df = requests_df[~requests_df.isna()["job_start_time"]]
        requests_df["job_start_time"] = pd.to_datetime(requests_df['job_start_time'], format="%Y-%m-%dT%H-%M-%S.%f")
        requests_df = requests_df[requests_df['job_start_time'] <= (dt.datetime.now()-dt.timedelta(hours=23))]
        if error_contains == "24":
            error_contains = None
        else:
            error_contains = error_contains.replace('24', '')
    if "RUNNING" in error_contains:
        status = ["RUNNING"]
        error_contains = None
    if error_contains is not None:
        requests_df = requests_df[~requests_df.isna()["error_msg"]]
        requests_df = requests_df[requests_df["error_msg"] != ""]
        requests_df = requests_df[requests_df["error_msg"].str.contains(error_contains)]
    pending_df = requests_df[requests_df["status"].isin(status)]

    pending_df = pending_df[pending_df["params"] < 33]
    print(pending_df["model"].values)

    #requests_df = requests_df[((requests_df["params"] <= 36) | (requests_df["precision"] == '4bit'))]
    for _, request in pending_df.iterrows():
        with open(request["filepath"], encoding='utf-8') as fp:
            request_dict = json.load(fp)
        request_dict['status'] = "RERUN"
        #request_dict['job_id'] = -1
        #request_dict['job_start_time'] = None
        if 'error_msg' in request_dict:
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