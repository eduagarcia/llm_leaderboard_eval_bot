import json
import os
from envs import EVAL_REQUESTS_PATH
import pandas as pd
from hf_util import commit_requests_folder

COLS = [
    "model_id",
    "model",
    "base_model",
    "revision",
    "private",
    "precision",
    "architectures",
    "weight_type",
    "status",
    "submitted_time",
    "model_type",
    "job_id",
    "job_start_time",
    "lm_eval_model_type",
    "error_msg",
    "traceback",
    "filepath",
    "eval_version",
    "result_metrics",
    "result_metrics_average"
]

def get_eval_results_df():
    entries = [entry for entry in os.listdir(EVAL_REQUESTS_PATH) if not entry.startswith(".")]
    all_evals = []

    for entry in entries:
        if ".json" in entry:
            file_path = os.path.join(EVAL_REQUESTS_PATH, entry)
            with open(file_path, encoding='utf-8') as fp:
                data = json.load(fp)
                data["model_id"] = entry.replace('.json', '')
                data["filepath"] = file_path

            all_evals.append(data)
        elif ".md" not in entry:
            # this is a folder
            sub_entries = [e for e in os.listdir(f"{EVAL_REQUESTS_PATH}/{entry}") if not e.startswith(".")]
            for sub_entry in sub_entries:
                file_path = os.path.join(EVAL_REQUESTS_PATH, entry, sub_entry)
                with open(file_path, encoding='utf-8') as fp:
                    data = json.load(fp)
                    data["model_id"] = f"{entry}/{sub_entry.replace('.json', '')}"
                    data["filepath"] = file_path

                all_evals.append(data)

    df = pd.DataFrame.from_records(all_evals, columns=COLS)
    df["lm_eval_model_type"] = df["lm_eval_model_type"].fillna("huggingface")
    return df  

def update_eval_version(requests_df, eval_version):
    update_list = []
    for _, request in requests_df.iterrows():
        if ("eval_version" not in request or request["eval_version"] != eval_version)\
            and request["status"] == "FINISHED":
            with open(request["filepath"], encoding='utf-8') as fp:
                request_dict = json.load(fp)
            request_dict["status"] = "PENDING_NEW_EVAL"
            with open(request["filepath"], "w", encoding="utf-8") as f:
                json.dump(request_dict, f, indent=4, ensure_ascii=False)
            update_list.append(request["model_id"])
    if len(update_list) > 0:
        print(f"Pending new eval_version for {len(update_list)} models")
        commit_requests_folder(f"New eval_version {eval_version}, updating {len(update_list)} models")


if __name__ == "__main__":
    #print(get_eval_results_df().iloc[0].to_dict())
    print(get_eval_results_df().sort_values('submitted_time'))