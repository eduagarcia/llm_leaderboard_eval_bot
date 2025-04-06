import os
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'
from hf_util import download_requests_repo, commit_requests_folder
from eval_queue import get_eval_results_df
import datetime as dt
import pandas as pd
import json
import logging
import sys

failed = []
download_requests_repo()
requests_df = get_eval_results_df()
status = ["FAILED", "RUNNING", "FINISHED"]

to_delete = []
to_keep = []
for precisions in [['float16', 'bfloat16'], ['4bit']]:
    pending_df = requests_df[~requests_df["status"].isin(status) & requests_df["precision"].isin(precisions)]
    finished_df = requests_df[requests_df["status"].isin(status) & requests_df["precision"].isin(precisions)]
    
    for _, request in pending_df.iterrows():
        res_pending_df = pending_df[pending_df['model_id'] != request['model_id']]
        
        if request['filepath'] in to_keep:
            continue
        if request['filepath'] in to_delete:
            continue
            
        if request['model'] in finished_df['model'].values:
            to_delete.append(request['filepath'])
            continue
        if request['model'] in res_pending_df['model'].values:
            id_to_keep = request['filepath']
            same_models = pending_df[pending_df['model'] == request['model']]
            same_models_main = same_models[same_models['revision'] == 'main']
            if len(same_models_main) >= 1:
                if len(same_models_main) > 1:
                    same_models_main_bfloat = same_models_main[same_models_main['precision'] == 'bfloat16']
                    if len(same_models_main_bfloat) >= 1:
                        same_models_main = same_models_main_bfloat
                id_to_keep = same_models_main.iloc[0]['filepath']
            same_models = same_models[same_models['filepath'] != id_to_keep]
            for filepath in same_models['filepath'].values:
                to_delete.append(filepath)
            to_keep.append(id_to_keep)

print(to_delete)

for filepath in to_delete:
    os.remove(filepath)

if len(to_delete) > 0:
    commit_requests_folder(f"Deleted {len(to_delete)} reapeated models")



