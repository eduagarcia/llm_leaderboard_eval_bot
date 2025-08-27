#!/root/miniconda3/envs/torch21/bin/python
import os
os.environ['HF_HUB_CACHE'] = '/workspace/datasets/hf_cache/'
os.environ['HF_CACHE'] = '/workspace/datasets/hf_cache/'
os.environ['HF_HOME'] = '/workspace/datasets/hf_cache/'
from huggingface_hub import HfApi
import time
repo_ids = [
    "eduagarcia/open_pt_llm_leaderboard",
    "datalawyer/legal_pt_llm_leaderboard",
    "eduagarcia/multilingual-chatbot-arena-leaderboard",
    "eduagarcia-temp/portuguese-leaderboard-results-to-modelcard",
]
api = HfApi()
for repo_id in repo_ids:
    print(f"Reseting {repo_id}")
    print(api.restart_space(repo_id=repo_id))
    time.sleep(120)
