#!/root/miniconda3/envs/torch21/bin/python
from huggingface_hub import HfApi
repo_ids = [
    "eduagarcia/open_pt_llm_leaderboard",
    "datalawyer/legal_pt_llm_leaderboard"
]
api = HfApi()
for repo_id in repo_ids:
    print(f"Reseting {repo_id}")
    print(api.restart_space(repo_id=repo_id))
