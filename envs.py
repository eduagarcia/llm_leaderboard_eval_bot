import os
from yaml import safe_load

TASK_CONFIG_NAME = os.getenv("TASK_CONFIG", "legal_config_0_0_4")
TASK_CONFIG_PATH = os.path.join('tasks_config', TASK_CONFIG_NAME + ".yaml")
with open(TASK_CONFIG_PATH, 'r', encoding='utf-8') as f:
    TASK_CONFIG = safe_load(f)

def get_config(name, default):
    res = None

    if name in os.environ:
        res = os.environ[name]
    elif 'config' in TASK_CONFIG:
        res = TASK_CONFIG['configs'].get(name, None)

    if res is None:
        return default
    return res

QUEUE_REPO = get_config("QUEUE_REPO", "datalawyer/llm_leaderboard_requests")
RESULTS_REPO = get_config("RESULTS_REPO", "datalawyer/llm_leaderboard_results")
RESULTS_REPO = get_config("RAW_RESULTS_REPO", None)

CACHE_PATH=get_config("HF_HOME", "./downloads")

EVAL_REQUESTS_PATH = os.path.join(CACHE_PATH, "eval-queue")
EVAL_RESULTS_PATH = os.path.join(CACHE_PATH, "eval-results")
DYNAMIC_INFO_PATH = os.path.join(CACHE_PATH, "dynamic-info")

TRUST_REMOTE_CODE = bool(get_config("TRUST_REMOTE_CODE", False))

EVAL_VERSION= os.getenv("EVAL_VERSION", TASK_CONFIG["version"])