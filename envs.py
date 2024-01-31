import os
from yaml import safe_load

TASK_CONFIG_NAME = os.getenv("TASK_CONFIG", "legal_config_0_0_4")
TASK_CONFIG_PATH = os.path.join('tasks_config', TASK_CONFIG_NAME + ".yaml")
with open(TASK_CONFIG_PATH, 'r', encoding='utf-8') as f:
    TASK_CONFIG = safe_load(f)

QUEUE_REPO = os.getenv("QUEUE_REPO", "datalawyer/llm_leaderboard_requests")
RESULTS_REPO = os.getenv("RESULTS_REPO", "datalawyer/llm_leaderboard_results")

CACHE_PATH=os.getenv("HF_HOME", "./downloads")

EVAL_REQUESTS_PATH = os.path.join(CACHE_PATH, "eval-queue")
EVAL_RESULTS_PATH = os.path.join(CACHE_PATH, "eval-results")
DYNAMIC_INFO_PATH = os.path.join(CACHE_PATH, "dynamic-info")

TRUST_REMOTE_CODE = bool(os.getenv("TRUST_REMOTE_CODE", False))

EVAL_VERSION=os.getenv("EVAL_VERSION", TASK_CONFIG["version"])