from dataclasses import dataclass, make_dataclass
from enum import Enum
from typing import List
from envs import TASK_CONFIG

@dataclass
class Task:
    benchmark: str
    metric: str
    col_name: str
    baseline: float = 0.0
    human_baseline: float = None
    expert_human_baseline: float = None
    few_shot: int = None
    limit: int = None
    task_list: List[str] = None
    link: str = None
    description: str = None
    sources: List[str] = None
    baseline_sources: List[str] = None

Tasks = Enum('Tasks', {k: Task(**v) for k, v in TASK_CONFIG['tasks'].items()})
