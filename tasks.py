from dataclasses import dataclass, make_dataclass
from enum import Enum
from typing import List

@dataclass
class Task:
    benchmark: str
    metric: str
    col_name: str
    baseline: float = 0.0
    human_baseline: float = 0.0
    few_shot: int = None
    limit: int = None
    task_list: List[str] = None
    link: str = None
    description: str = None

class Tasks(Enum):
    oab_exams = Task(
        benchmark="oab_exams",
        metric="exact_match",
        col_name="OAB Exams",
        baseline=25.0, 
        human_baseline=50.0,
        few_shot=5,
        limit=None,
        task_list=["oab_exams_generate"],
        link="https://huggingface.co/datasets/eduagarcia/oab_exams",
        description="OAB Exams is a dataset of 1,000 questions from the Brazilian Bar Association's exams."
    )
    brazilian_court_decisions_judgment = Task(
        benchmark="brazilian_court_decisions_judgment",
        metric="f1_macro",
        col_name="BR Court Decisions",
        baseline=33.33, 
        human_baseline=100.0,
        few_shot=5,
        limit=None,
        task_list=["brazilian_court_decisions_judgment_generate"],
        link="https://huggingface.co/datasets/joelniklaus/brazilian_court_decisions",
        description="A classification dataset of court decisions from the Tribunal de Justiça de Alagoas (TJAL, the State Supreme Court of Alagoas (Brazil)."
    )
    datalawyer_frases = Task(
        benchmark="datalawyer_frases",
        metric="f1_macro",
        col_name="DL Frases",
        baseline=10.0, 
        human_baseline=100.0,
        few_shot=15,
        limit=2000,
        task_list=["datalawyer_frases_generate"],
        link="https://huggingface.co/datasets/eduagarcia/portuguese_benchmark",
        description="A classification dataset"
    )
    rrip = Task(
        benchmark="rrip",
        metric="f1_macro",
        col_name="RRIP",
        baseline=12.5, 
        human_baseline=100.0,
        few_shot=15,
        limit=None,
        task_list=["rrip_generate"],
        link="https://huggingface.co/datasets/eduagarcia/portuguese_benchmark",
        description="A classification dataset"
    )
