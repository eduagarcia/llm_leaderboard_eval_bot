import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Union

import numpy as np

from lm_eval import evaluator, utils
from lm_eval.api.registry import ALL_TASKS
from lm_eval.tasks import include_path, initialize_tasks
from lm_eval.utils import make_table

def _handle_non_serializable(o):
    if isinstance(o, np.int64) or isinstance(o, np.int32):
        return int(o)
    elif isinstance(o, set):
        return list(o)
    else:
        return str(o)
    
initialize_tasks("INFO")

def evaluate(
        model="hf",
        tasks=None,
        model_args="",
        num_fewshot=None,
        batch_size=1,
        max_batch_size=None,
        device=None,
        output_path=None,
        limit=None,
        use_cache=None,
        decontamination_ngrams_path=None,
        check_integrity=False,
        write_out=False,
        log_samples=False,
        show_config=False,
        include_path=None,
        gen_kwargs=None,
        verbosity="INFO"
) -> None:
    """Evaluate a model on a set of tasks."""
    eval_logger = utils.eval_logger
    eval_logger.setLevel(getattr(logging, f"{verbosity}"))
    eval_logger.info(f"Verbosity set to {verbosity}")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if limit:
        eval_logger.warning(
            " --limit SHOULD ONLY BE USED FOR TESTING."
            "REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT."
        )
    if include_path is not None:
        eval_logger.info(f"Including path: {include_path}")
        include_path(include_path)

    if tasks is None:
        task_names = ALL_TASKS
    elif tasks == "list":
        eval_logger.info(
            "Available Tasks:\n - {}".format("\n - ".join(sorted(ALL_TASKS)))
        )
        sys.exit()
    else:
        if os.path.isdir(tasks):
            import glob

            task_names = []
            yaml_path = os.path.join(tasks, "*.yaml")
            for yaml_file in glob.glob(yaml_path):
                config = utils.load_yaml_config(yaml_file)
                task_names.append(config)
        else:
            tasks_list = tasks.split(",")
            task_names = utils.pattern_match(tasks_list, ALL_TASKS)
            for task in [task for task in tasks_list if task not in task_names]:
                if os.path.isfile(task):
                    config = utils.load_yaml_config(task)
                    task_names.append(config)
            task_missing = [
                task
                for task in tasks_list
                if task not in task_names and "*" not in task
            ]  # we don't want errors if a wildcard ("*") task name was used

            if task_missing:
                missing = ", ".join(task_missing)
                eval_logger.error(
                    f"Tasks were not found: {missing}\n"
                    f"{utils.SPACING}Try `lm-eval --tasks list` for list of available tasks",
                )
                raise ValueError(
                    f"Tasks not found: {missing}. Try `lm-eval --tasks list` for list of available tasks, or '--verbosity DEBUG' to troubleshoot task registration issues."
                )

    if output_path:
        path = Path(output_path)
        # check if file or 'dir/results.json' exists
        if path.is_file() or Path(output_path).joinpath("results.json").is_file():
            eval_logger.warning(
                f"File already exists at {path}. Results will be overwritten."
            )
            output_path_file = path.joinpath("results.json")
            assert not path.is_file(), "File already exists"
        # if path json then get parent dir
        elif path.suffix in (".json", ".jsonl"):
            output_path_file = path
            path.parent.mkdir(parents=True, exist_ok=True)
            path = path.parent
        else:
            path.mkdir(parents=True, exist_ok=True)
            output_path_file = path.joinpath("results.json")
    elif log_samples and not output_path:
        assert output_path, "Specify --output_path"

    eval_logger.info(f"Selected Tasks: {task_names}")

    results = evaluator.simple_evaluate(
        model=model,
        model_args=model_args,
        tasks=task_names,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        max_batch_size=max_batch_size,
        device=device,
        use_cache=use_cache,
        limit=limit,
        decontamination_ngrams_path=decontamination_ngrams_path,
        check_integrity=check_integrity,
        write_out=write_out,
        log_samples=log_samples,
        gen_kwargs=gen_kwargs,
    )

    if results is not None:
        if log_samples:
            samples = results.pop("samples")
        dumped = json.dumps(
            results, indent=2, default=_handle_non_serializable, ensure_ascii=False
        )
        if show_config:
            print(dumped)

        batch_sizes = ",".join(map(str, results["config"]["batch_sizes"]))

        if output_path:
            output_path_file.open("w", encoding='utf-8').write(dumped)

            if log_samples:
                for task_name, config in results["configs"].items():
                    output_name = "{}_{}".format(
                        re.sub("/|=", "__", model_args), task_name
                    )
                    filename = path.joinpath(f"{output_name}.jsonl")
                    samples_dumped = json.dumps(
                        samples[task_name],
                        indent=2,
                        default=_handle_non_serializable,
                        ensure_ascii=False,
                    )
                    filename.write_text(samples_dumped, encoding="utf-8")

        print(
            f"{model} ({model_args}), gen_kwargs: ({gen_kwargs}), limit: {limit}, num_fewshot: {num_fewshot}, "
            f"batch_size: {batch_size}{f' ({batch_sizes})' if batch_sizes else ''}"
        )
        print(make_table(results))
        if "groups" in results:
            print(make_table(results, "groups"))
    
    return results