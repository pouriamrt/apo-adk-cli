from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TypedDict


class Task(TypedDict, total=False):
    input: str
    expected_output: str


class DatasetError(Exception):
    pass


def load_dataset(path: Path) -> list[Task]:
    """Load dataset from JSON or CSV file."""
    if not path.exists():
        raise DatasetError(f"Dataset file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".json":
        tasks = _load_json(path)
    elif suffix == ".csv":
        tasks = _load_csv(path)
    else:
        raise DatasetError(f"Unsupported file format: {suffix}. Use .json or .csv")

    if not tasks:
        raise DatasetError("Dataset is empty")

    for i, task in enumerate(tasks):
        if "input" not in task:
            raise DatasetError(f"Task {i} missing required 'input' field")

    return tasks


def _load_json(path: Path) -> list[Task]:
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise DatasetError("JSON dataset must be an array of objects")
    return data


def _load_csv(path: Path) -> list[Task]:
    tasks: list[Task] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            task: Task = {"input": row["input"]}
            if "expected_output" in row and row["expected_output"]:
                task["expected_output"] = row["expected_output"]
            tasks.append(task)
    return tasks


def split_dataset(tasks: list[Task], val_ratio: float = 0.3) -> tuple[list[Task], list[Task]]:
    """Split dataset into train and validation sets."""
    split_idx = max(1, int(len(tasks) * (1 - val_ratio)))
    return tasks[:split_idx], tasks[split_idx:]
