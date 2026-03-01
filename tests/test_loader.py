import json
import tempfile
from pathlib import Path

from apo.data.loader import load_dataset, DatasetError


def test_load_json_dataset():
    data = [
        {"input": "hello", "expected_output": "world"},
        {"input": "foo"},
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        path = f.name

    tasks = load_dataset(Path(path))
    assert len(tasks) == 2
    assert tasks[0]["input"] == "hello"
    assert tasks[0]["expected_output"] == "world"
    assert tasks[1]["input"] == "foo"
    assert tasks[1].get("expected_output") is None


def test_load_csv_dataset():
    csv_content = "input,expected_output\nhello,world\nfoo,bar\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(csv_content)
        path = f.name

    tasks = load_dataset(Path(path))
    assert len(tasks) == 2
    assert tasks[0]["input"] == "hello"


def test_load_dataset_missing_input_field():
    data = [{"text": "hello"}]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        path = f.name

    try:
        load_dataset(Path(path))
        assert False, "Should have raised DatasetError"
    except DatasetError:
        pass


def test_load_dataset_empty():
    data = []
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        path = f.name

    try:
        load_dataset(Path(path))
        assert False, "Should have raised DatasetError"
    except DatasetError:
        pass
