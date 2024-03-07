import os
import subprocess

import pytest


# example file is from 003 so should always lose against gpt4 turbo
@pytest.mark.slow
def test_cli_evaluate_example():
    env = os.environ.copy()
    env["IS_ALPACA_EVAL_2"] = "True"

    result = subprocess.run(
        [
            "alpaca_eval",
            "--model_outputs",
            "example/outputs.json",
            "--max_instances",
            "3",
            "--annotators_config",
            "claude",
            "--is_avoid_reannotations",
            "False",
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    normalized_output = " ".join(result.stdout.split())
    expected_output = " ".join("example 0.00 0.00 3".split())

    assert expected_output in normalized_output


@pytest.mark.slow
def test_openai_fn_evaluate_example():
    env = os.environ.copy()
    env["IS_ALPACA_EVAL_2"] = "True"
    result = subprocess.run(
        [
            "alpaca_eval",
            "--model_outputs",
            "example/outputs.json",
            "--max_instances",
            "2",
            "--annotators_config",
            "alpaca_eval_gpt4_fn",
            "--is_avoid_reannotations",
            "False",
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    normalized_output = " ".join(result.stdout.split())
    expected_output = " ".join("example 0.00 0.00 2".split())

    assert expected_output in normalized_output