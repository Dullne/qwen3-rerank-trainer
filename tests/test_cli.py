"""CLI smoke tests for dependency and argument handling."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run_module(module: str, *args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [sys.executable, "-m", module, *args],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_sft_cli_missing_data_returns_clean_error():
    result = _run_module(
        "qwen3_rerank_trainer.training.cli",
        "--model",
        "/no/model",
        "--data",
        "/no/data",
        "--output",
        "/tmp/qwen-sft-test",
    )

    combined = result.stdout + result.stderr
    assert result.returncode == 1
    assert "数据文件不存在" in combined
    assert "Traceback" not in combined


def test_rl_cli_missing_data_returns_clean_error():
    result = _run_module(
        "qwen3_rerank_trainer.training.rl_cli",
        "--sft_model",
        "/no/sft",
        "--base_model",
        "/no/base",
        "--data",
        "/no/data",
        "--output",
        "/tmp/qwen-rl-test",
    )

    combined = result.stdout + result.stderr
    assert result.returncode == 1
    assert "训练文件不存在" in combined
    assert "Traceback" not in combined


def test_rl_cli_exposes_precision_flags_in_help():
    result = _run_module("qwen3_rerank_trainer.training.rl_cli", "--help")

    assert result.returncode == 0
    assert "--bf16" in result.stdout
    assert "--fp16" in result.stdout


def test_sft_cli_rejects_invalid_fixed_positive_count_before_data_load():
    result = _run_module(
        "qwen3_rerank_trainer.training.cli",
        "--model",
        "/no/model",
        "--data",
        "/no/data",
        "--n-docs",
        "2",
        "--n-pos",
        "3",
    )

    combined = result.stdout + result.stderr
    assert result.returncode == 1
    assert "必须小于" in combined
    assert "Traceback" not in combined
