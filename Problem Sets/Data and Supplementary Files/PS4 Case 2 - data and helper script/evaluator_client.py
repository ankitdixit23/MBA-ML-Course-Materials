# evaluator_client.py
from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd
import requests


class EvaluatorClientError(Exception):
    pass


def _load_submission_df(submission: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
    if isinstance(submission, pd.DataFrame):
        df = submission.copy()
    else:
        df = pd.read_csv(submission)

    required = {"experiment_id", "implement"}
    missing = required - set(df.columns)
    if missing:
        raise EvaluatorClientError(f"Submission missing required columns: {sorted(missing)}")

    # Keep only required columns
    df = df[["experiment_id", "implement"]].copy()

    # Validate
    if df["experiment_id"].duplicated().any():
        raise EvaluatorClientError("Duplicate experiment_id values in submission.")

    try:
        df["experiment_id"] = df["experiment_id"].astype(int)
        df["implement"] = df["implement"].astype(int)
    except Exception as e:
        raise EvaluatorClientError(f"Could not coerce columns to int: {e}")

    bad = ~df["implement"].isin([0, 1])
    if bad.any():
        raise EvaluatorClientError("Column 'implement' must contain only 0 or 1.")

    return df


def _submit(
    submission: Union[str, Path, pd.DataFrame],
    team_id: str,
    token: str,
    base_url: str,
    mode: str,
    timeout_seconds: int = 30,
) -> dict:
    df = _load_submission_df(submission)

    payload = {
        "team_id": team_id,
        "token": token,
        "decisions": df.to_dict(orient="records"),
    }

    url = f"{base_url.rstrip('/')}/score/{mode}"
    try:
        resp = requests.post(url, json=payload, timeout=timeout_seconds)
    except requests.RequestException as e:
        raise EvaluatorClientError(f"Request failed: {e}")

    if resp.status_code != 200:
        # Try to surface API detail nicely
        try:
            detail = resp.json()
        except Exception:
            detail = resp.text
        raise EvaluatorClientError(f"API error {resp.status_code}: {detail}")

    return resp.json()


def practice_evaluate(
    submission: Union[str, Path, pd.DataFrame],
    team_id: str,
    token: str,
    base_url: str,
    timeout_seconds: int = 30,
    verbose: bool = True,
) -> dict:
    result = _submit(
        submission=submission,
        team_id=team_id,
        token=token,
        base_url=base_url,
        mode="practice",
        timeout_seconds=timeout_seconds,
    )
    if verbose:
        print_result(result)
    return result


def final_evaluate(
    submission: Union[str, Path, pd.DataFrame],
    team_id: str,
    token: str,
    base_url: str,
    timeout_seconds: int = 30,
    verbose: bool = True,
) -> dict:
    result = _submit(
        submission=submission,
        team_id=team_id,
        token=token,
        base_url=base_url,
        mode="final",
        timeout_seconds=timeout_seconds,
    )
    if verbose:
        print_result(result)
    return result


def print_result(result: dict) -> None:
    print("\n=== Evaluator Result ===")
    print(f"Mode: {result.get('mode')}")
    print(f"Team: {result.get('team_id')}")
    print(f"Submission #: {result.get('submission_number')}")
    print(f"Eval experiments: {result.get('n_eval_experiments')}")
    print(f"Implemented: {result.get('n_implemented')} ({100*result.get('deployment_rate', 0):.1f}%)")
    print()
    print(f"Stationary total incremental profit:    {result.get('score_stationary_total'):,.0f}")
    print(f"Nonstationary total incremental profit: {result.get('score_nonstationary_total'):,.0f}")
    print()
    print(
        "Avg profit / eval experiment (stationary):    "
        f"{result.get('avg_profit_per_experiment_stationary'):,.2f}"
    )
    print(
        "Avg profit / eval experiment (nonstationary): "
        f"{result.get('avg_profit_per_experiment_nonstationary'):,.2f}"
    )

    stat_pc = result.get("avg_profit_per_deployed_customer_stationary")
    non_pc = result.get("avg_profit_per_deployed_customer_nonstationary")
    if stat_pc is not None and non_pc is not None:
        print()
        print(f"Avg profit / deployed customer (stationary):    {stat_pc:,.4f}")
        print(f"Avg profit / deployed customer (nonstationary): {non_pc:,.4f}")