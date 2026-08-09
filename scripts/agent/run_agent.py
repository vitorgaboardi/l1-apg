"""Run the tool-calling agent with predefined utterances or interactive input."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from agent.agent import Agent

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
RESULTS_DIR = PROJECT_ROOT / "results"
EVAL_CONFIG_PATH = PROJECT_ROOT / "config" / "evaluation.yaml"

sys.path.insert(0, str(SRC_DIR))
console = Console()


def _load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Could not find config file at: {path}")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    required = [
        "dataset_path",
        "generation_method",
        "llm_used_for_data_generation",
        "llm_used_in_agent",
        "llm_url",
        "api_key",
        "max_steps",
        "temperature",
    ]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"Missing required config fields in {path}: {missing}")
    return cfg


def _safe_name(name: str) -> str:
    return name.replace("/", "__").replace(" ", "_")


def _resolve_api_key(value: str) -> str:
    if value in os.environ:
        return os.environ[value]
    return value


def _load_test_utterances(dataset_root: Path, generation_method: str) -> list[str]:
    dataset_file = dataset_root / "test" / f"{generation_method}.csv"
    if not dataset_file.exists():
        raise FileNotFoundError(
            f"Test dataset not found for method '{generation_method}': {dataset_file}"
        )

    df = pd.read_csv(dataset_file, sep=None, engine="python")
    if "utterance" not in df.columns:
        raise ValueError(f"Column 'utterance' not found in {dataset_file}")

    utterances = df["utterance"].dropna().astype(str).tolist()
    if not utterances:
        raise ValueError(f"No utterances found in {dataset_file}")
    return utterances


def _resolve_from_project(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _build_client(base_url: str, api_key: str) -> OpenAI:
    return OpenAI(base_url=base_url, api_key=api_key)


def _save_results_jsonl(results: list[dict], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _append_result_jsonl(result: dict, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


def _load_existing_results(output_file: Path) -> set[str]:
    """Load already-processed utterances from existing results file."""
    if not output_file.exists():
        return set()
    
    processed_utterances = set()
    with output_file.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    result = json.loads(line)
                    if "utterance" in result:
                        processed_utterances.add(result["utterance"])
                except json.JSONDecodeError:
                    continue
    return processed_utterances


def _summarize_call_success(execution_results: list[dict]) -> tuple[int, int, float]:
    total = len(execution_results)
    if total == 0:
        return 0, 0, 0.0
    ok = sum(1 for r in execution_results if r.get("status_code") == 200)
    return ok, total, ok / total


def run_batch(agent: Agent, utterances: list[str], output_file: Path) -> None:
    all_rows: list[dict] = []
    
    # Check if file exists and load already-processed utterances
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    if output_file.exists():
        # File exists - load processed utterances and continue
        processed_utterances = _load_existing_results(output_file)
        if processed_utterances:
            console.print(f"[yellow]Found existing results with {len(processed_utterances)} processed utterances. Continuing from where we left off.[/yellow]")
            # Filter out already-processed utterances
            utterances = [u for u in utterances if u not in processed_utterances]
            if not utterances:
                console.print("[green]All utterances already processed. Nothing to do.[/green]")
                return
        else:
            console.print("[yellow]Found existing results file but it's empty. Starting fresh.[/yellow]")
    else:
        # File doesn't exist - create it fresh
        output_file.write_text("", encoding="utf-8")
        console.print("[cyan]Creating new results file.[/cyan]")
    
    console.print(Panel.fit(f"Running {len(utterances)} utterances", title="Agent Batch Test"))

    for i, utterance in enumerate(utterances, start=1):
        console.rule(f"[{i}/{len(utterances)}] {utterance}")
        result = agent.run(utterance)

        ok, total, esr = _summarize_call_success(result.execution_results)

        console.print(f"Predicted calls: {len(result.predicted_calls)}")
        console.print(f"Execution success: {ok}/{total} (ESR={esr:.2%})")
        if result.error:
            console.print(f"[red]Agent error:[/red] {result.error}")
        if result.final_text:
            console.print(f"Final text: {result.final_text}")

        row = {
            "utterance": result.utterance,
            "predicted_calls": result.predicted_calls,
            "execution_results": result.execution_results,
            "raw_response": result.raw_response,
            "final_text": result.final_text,
            "error": result.error,
            "execution_successes": ok,
            "execution_total": total,
            "timestamp": datetime.utcnow().isoformat(),
        }
        all_rows.append(row)
        _append_result_jsonl(row, output_file)

    console.print(f"\nSaved {len(all_rows)} results to: {output_file}")


def run_interactive(agent: Agent) -> None:
    console.print(Panel.fit("Interactive mode. Type 'exit' to stop.", title="Agent Interactive"))
    while True:
        utterance = console.input("\n[bold cyan]User[/bold cyan]: ").strip()
        if utterance.lower() in {"exit", "quit"}:
            console.print("Exiting interactive mode.")
            break
        if not utterance:
            continue

        result = agent.run(utterance)
        ok, total, esr = _summarize_call_success(result.execution_results)
        console.print(f"[bold green]Predicted calls[/bold green]: {result.predicted_calls}")
        console.print(f"[bold green]ESR[/bold green]: {ok}/{total} ({esr:.2%})")
        if result.final_text:
            console.print(f"[bold magenta]Assistant[/bold magenta]: {result.final_text}")
        elif result.error:
            console.print(f"[red]Error:[/red] {result.error}")
        else:
            console.print("No final text returned by the model.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run tool-calling agent tests.")
    parser.add_argument("--config", type=Path, default=EVAL_CONFIG_PATH)
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--generation-method", default=None)
    parser.add_argument("--data-model", default=None)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = _load_config(args.config)

    agent_model = args.model or cfg["llm_used_in_agent"]
    base_url = args.base_url or cfg["llm_url"]
    api_key = _resolve_api_key(args.api_key or cfg["api_key"])
    max_steps = args.max_steps if args.max_steps is not None else int(cfg["max_steps"])
    temperature = args.temperature if args.temperature is not None else float(cfg["temperature"])

    generation_method = args.generation_method or cfg["generation_method"]
    data_model = args.data_model or cfg["llm_used_for_data_generation"]
    dataset_root = _resolve_from_project(cfg["dataset_path"])

    if args.output_file is not None:
        output_file = args.output_file
    else:
        output_file = RESULTS_DIR / "agent" / generation_method / f"{_safe_name(agent_model)}.jsonl"

    client = _build_client(base_url=base_url, api_key=api_key)
    agent = Agent(
        client=client,
        model=agent_model,
        max_steps=max_steps,
        temperature=temperature,
    )

    if args.interactive:
        run_interactive(agent)
    else:
        utterances = _load_test_utterances(dataset_root=dataset_root, generation_method=generation_method)
        console.print(
            f"Config -> generation_method={generation_method}, data_model={data_model}, "
            f"agent_model={agent_model}, test_utterances={len(utterances)}"
        )
        run_batch(agent=agent, utterances=utterances, output_file=output_file)


if __name__ == "__main__":
    main()