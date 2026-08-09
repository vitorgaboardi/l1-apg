"""Interactive test script to debug agent behavior with detailed logging."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.tree import Tree

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from agent.agent import Agent  # noqa: E402

console = Console()
EVAL_CONFIG_PATH = PROJECT_ROOT / "config" / "evaluation.yaml"


def load_config(path: Path) -> dict:
    """Load configuration file."""
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r") as f:
        return yaml.safe_load(f)


def resolve_api_key(value: str) -> str:
    """Resolve API key from environment variable if needed."""
    if value in os.environ:
        return os.environ[value]
    return value


def main():
    parser = argparse.ArgumentParser(description="Test agent with detailed logging")
    parser.add_argument(
        "utterance",
        nargs="?",
        default=None,
        help="Utterance to test (if not provided, uses example)",
    )
    parser.add_argument("--config", type=Path, default=EVAL_CONFIG_PATH)
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)

    # Set up parameters
    model = args.model or cfg["llm_used_in_agent"]
    base_url = args.base_url or cfg["llm_url"]
    api_key = resolve_api_key(args.api_key or cfg["api_key"])
    max_steps = args.max_steps if args.max_steps is not None else int(cfg["max_steps"])
    temperature = args.temperature if args.temperature is not None else float(cfg["temperature"])

    # Default utterance for testing
    utterance = args.utterance or "What's the weather in Paris today?"

    # Display configuration
    console.print("\n[bold cyan]Agent Test Configuration[/bold cyan]")
    config_table = Table(show_header=False, box=None)
    config_table.add_column("Parameter", style="yellow")
    config_table.add_column("Value", style="white")
    config_table.add_row("Model", model)
    config_table.add_row("Base URL", base_url)
    config_table.add_row("Max Steps", str(max_steps))
    config_table.add_row("Temperature", str(temperature))
    console.print(config_table)

    # Display utterance
    console.print(Panel.fit(utterance, title="[bold green]Input Utterance[/bold green]", border_style="green"))

    # Create client and agent
    client = OpenAI(base_url=base_url, api_key=api_key)
    agent = Agent(
        client=client,
        model=model,
        max_steps=max_steps,
        temperature=temperature,
    )

    # Run agent
    console.print("\n[bold cyan]Running Agent...[/bold cyan]\n")
    result = agent.run(utterance)

    # Display detailed results
    console.rule("[bold]Agent Execution Details[/bold]")

    # Show each step
    if result.steps:
        console.print(f"\n[bold yellow]Total Steps: {len(result.steps)}[/bold yellow]\n")
        
        for i, step in enumerate(result.steps, 1):
            console.rule(f"[bold cyan]Step {i}[/bold cyan]")
            
            # Thought/Assistant response
            if step.thought:
                console.print(Panel(
                    step.thought,
                    title=f"[bold]Assistant Response #{i}[/bold]",
                    border_style="blue",
                ))
            
            # Tool call
            console.print(f"\n[bold green]Tool Call:[/bold green] {step.action_name}")
            args_json = json.dumps(step.action_args, indent=2, ensure_ascii=False)
            console.print(Panel(
                Syntax(args_json, "json", theme="monokai"),
                title="Arguments",
                border_style="green",
            ))
            
            # Observation/Result
            obs_json = json.dumps(step.observation, indent=2, ensure_ascii=False)
            status = step.observation.get("status_code", "N/A")
            
            if status == 200:
                style = "green"
                status_text = f"✓ Success ({status})"
            else:
                style = "red"
                status_text = f"✗ Failed ({status})"
            
            console.print(f"\n[bold {style}]Execution Result:[/bold {style}] {status_text}")
            console.print(Panel(
                Syntax(obs_json, "json", theme="monokai"),
                title="Observation",
                border_style=style,
            ))
            console.print()
    else:
        console.print("[yellow]No steps recorded (agent returned immediately)[/yellow]")

    # Summary
    console.rule("[bold]Summary[/bold]")
    
    summary_table = Table(show_header=False, box=None)
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="white")
    
    summary_table.add_row("Total Predicted Calls", str(len(result.predicted_calls)))
    summary_table.add_row("Total Executions", str(len(result.execution_results)))
    
    if result.execution_results:
        successes = sum(1 for r in result.execution_results if r.get("status_code") == 200)
        esr = successes / len(result.execution_results)
        summary_table.add_row("Successful Executions", f"{successes}/{len(result.execution_results)}")
        summary_table.add_row("ESR", f"{esr:.2%}")
    
    if result.final_text:
        summary_table.add_row("Final Response", result.final_text[:100] + "..." if len(result.final_text) > 100 else result.final_text)
    
    if result.error:
        summary_table.add_row("Error", f"[red]{result.error}[/red]")
    
    console.print(summary_table)

    # Show all predicted calls
    if result.predicted_calls:
        console.print("\n[bold cyan]All Predicted Calls:[/bold cyan]")
        for i, call in enumerate(result.predicted_calls, 1):
            console.print(f"{i}. [green]{call['name']}[/green]: {json.dumps(call['arguments'])}")
        
        # Check for duplicates
        unique_calls = []
        for call in result.predicted_calls:
            call_str = json.dumps(call, sort_keys=True)
            if call_str not in unique_calls:
                unique_calls.append(call_str)
        
        if len(unique_calls) < len(result.predicted_calls):
            console.print(f"\n[bold red]⚠ WARNING: Detected duplicate calls![/bold red]")
            console.print(f"Total calls: {len(result.predicted_calls)}, Unique: {len(unique_calls)}")

    # Raw response
    if result.raw_response and result.raw_response.strip():
        console.print("\n[bold cyan]Raw Assistant Response:[/bold cyan]")
        console.print(Panel(result.raw_response.strip(), border_style="dim"))

    console.print()


if __name__ == "__main__":
    main()
