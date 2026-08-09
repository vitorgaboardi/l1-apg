"""
Full evaluation orchestrator combining:
  - agent_evaluation: Tests agent API-calling across multiple models and generation methods
  - linguistic_diversity: Tests diversity metrics across generation methods and data models
  - linguistic_patterns: Tests stylometric patterns for L1-APG across data models

This orchestrator reads from config/full_evaluation.yaml and calls the main() functions
from each evaluation script, then saves aggregated results to results/results.xlsx.
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Any, Dict
import pandas as pd
import yaml
from rich.console import Console

# Add src and scripts to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

console = Console()

FULL_EVAL_CONFIG_PATH = PROJECT_ROOT / "config" / "full_evaluation.yaml"
BASE_CONFIG_PATH = PROJECT_ROOT / "config" / "evaluation.yaml"
RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_DIR = RESULTS_DIR / "evaluation"


def load_configs() -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Load both the orchestration config and the base evaluation config."""
    for path in (FULL_EVAL_CONFIG_PATH, BASE_CONFIG_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {path}")
    with FULL_EVAL_CONFIG_PATH.open("r", encoding="utf-8") as f:
        full_cfg = yaml.safe_load(f)
    with BASE_CONFIG_PATH.open("r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)
    return full_cfg, base_cfg


def save_results_spreadsheet(results: Dict[str, Any]) -> Path:
    """Save all results to a multi-sheet xlsx file at results/results.xlsx."""
    output_path = RESULTS_DIR / "results.xlsx"
    
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:

        # Sheet 1: Agent Evaluation
        agent_rows = []
        for key, metrics in results["agent_evaluation"].items():
            if metrics:
                agent_model, gen_method = key.split("___", 1)
                row = {
                    "agent_model": agent_model.replace("__", "/"),
                    "generation_method": gen_method,
                    **{k: v for k, v in metrics.items() if k not in ("test_file", "results_file")},
                }
                agent_rows.append(row)
        if agent_rows:
            pd.DataFrame(agent_rows).to_excel(writer, sheet_name="agent_evaluation", index=False)

        # Sheet 2: Linguistic Diversity
        diversity_rows = []
        for key, metrics in results["linguistic_diversity"].items():
            if metrics:
                gen_method, data_model = key.split("___", 1)
                row = {
                    "generation_method": gen_method,
                    "data_model": data_model.replace("__", "/"),
                    **metrics,
                }
                diversity_rows.append(row)
        if diversity_rows:
            pd.DataFrame(diversity_rows).to_excel(writer, sheet_name="linguistic_diversity", index=False)

        # Sheet 3: Linguistic Patterns
        patterns_rows = []
        for key, metrics in results["linguistic_patterns"].items():
            if metrics:
                row = {
                    "data_model": key.replace("__", "/"),
                    **metrics,
                }
                patterns_rows.append(row)
        if patterns_rows:
            pd.DataFrame(patterns_rows).to_excel(writer, sheet_name="linguistic_patterns", index=False)

    return output_path


def main():
    """Run full evaluation orchestration."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Import evaluation main functions
    from evaluation.agent_evaluation import main as agent_main
    from evaluation.linguistic_diversity import main as diversity_main
    from evaluation.linguistic_patterns import main as patterns_main

    
    console.print(f"[bold cyan]Full Evaluation Orchestrator[/]")

    # Load configs
    full_cfg, base_cfg = load_configs()
    enable = full_cfg.get("enable_evaluations", {})
    run_agent = enable.get("agent", True)
    run_diversity = enable.get("diversity", True)
    run_patterns = enable.get("patterns", True)
    console.print(f"[bold cyan]Embedding Model: {base_cfg.get('emb_name', 'N/A')}[/]")

    agent_pairs = [tuple(p) for p in full_cfg.get("agent_eval_pairs", [])]
    diversity_pairs = [tuple(p) for p in full_cfg.get("diversity_eval_pairs", [])]
    patterns_models = full_cfg.get("patterns_eval_models", [])

    console.print(f"  Agent evaluation:       {'enabled' if run_agent else 'disabled'}")
    console.print(f"  Linguistic diversity:   {'enabled' if run_diversity else 'disabled'}")
    console.print(f"  Linguistic patterns:    {'enabled' if run_patterns else 'disabled'}")
    console.print()

    results = {
        "timestamp": datetime.now().isoformat(),
        "agent_evaluation": {},
        "linguistic_diversity": {},
        "linguistic_patterns": {},
    }

    # Agent evaluation
    if run_agent:
        console.print("[bold]=== Agent Evaluation ===[/]")
        agent_output_dir = OUTPUT_DIR / "agent"
        agent_output_dir.mkdir(parents=True, exist_ok=True)

        for agent_model, gen_method in agent_pairs:
            try:
                console.print(f"[cyan]Running:[/] {agent_model} on {gen_method}")
                cfg = base_cfg.copy()
                cfg["generation_method"] = gen_method
                cfg["llm_used_in_agent"] = agent_model
                result = agent_main(cfg=cfg, generation_method=gen_method, agent_model=agent_model, output_dir=agent_output_dir)
                safe_name = f"{agent_model.replace('/', '__')}___{gen_method}"
                results["agent_evaluation"][safe_name] = result
                console.print(f"[green]✓ Done[/]\n")
            except Exception as e:
                console.print(f"[red]Error: {e}[/]\n")

    # Linguistic diversity
    if run_diversity:
        console.print("[bold]=== Linguistic Diversity ===[/]")
        # Load embedding model once and reuse across all diversity evaluations to avoid GPU OOM
        embeddings_model = base_cfg.get("emb_name", "all-MiniLM-L6-v2")
        from sentence_transformers import SentenceTransformer
        emb_model = SentenceTransformer(embeddings_model, device='cuda')
        
        for gen_method, data_model in diversity_pairs:
            try:
                console.print(f"[cyan]Running:[/] {gen_method} with {data_model}")
                cfg = base_cfg.copy()
                cfg["generation_method"] = gen_method
                cfg["llm_used_for_data_generation"] = data_model
                result = diversity_main(cfg=cfg, EMB_MODEL=emb_model)
                safe_name = f"{gen_method}___{data_model.replace('/', '__')}"
                results["linguistic_diversity"][safe_name] = result
                console.print(f"[green]✓ Done[/]\n")
            except Exception as e:
                console.print(f"[red]Error: {e}[/]\n")
        
        # Free GPU memory after diversity evaluations
        del emb_model
        import gc
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except:
            pass

    # Linguistic patterns
    if run_patterns:
        console.print("[bold]=== Linguistic Patterns (L1-APG) ===[/]")
        for data_model in patterns_models:
            try:
                console.print(f"[cyan]Running:[/] {data_model}")
                cfg = base_cfg.copy()
                cfg["generation_method"] = "l1-apg"
                cfg["llm_used_for_data_generation"] = data_model
                result = patterns_main(cfg=cfg)
                safe_name = data_model.replace("/", "__")
                results["linguistic_patterns"][safe_name] = result
                console.print(f"[green]✓ Done[/]\n")
            except Exception as e:
                console.print(f"[red]Error: {e}[/]\n")

    # Save JSON summary
    json_file = OUTPUT_DIR / f"evaluation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with json_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    console.print(f"[green]✓ JSON summary saved to: {json_file}[/]")

    # Save spreadsheet
    xlsx_path = save_results_spreadsheet(results)
    console.print(f"[green]✓ Spreadsheet saved to: {xlsx_path}[/]")


if __name__ == "__main__":
    main()
