"""
Overview: 
    Computes diversity metrics for one dataset considering one single configuration.
    Consists of one single unit of testing.
"""
import os
import sys
import yaml
import json
import pandas as pd
from evaluation.diversity_metrics import compute_metrics
from collections import Counter
from dotenv import load_dotenv
from pathlib import Path
from sentence_transformers import SentenceTransformer
from rich.console import Console

console = Console()
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

def load_config(path: Path) -> dict:
    """Loads configuration to be used in the generation method."""
    if not path.exists():
        console.log(f"[red]Error:[/] Could not find configuration file at {path}")
        sys.exit(1)
    with path.open("r") as f:
        cfg = yaml.safe_load(f)
    required = ["dataset_path", "llm_used_for_data_generation", "generation_method"]
    for key in required:
        if key not in cfg:
            console.log(f"[red]Error:[/] Missing required field '{key}' in {path}")
            sys.exit(1)
    return cfg

def main(cfg = None, EMB_MODEL = None):
    if cfg is None: 
        cfg = load_config(Path(__file__).parent.parent.parent / "config" / "evaluation.yaml")

    # reading config file
    if EMB_MODEL is None:
        embeddings_model = cfg["emb_name"]
        EMB_MODEL = SentenceTransformer(embeddings_model, device='cuda')    
    llm_model = cfg["llm_used_for_data_generation"]
    dataset_path = Path(cfg["dataset_path"]) / "full"
    gen_strategy = cfg["generation_method"]
    llm_name = llm_model if not '/' in llm_model else llm_model.split('/')[1]
    metrics_to_compute = cfg.get("diversity_metrics", None)  
    console.log(f"Evaluating diversity metrics for '{llm_model}' and generation strategy '{gen_strategy}'.")

    # reading dataset
    utterances_filepath = Path(dataset_path) / gen_strategy / f"{llm_name}.csv"
    train_df = pd.read_csv(utterances_filepath, sep=None, engine="python")
    
    # compute metrics
    metrics = []
    for api_method, group in train_df.groupby('api_method_name'):
        utterances = group['utterance'].dropna().tolist()
        if len(utterances) < 2:
            console.log(f"[yellow]Skipping API method {api_method}: fewer than 2 utterances.[/]")
            continue
        metrics_per_api = compute_metrics(utterances, EMB_MODEL, metrics=metrics_to_compute)
        metrics.append(metrics_per_api)
        console.log(f"Metrics for API method {api_method}: {metrics_per_api}")

    if not metrics:
        console.log("[red]Error:[/] No API method groups had enough utterances to compute metrics.")
        return {}

    metrics_keys = list(metrics[0].keys())
    results = {
        key: round(float(sum(d[key] for d in metrics if key in d) / len(metrics)), 4)
        for key in metrics_keys
    }

    console.log(f"Results: {results}")
    return results

if __name__ == "__main__":
    main()