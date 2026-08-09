"""
Train/test split for all three generation methods.

Strategy
--------
1. l1-apg is split first using stratified sampling over
   (api_method_name × country) groups — this guarantees equal country
   representation in both train and test for every API method.

2. The number of test samples per api_method derived from step 1
   (n_test_per_method = n_countries × n_test_per_stratum) is then
   re-used for toolalpaca and toolalpaca_persona.
   Each of their api_method_name groups is sampled uniformly to match
   the same test count.

3. Output files are named by generation method (not by LLM) and placed
   under data/generated_dataset/train/ and data/generated_dataset/test/.

Configuration is read from config/train_test_split.yaml.
CLI flags can override individual config values.

Usage
-----
    python scripts/preprocessing/train_test_split.py
    python scripts/preprocessing/train_test_split.py --model gpt-5.4 --seed 0
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

FULL_DIR  = PROJECT_ROOT / "data" / "generated_dataset" / "full"
TRAIN_DIR = PROJECT_ROOT / "data" / "generated_dataset" / "train"
TEST_DIR  = PROJECT_ROOT / "data" / "generated_dataset" / "test"
CONFIG_PATH = PROJECT_ROOT / "config" / "train_test_split.yaml"
CSV_SEP = ";"

def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=None, engine="python")


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, sep=CSV_SEP, quoting=csv.QUOTE_ALL)
    print(f"  saved {len(df):>5} rows → {path.relative_to(PROJECT_ROOT)}")


def _allocate_counts_exact(group_sizes: dict[str, int], target_total: int) -> dict[str, int]:
    """
    Allocate integer counts per group summing exactly to target_total,
    proportionally to group sizes (largest-remainder method).
    """
    if not group_sizes:
        return {}

    total = sum(group_sizes.values())
    if total == 0:
        return {k: 0 for k in group_sizes}

    # Clamp to feasible range.
    target_total = max(0, min(target_total, total))

    alloc: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    used = 0

    for key, size in group_sizes.items():
        exact = (size * target_total) / total
        base = int(exact)
        alloc[key] = base
        used += base
        remainders.append((exact - base, key))

    remaining = target_total - used
    remainders.sort(key=lambda x: x[0], reverse=True)

    for _, key in remainders:
        if remaining <= 0:
            break
        if alloc[key] < group_sizes[key]:
            alloc[key] += 1
            remaining -= 1

    return alloc


def _split_l1apg(
    df: pd.DataFrame,
    test_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """
    Stratified split by (api_method_name, country).

    For each API method, computes an exact test target count using
    round(method_total * test_size), then allocates that target across
    countries proportionally to each country's representation.

    This guarantees the exact requested split ratio per API method while
    preserving country balance as much as possible, even with uneven data.

    Returns (train_df, test_df) and prints the stratum stats.
    """
    train_idx: list[int] = []
    test_idx:  list[int] = []

    n_test_per_method: dict[str, int] = {}

    for method, method_df in df.groupby("api_method_name", sort=True):
        method_total = len(method_df)
        method_target_test = int(round(method_total * test_size))
        method_target_test = max(0, min(method_target_test, method_total))

        country_sizes = {
            country: len(grp)
            for country, grp in method_df.groupby("country", sort=True)
        }
        country_test_alloc = _allocate_counts_exact(country_sizes, method_target_test)

        for country, grp in method_df.groupby("country", sort=True):
            n_test = country_test_alloc.get(country, 0)
            if n_test > 0:
                sampled_test = grp.sample(n=n_test, random_state=seed)
            else:
                sampled_test = grp.iloc[0:0]
            sampled_train = grp.drop(sampled_test.index)

            test_idx.extend(sampled_test.index.tolist())
            train_idx.extend(sampled_train.index.tolist())

        n_test_per_method[method] = method_target_test

    train_df = df.loc[train_idx].reset_index(drop=True)
    test_df  = df.loc[test_idx].reset_index(drop=True)

    n_countries = df["country"].nunique()
    print(f"  l1-apg: {n_countries} countries | exact method-level allocation")
    _print_split_summary("l1-apg", train_df, test_df)
    return train_df, test_df, n_test_per_method


def _split_by_method_count(
    df: pd.DataFrame,
    n_test_per_method: dict[str, int],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each api_method_name group, sample exactly n_test_per_method[method]
    rows for test. This matches the test count derived from l1-apg.
    """
    train_idx: list[int] = []
    test_idx:  list[int] = []

    for method, grp in df.groupby("api_method_name", sort=True):
        default_target = int(round(len(grp) * 0.2))
        n_test = n_test_per_method.get(method, default_target)
        n_test = max(0, min(n_test, len(grp)))
        sampled_test = grp.sample(n=n_test, random_state=seed)
        sampled_train = grp.drop(sampled_test.index)

        test_idx.extend(sampled_test.index.tolist())
        train_idx.extend(sampled_train.index.tolist())

    train_df = df.loc[train_idx].reset_index(drop=True)
    test_df  = df.loc[test_idx].reset_index(drop=True)
    return train_df, test_df


def _print_split_summary(name: str, train: pd.DataFrame, test: pd.DataFrame) -> None:
    total = len(train) + len(test)
    print(f"  {name}: total={total}  train={len(train)}  test={len(test)}  "
          f"({len(test)/total:.1%} test)")
    # per-api_method balance check
    method_counts = (
        pd.concat([
            train.groupby("api_method_name").size().rename("train"),
            test.groupby("api_method_name").size().rename("test"),
        ], axis=1)
        .fillna(0)
        .astype(int)
    )
    print(method_counts.to_string())


def _load_config() -> dict:
    """Load config/train_test_split.yaml and return as a dict."""
    if not CONFIG_PATH.exists():
        sys.exit(f"[ERROR] Config file not found: {CONFIG_PATH}")
    with CONFIG_PATH.open() as fh:
        return yaml.safe_load(fh)


def main() -> None:
    cfg = _load_config()

    # CLI flags can override YAML values
    parser = argparse.ArgumentParser(description="Create balanced train/test splits.")
    parser.add_argument(
        "--model",
        default=cfg.get("llm_used_for_data_generation"),
        help="LLM name used to select the input CSV file (overrides YAML).",
    )
    parser.add_argument(
        "--test-size", type=float,
        default=1.0 - float(cfg.get("data_split", 0.8)),
        help="Fraction of data to use for the test set (overrides YAML data_split).",
    )
    parser.add_argument(
        "--seed", type=int,
        default=cfg.get("seed", 42),
        help="Random seed for reproducibility (overrides YAML).",
    )
    args = parser.parse_args()

    model      = args.model
    test_size  = args.test_size
    seed       = args.seed
    methods    = cfg.get("generation_methods", ["l1-apg", "toolalpaca_persona", "toolalpaca"])

    if not model:
        sys.exit("[ERROR] No model specified in config or via --model.")

    print(f"\nTrain/test split  model={model}  test_size={test_size}  seed={seed}")
    print(f"methods={methods}")
    print("=" * 60)

    n_test_per_method: dict[str, int] = {}

    # ---- l1-apg must be processed first to derive per-method test counts --
    if "l1-apg" in methods:
        l1_path = FULL_DIR / "l1-apg" / f"{model}.csv"
        if not l1_path.exists():
            sys.exit(f"[ERROR] l1-apg file not found: {l1_path}")
        print("\n[l1-apg]")
        df_l1 = _read_csv(l1_path)
        train_l1, test_l1, n_test_per_method = _split_l1apg(df_l1, test_size, seed)
        _write_csv(train_l1, TRAIN_DIR / "l1-apg.csv")
        _write_csv(test_l1,  TEST_DIR  / "l1-apg.csv")

    # ---- remaining methods -----------------------------------------------
    for method in methods:
        if method == "l1-apg":
            continue
        method_path = FULL_DIR / method / f"{model}.csv"
        if not method_path.exists():
            print(f"\n[{method}] NOT FOUND for model '{model}', skipping.")
            continue
        print(f"\n[{method}]")
        df = _read_csv(method_path)
        train_df, test_df = _split_by_method_count(df, n_test_per_method, seed)
        _print_split_summary(method, train_df, test_df)
        _write_csv(train_df, TRAIN_DIR / f"{method}.csv")
        _write_csv(test_df,  TEST_DIR  / f"{method}.csv")

    print("\nDone.")


if __name__ == "__main__":
    main()
