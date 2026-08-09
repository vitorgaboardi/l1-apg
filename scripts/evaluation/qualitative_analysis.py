"""
Overview:
    Qualitative annotation task helper.
    Randomly selects N personas (maximising country diversity) across a list of
    models, joins each with the persona's actual ICNALE essays, and produces a
    CSV ready for human annotation with an empty 'label' column (yes/no).

Output columns:
    persona_id, country, model, utterance_id, persona_essays,
    linguistic_analysis, utterance, label
"""

import random
from pathlib import Path
import pandas as pd
from openpyxl.styles import Alignment, PatternFill, Font
from openpyxl.utils import get_column_letter
from rich.console import Console

console = Console()
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

ICNALE_PATH = PROJECT_ROOT / "data/icnale/processed_icnale.xlsx"
DATASET_DIR = PROJECT_ROOT / "data/generated_dataset/full/l1-apg"
OUTPUT_PATH = PROJECT_ROOT / "results/qualitative_annotation.xlsx"

# parameters
MODEL_FILES = [
    "gpt-5.4.csv",
    "gpt-4o-mini.csv",
    "DeepSeek-V4-Pro.csv",
    "Llama-4-Maverick-17B-128E-Instruct-Turbo.csv",
    "claude-4-sonnet.csv",
]
N_PERSONAS = 20
EXCLUDE_COUNTRIES = {"Mongolia", "Sri Lanka"}
RANDOM_SEED = 42


def load_model_data(model_files: list[str]) -> pd.DataFrame:
    frames = []
    for fname in model_files:
        fpath = DATASET_DIR / fname
        if not fpath.exists():
            console.log(f"[yellow]Warning:[/] {fpath} not found – skipping.")
            continue
        df = pd.read_csv(fpath)
        # Normalise column names across pipeline versions
        if "persona_number" in df.columns and "persona_id" not in df.columns:
            df = df.rename(columns={"persona_number": "persona_id"})
        if "stylistic_analysis" in df.columns and "linguistic_analysis" not in df.columns:
            df = df.rename(columns={"stylistic_analysis": "linguistic_analysis"})
        if "llm_name" in df.columns and "model" not in df.columns:
            df = df.rename(columns={"llm_name": "model"})
        if "utt_id" in df.columns and "utterance_id" not in df.columns:
            df = df.rename(columns={"utt_id": "utterance_id"})
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No model CSV files found in {DATASET_DIR}")
    return pd.concat(frames, ignore_index=True)


def sample_personas_by_country(
    all_persona_country: pd.DataFrame,
    n: int,
    rng: random.Random,
) -> list[int]:
    countries = [c for c in all_persona_country["country"].unique() if c not in EXCLUDE_COUNTRIES]
    rng.shuffle(countries)

    # Build per-country persona lists
    per_country: dict[str, list[int]] = {}
    for country in countries:
        ids = all_persona_country[all_persona_country["country"] == country]["persona_id"].unique().tolist()
        rng.shuffle(ids)
        per_country[country] = ids

    selected: list[int] = []
    seen: set[int] = set()

    # Round-robin
    while len(selected) < n:
        progress = False
        for country in countries:
            if len(selected) >= n:
                break
            pool = per_country.get(country, [])
            for pid in pool:
                if pid not in seen:
                    selected.append(pid)
                    seen.add(pid)
                    per_country[country] = [p for p in pool if p != pid]
                    progress = True
                    break
        if not progress:
            break  # all personas exhausted

    return selected[:n]


def main() -> None:
    rng = random.Random(RANDOM_SEED)

    # 1 – Load ICNALE essays
    console.log(f"Loading ICNALE from {ICNALE_PATH}")
    icnale_df = pd.read_excel(ICNALE_PATH)
    icnale_df = icnale_df.rename(columns={"persona_number": "persona_id"})
    icnale_df = icnale_df[~icnale_df["country"].isin(EXCLUDE_COUNTRIES)]
    icnale_lookup = icnale_df.set_index("persona_id")[["country", "essays"]].to_dict("index")

    # 2 – Load all model utterances
    console.log(f"Loading model datasets from {DATASET_DIR}")
    gen_df = load_model_data(MODEL_FILES)
    gen_df = gen_df[~gen_df["country"].isin(EXCLUDE_COUNTRIES)]

    # Keep only personas present in both sources
    common_personas = set(icnale_lookup.keys()) & set(gen_df["persona_id"].unique())
    gen_df = gen_df[gen_df["persona_id"].isin(common_personas)]

    # 3 – Sample N personas maximising country diversity
    persona_country = gen_df[["persona_id", "country"]].drop_duplicates()
    selected_persona_ids = sample_personas_by_country(persona_country, N_PERSONAS, rng)
    console.log(f"Selected {len(selected_persona_ids)} personas: {selected_persona_ids}")

    # 4 – For each persona × model, sample utterances and build annotation rows
    rows = []
    gen_df_indexed = gen_df[gen_df["persona_id"].isin(selected_persona_ids)]

    for persona_id in selected_persona_ids:
        persona_info = icnale_lookup.get(persona_id, {})
        persona_essays = persona_info.get("essays", "")
        country = persona_info.get("country", "")

        persona_gen = gen_df_indexed[gen_df_indexed["persona_id"] == persona_id]

        for model_name in persona_gen["model"].unique():
            model_utts = persona_gen[persona_gen["model"] == model_name]

            for _, row in model_utts.iterrows():
                rows.append({
                    "persona_id": persona_id,
                    "country": country,
                    "model": model_name,
                    "utterance_id": row.get("utterance_id", ""),
                    "persona_essays": persona_essays,
                    "linguistic_analysis": row.get("linguistic_analysis", ""),
                    "utterance": row.get("utterance", ""),
                    "label": "",  # to be filled by human annotator
                })

    if not rows:
        console.log("[red]Error:[/] No rows were generated. Check model CSV paths and persona overlap.")
        return

    output_df = pd.DataFrame(rows)

    output_df = output_df.sort_values(["persona_id", "model"], kind="stable").reset_index(drop=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # --- Write with openpyxl formatting ---
    # Column widths (characters): narrow for IDs, wide for long-text fields.
    COL_WIDTHS = {
        "persona_id": 12,
        "country": 16,
        "model": 22,
        "utterance_id": 14,
        "persona_essays": 70,
        "linguistic_analysis": 60,
        "utterance": 60,
        "label": 12,
    }
    WRAP_COLS = {"persona_essays", "linguistic_analysis", "utterance"}

    with pd.ExcelWriter(OUTPUT_PATH, engine="openpyxl") as writer:
        output_df.to_excel(writer, index=False, sheet_name="Annotation")
        ws = writer.sheets["Annotation"]

        # Header style
        header_fill = PatternFill("solid", fgColor="2F4858")
        header_font = Font(bold=True, color="FFFFFF")
        for cell in ws[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(wrap_text=True, vertical="center")

        # Column widths + wrap for data cells
        for col_idx, col_name in enumerate(output_df.columns, start=1):
            col_letter = get_column_letter(col_idx)
            ws.column_dimensions[col_letter].width = COL_WIDTHS.get(col_name, 20)
            if col_name in WRAP_COLS:
                for cell in ws[col_letter][1:]:  # skip header
                    cell.alignment = Alignment(wrap_text=True, vertical="top")

        # Row heights: taller for data rows so wrapped text is visible
        for row_idx in range(2, ws.max_row + 1):
            ws.row_dimensions[row_idx].height = 150

        # Freeze the header row
        ws.freeze_panes = "A2"

    console.log(f"[green]Done.[/] Annotation file saved to {OUTPUT_PATH}")
    console.log(f"Total rows: {len(output_df)} | Personas: {output_df['persona_id'].nunique()} | Models: {output_df['model'].nunique()} | Countries: {output_df['country'].nunique()}")


if __name__ == "__main__":
    main()
