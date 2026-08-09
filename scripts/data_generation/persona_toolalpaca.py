import os
import sys
import json
import csv
import yaml
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path
from rich.console import Console
from data_generation.persona_toolalpaca.utterance_generator import ToolAlpacaPersonaUtteranceGenerator

env_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)
console = Console()

CSV_SEPARATOR = ";"


def save_output(df: pd.DataFrame, output_file: Path) -> None:
    """Persist with explicit quoting so punctuation in utterances does not break columns."""
    df.to_csv(
        output_file,
        index=False,
        sep=CSV_SEPARATOR,
        quoting=csv.QUOTE_ALL,
    )

def load_config(path: Path) -> dict:
    """Loads configuration to be used in the generation method."""
    if not path.exists():
        console.log(f"[red]Error:[/] Could not find configuration file at {path}")
        sys.exit(1)
    with path.open("r") as f:
        cfg = yaml.safe_load(f)
    required = [
        "oas_path",
        "output_folder",
        "llm_temp",
        "utterances",
        "llm_name",
        "llm_url",
        "api_key",
        "personahub_path",
    ]
    for key in required:
        if key not in cfg:
            console.log(f"[red]Error:[/] Missing required field '{key}' in {path}")
            sys.exit(1)
    return cfg


def main():
    # 1 - load configuration
    config_path = Path(__file__).resolve().parent.parent.parent / "config" / "utterance_generation.yaml"
    config = load_config(config_path)

    oas_path = Path(config["oas_path"])
    output_folder = Path(config["output_folder"])
    number_of_utterances = config.get("utterances", 10)
    number_of_persona = config.get("number_of_persona_from_personahub", 60)
    personahub_path = config.get("personahub_path")
    
    llm_temp = config.get("llm_temp", 1.0)
    llm_name = config["llm_name"]
    llm_url = config["llm_url"]
    api_key = os.getenv(config["api_key"])
    if not api_key:
        console.log(f"[red]Error:[/] API key env var '{config['api_key']}' is missing or empty.")
        sys.exit(1)
    save_name = llm_name if not '/' in llm_name else llm_name.split('/')[1]
    output_file = output_folder / "toolalpaca_persona" / f"{save_name}.csv"    
    target_columns = [
        "utt_id",
        "utterance",
        "api_calls",
        "api_name",
        "api_method_name",
        "documentation",
        "persona_number",
        "persona_description",
        "llm_name",
    ]

    # 2 - reading existing output file if exists    
    if (output_file).exists():
        console.log(f"[yellow]Warning:[/] Output file {output_file} already exists. Reading file to continue generation.")
        train_df = pd.read_csv(output_file, sep=None, engine="python")
        for col in target_columns:
            if col not in train_df.columns:
                train_df[col] = ""
        train_df = train_df[target_columns]
        number_of_api_methods = train_df["api_method_name"].nunique()
        personas_used = [int(x) for x in train_df["persona_number"].dropna().unique().tolist()]
        if train_df["utt_id"].astype(str).str.strip().ne("").any():
            utt_id_counter = int(pd.to_numeric(train_df["utt_id"], errors="coerce").max()) + 1
        else:
            utt_id_counter = len(train_df)
        console.log(f"[green]Info:[/] Loaded {len(train_df)} utterances from {number_of_api_methods} API methods.")
    else:
        os.makedirs(output_file.parent, exist_ok=True)
        train_df = pd.DataFrame(columns=target_columns)
        number_of_api_methods = 0
        personas_used = []
        utt_id_counter = 0

    utterance_generator = ToolAlpacaPersonaUtteranceGenerator(personahub_path=personahub_path, temperature=llm_temp, number_of_utterances=number_of_utterances, persona_ids_used=personas_used)
    utterance_generator.set_model(openai_api_key=api_key, base_url=llm_url, model=llm_name)

    for root, _, files in os.walk(oas_path):
        for filename in files:
            # reading the API spec file
            file_path = os.path.join(root, filename)  # path to the API spec file
            with open(file_path, 'r') as f:
                api = json.load(f) # load the API spec
            api_methods = utterance_generator.get_api_methods(api)

            # generating utterances for each API method
            for method in api_methods:
                api_name = method.get("api_name", "")
                api_method_name = method.get("api_method_name", "")
                expected_instances = number_of_utterances * number_of_persona
                instances = len(
                    train_df[
                        (train_df["api_name"] == api_name)
                        & (train_df["api_method_name"] == api_method_name)
                    ]
                )

                # checking if the API method was already processed
                if instances < expected_instances:
                    # generating utterances using N random different personas
                    for _ in range(number_of_persona):
                        utterances, api_calls, persona = utterance_generator.generate_utterance(method)

                        for i, utt in enumerate(utterances):
                            api_call = api_calls[i] if i < len(api_calls) else []
                            train_df = pd.concat([train_df, pd.DataFrame({
                                "utt_id": [utt_id_counter],
                                "utterance": [utt],
                                "api_calls": [json.dumps(api_call, ensure_ascii=False)],
                                "api_name": [api_name],
                                "api_method_name": [api_method_name],
                                "documentation": [json.dumps(method, ensure_ascii=False)],
                                "persona_number": [persona['persona_number']],
                                "persona_description": [persona['description']],
                                "llm_name": [save_name]
                            })], ignore_index=True)
                            utt_id_counter += 1
                        save_output(train_df, output_file)

                console.log(f"[green]Info:[/] Generated utterances for API {api_name} method {api_method_name}.")
                number_of_api_methods += 1

if __name__ == "__main__":
    main()