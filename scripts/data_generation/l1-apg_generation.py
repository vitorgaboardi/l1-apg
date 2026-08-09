"""
Script to generate data using the l1-apg method. 
"""

import os
import sys
import json
import yaml
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from pathlib import Path
from rich.console import Console
from data_generation.l1_apg.utterance_generator import L1APGUtteranceGenerator

env_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)
console = Console()

def load_config(path: Path) -> dict:
    """Loads configuration to be used in the generation method."""
    if not path.exists():
        console.log(f"[red]Error:[/] Could not find configuration file at {path}")
        sys.exit(1)
    with path.open("r") as f:
        cfg = yaml.safe_load(f)
    required = ["oas_path", "output_folder", "llm_temp", "utterances", "llm_name", "llm_url", "api_key", "icnale_path"]
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
    llm_temp = config.get("llm_temp", 1.0)
    number_of_utterances = config.get("utterances", 10)
    number_of_persona = config.get("number_of_persona_per_country", 3)
    icnale_path = config.get("icnale_path")

    # 2 - initialize Utterance Generator
    llm_name = config["llm_name"]
    llm_url = config["llm_url"]
    api_key = os.getenv(config["api_key"])
    save_name = llm_name if not '/' in llm_name else llm_name.split('/')[1]
    output_file = output_folder / "l1-apg" / f"{save_name}.csv"

    # 3 - reading existing output file if exists    
    if (output_file).exists():
        console.log(f"[yellow]Warning:[/] Output file {output_file} already exists. Reading file to continue generation.")
        train_df = pd.read_csv(output_file)
        number_of_methods = len(train_df['api_method_name'].unique())
        utterances_generated = len(train_df)
        personas_used = train_df['persona_number'].unique().tolist()
        console.log(f"[green]Info:[/] Loaded {utterances_generated} utterances from {number_of_methods} API methods.")
    else:
        os.makedirs(output_file.parent, exist_ok=True)
        train_df = pd.DataFrame(columns=["utt_id", "utterance", "api_calls", "api_name", "api_method_name", "documentation", "persona_number", "country", "proficiency_level", "stylistic_analysis", "llm_name"])
        number_of_methods = 0
        utterances_generated = 0
        personas_used = []
    
    utterance_generator = L1APGUtteranceGenerator(icnale_path=icnale_path, temperature=llm_temp, number_of_utterances=number_of_utterances, persona_ids_used=personas_used)
    utterance_generator.set_model(openai_api_key=api_key, base_url=llm_url, model=llm_name)

    # 4 - iterating through all OAS files and generating utterances for each API method
    all_countries = utterance_generator.country_list()
    all_countries.remove("Sri Lanka")
    all_countries.remove("Mongolia")

    for root, _, files in os.walk(oas_path):
        for filename in files:
            # reading the API spec file
            file_path = os.path.join(root, filename)  # path to the API spec file     
            with open(file_path, 'r') as f:
                api = json.load(f) # load the API spec
            api_methods = utterance_generator.get_api_methods(api)

            # generating utterances for each API method
            for method in api_methods: 
                api_id = method["api_method_name"]

                # generating utterances for each country in the country list
                country_done = train_df[train_df['api_method_name'] == api_id]['country'].unique().tolist()
                country_list = [c for c in all_countries if c not in country_done]

                if country_list:
                    for country in country_list:
                        # generating utterances using N random different personas from the same country
                        for _ in range(number_of_persona):
                            utterances, api_calls, persona, stylistic_analysis = utterance_generator.generate_utterance(method, country=country)

                            # saving dataframe with generated utterances
                            for utt, api_call in zip(utterances, api_calls):
                                train_df = pd.concat([train_df, pd.DataFrame([{
                                    "utt_id": utterances_generated,
                                    "utterance": utt,
                                    "api_calls": api_call,
                                    "api_name": method['api_name'],
                                    "api_method_name": method['api_method_name'],
                                    "documentation": method,
                                    "persona_number": persona["persona_number"],
                                    "country": persona["country"],
                                    "proficiency_level": persona["proeficiency_level"],
                                    "stylistic_analysis": stylistic_analysis,
                                    "llm_name": llm_name
                                }])], ignore_index=True)
                                utterances_generated += 1
                            train_df.to_csv(output_file, index=False)

                    number_of_methods += 1
                    console.log(f"[green]Info:[/] Generated utterances for API method {api_id}. Total API methods processed: {number_of_methods}.")


if __name__ == "__main__":
    main()