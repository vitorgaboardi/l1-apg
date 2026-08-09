"""
Generates utterances using the L1-APG prompting pipeline. 
The class receives information about an API method and a persona and generates natural language utterances. 
"""

import json
import ast
import copy
import os
import numpy as np
import random
import itertools
import re
import pandas as pd
from typing import Dict, List
from openai import OpenAI
from .prompts import L1_APG_UTTERANCE_GENERATOR_PROMPT

random.seed(42)

class L1APGUtteranceGenerator:
    def __init__(self, 
                 icnale_path: str,
                 temperature: float = 1.0,
                 number_of_utterances: int = 10,
                 persona_ids_used: List[int] = [],
                 max_retries: int = 5):
        self.model = None
        self.openai_client = None        
        self.temperature = temperature
        self.max_retries = max_retries
        self.number_of_utterances = number_of_utterances

        # icnale dataset
        self.icnale_df = pd.read_excel(icnale_path)
        self.persona_ids_used = persona_ids_used
        self.countries = self.icnale_df['country'].unique().tolist()
        self.current_country = 0
        print(f"Loaded ICNALE dataset with {len(self.icnale_df)} entries.")
        print(f"Available countries: {self.countries}")

    def country_list(self) -> List[str]:
        return self.countries

    def set_model(self, openai_api_key: str, base_url: str = "https://api.openai.com/v1", model: str = "gpt-4o"):
        self.openai_client = OpenAI(api_key=openai_api_key, base_url=base_url)
        self.model = model

    def select_persona(self, country: str = None) -> Dict:
        # Cycle through countries to ensure diversity
        if country is None:
            country = self.countries[self.current_country]
            self.current_country = (self.current_country + 1) % len(self.countries)

        country_df = self.icnale_df[self.icnale_df['country'] == country]
        available_personas = country_df[~country_df['persona_number'].isin(self.persona_ids_used)]

        if available_personas.empty:
            print(f"All personas from {country} have been used. Moving for next country.")
            self.countries.remove(country)
            return self.select_persona()

        selected_row = available_personas.sample(n=1, random_state=42).iloc[0]
        self.persona_ids_used.append(selected_row['persona_number'])

        persona = {
            "persona_number": int(selected_row['persona_number']),
            "l1": selected_row['l1'],
            "age": int(selected_row['age']),
            "sex": "Male" if selected_row['sex'] == "M" else "Female",
            "country": selected_row['country'],
            "major": selected_row['major'],
            "proeficiency_level": selected_row['proeficiency_level'],
            "essays": selected_row['essays']
        }

        return persona

    def generate_utterance(self, api_method: Dict, country: str = None) -> str:
        if self.openai_client is None or self.model is None:
            raise ValueError("OpenAI client and model must be set before generating utterances.")
        persona = self.select_persona(country)
        prompt = self._build_prompt(persona, str(api_method))
        regex_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
        
        for attempt in range(self.max_retries):
            try:
                response = self.openai_client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature
                )
                data = response.choices[0].message.content.strip()
                data = data.replace("```json", "").replace("```", "").strip()  # Remove code block markers
                data = json.loads(data)
                stylistic_analysis = data["stylistic_analysis"]

                processed_utterances = []
                processed_api_calls = []
                if isinstance(data["data"], list):
                    for instance in data["data"]:
                        if isinstance(instance, dict) and "utterance" in instance:
                            utt = instance["utterance"].strip()
                            api_call = instance.get("api_call", [])
                        processed_utterances.append(utt)
                        processed_api_calls.append(api_call)
                
                print(f"Stylistic Analysis provided by the LLM: {stylistic_analysis}")
                print("="*100)
                print("Generated Utterances by the LLM:")
                for index, utt in enumerate(processed_utterances):
                    print(f"{index + 1}: {utt}")
                print("="*100)

                return processed_utterances, processed_api_calls, persona, stylistic_analysis
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {response.choices[0].message.content}")
        raise RuntimeError("Max retries exceeded for utterance generation.")


    def get_api_methods(self, oas: Dict) -> List[Dict]:
        api_methods = []
        for path in oas.get("api_list", {}):
            api_method = {
                "api_name": oas.get("tool_name", ''),
                "api_description": oas.get("tool_description", ''),
                "api_method_name": path.get("name", ''),
                "api_method_description": path.get("description", ''),
                "api_method_parameters": path.get("parameters", ''),
            }
            api_methods.append(api_method)
        return api_methods        


    def _build_prompt(self, persona: Dict, api_method: str) -> str:
        if persona['l1'] == 'English':
            persona_description = f"You are a {persona['age']} year old {persona['sex']} from {persona['country']} who speaks {persona['l1']} as your first language. You have a major in {persona['major']}."
        else:
            persona_description = f"You are a {persona['age']} year old {persona['sex']} from {persona['country']} who speaks {persona['l1']} and has a proficiency level of {persona['proeficiency_level']} in English. Your job involves working with {persona['major']}." # used in v4
        
        # building the final prompt
        prompt = L1_APG_UTTERANCE_GENERATOR_PROMPT.format(
            persona_description=persona_description,
            essay_examples=persona['essays'],
            number_of_utterances=self.number_of_utterances,
            api_method_description=api_method
        )

        return prompt
