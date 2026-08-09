"""
Generates utterances using the ToolAlpaca + PersonaHub prompting pipeline. 
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
from .prompts import TOOLALPACA_PERSONA_PROMPT_UTTERANCE_GENERATION

random.seed(42)

class ToolAlpacaPersonaUtteranceGenerator:
    def __init__(self, 
                 personahub_path: str,
                 temperature: float = 1.0,
                 number_of_utterances: int = 10,
                 persona_ids_used: List[int] = [],
                 max_retries: int = 5):
        self.model = None
        self.openai_client = None        
        self.temperature = temperature
        self.max_retries = max_retries
        self.number_of_utterances = number_of_utterances

        # personahub dataset processing
        with open(personahub_path, "r") as f:
            personas = [json.loads(line) for line in f]
        self.personahub_df = pd.DataFrame(personas)
        self.persona_ids_available = list(range(1, len(self.personahub_df) + 1))
        random.shuffle(self.persona_ids_available)
        self.persona_ids_available = [pid for pid in self.persona_ids_available if pid not in persona_ids_used]
        print(f"Loaded PersonaHub dataset with {len(self.personahub_df)} entries.")

    def set_model(self, openai_api_key: str, base_url: str = "https://api.openai.com/v1", model: str = "gpt-4o"):
        self.openai_client = OpenAI(api_key=openai_api_key, base_url=base_url)
        self.model = model

    def get_api_methods(self, oas: Dict) -> List[Dict]:
        api_methods = []
        for path in oas.get("api_list", {}):
            api_method = {
                "api_name": oas.get("name", ''),
                "api_description": oas.get("tool_description", ''),
                "api_method_name": path.get("name", ''),
                "api_method_description": path.get("description", ''),
                "api_method_required_parameters": path.get("parameters", ''),
            }
            api_methods.append(api_method)
        return api_methods

    def select_persona(self) -> Dict:
        id = self.persona_ids_available.pop(0)
        selected_row = self.personahub_df.iloc[id - 1]
        
        persona = {
            "persona_number": id,
            "description": selected_row['persona'],
        }
        return persona

    def generate_utterance(self, api_method: Dict) -> tuple[List[str], List[List[Dict]], Dict]:
        if self.openai_client is None or self.model is None:
            raise ValueError("OpenAI client and model must be set before generating utterances.")
        persona = self.select_persona()
        prompt = self._build_prompt(persona, str(api_method))
        
        for attempt in range(self.max_retries):
            try:
                response = self.openai_client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature
                )
                content = response.choices[0].message.content.strip()
                content = content.replace("```json", "").replace("```", "").strip()  # Remove code block markers
                payload = json.loads(content)

                if isinstance(payload, dict):
                    data_items = payload.get("data", [])
                elif isinstance(payload, list):
                    data_items = payload
                else:
                    data_items = []

                processed_utterances = []
                processed_api_calls = []
                if isinstance(data_items, list):
                    for item in data_items:
                        if isinstance(item, dict):
                            utt = str(item.get("utterance", "")).strip()
                            api_call = item.get("api_call", [])
                        else:
                            utt = str(item).strip()
                            api_call = []
                        processed_utterances.append(utt)
                        processed_api_calls.append(api_call)
                
                print(f"Generated {len(processed_utterances)} utterances for API {api_method['api_name']} method {api_method['api_method_name']} with the following persona: {persona['description']} (ID: {persona['persona_number']}):")
                for index, utt in enumerate(processed_utterances):
                    print(f"{index + 1}: {utt}")
                print("="*150)

                return processed_utterances, processed_api_calls, persona
            except Exception as e:
                response_text = ""
                if 'response' in locals() and response and response.choices:
                    response_text = response.choices[0].message.content
                print(f"Attempt {attempt + 1} failed: {response_text}")
        raise RuntimeError("Max retries exceeded for utterance generation.")

    def _build_prompt(self, persona: Dict, api_method: str) -> str:
        prompt = TOOLALPACA_PERSONA_PROMPT_UTTERANCE_GENERATION.format(
            persona_description=persona["description"],
            number_of_utterances=self.number_of_utterances,
            api_method_description=api_method
        )

        return prompt