"""
Process the raw ICNALE dataset into a structured format suitable for analysis.
The base_folder parameter specifies the location of the raw dataset, including:
- Written essays
- CSV describing metadata
"""

import os
import json
import shutil
import random
import pandas as pd
from rich.console import Console

console = Console()

class ICNALEPreProcessing:
    def __init__(self, 
                 base_folder: str): 

        # standard parameters                
        self.base_folder = base_folder
        self.essays_folder = os.path.join(self.base_folder, 'written_essays') # all WE essays .txt files must be here.
        self.metadata_file = os.path.join(self.base_folder, 'ICNALE_Survey_202601.xlsx') # the metadata file must be in the base folder, with this name.
        self.metadata_file_uae = os.path.join(self.base_folder, 'ICNALE WE UAE Module Learner Profile.xlsx') # the metadata file for the UAE module must be in the base folder, with this name.
        self.output_file = os.path.join(self.base_folder, 'processed_icnale.xlsx')
        self.processed_data = []
        self.persona_number = 0

    def process_essays(self):
        "Processes the essays and their metadata into a structured format."
        metadata_df = pd.read_excel(self.metadata_file, sheet_name="Participants")
        metadata_df = metadata_df[(metadata_df['Module'] == 'WE') | (metadata_df['Module'] == 'WEP')]

        for index, row in metadata_df.iterrows():
            module = row['Module']
            code = row['Code']
            country = row['Country'].rstrip().title()
            l1 = row['L1']
            sex = row['Sex']
            age = row['Age']
            major = row['Major/ Occupation']
            proeficiency_level = row['CEFR Level'] if len(row['CEFR Level']) == 4 else "B2_0"
            
            # correcting mistakes for reading files.
            if module == 'WE':
                index = 6
            elif module == 'WEP':
                index = 7

            if l1 == "English":
                continue # skip English L1 for now

            # getting essay files from essays folder
            filename_1 = f"{code[:index]}_PTJ0{code[index:]}_{proeficiency_level}.txt"
            part_time_essay_file_path = os.path.join(self.essays_folder, filename_1)
            if os.path.exists(part_time_essay_file_path):
                with open(part_time_essay_file_path, 'r', encoding='utf-8') as f:
                    part_time_essay_text = f.read()

            filename_2 = f"{code[:index]}_SMK0{code[index:]}_{proeficiency_level}.txt"
            smoking_essay_file_path = os.path.join(self.essays_folder, filename_2)
            if os.path.exists(smoking_essay_file_path):
                with open(smoking_essay_file_path, 'r', encoding='utf-8') as f:
                    smoking_essay_text = f.read()
            
            # define essays content
            if not os.path.exists(part_time_essay_file_path) and not os.path.exists(smoking_essay_file_path):
                print(f"Essay files for code {code} not found. Skipping. Paths: {part_time_essay_file_path}")
                continue
            elif not os.path.exists(part_time_essay_file_path):
                essays = f"Essay about 'non-smoking at restaurants': {smoking_essay_text}"
            elif not os.path.exists(smoking_essay_file_path):
                essays = f"Essay about 'a part-time job for college students': {part_time_essay_text}"
            else:
                essays = f"Essay about 'a part-time job for college students': {part_time_essay_text}\n\nEssay about 'non-smoking at restaurants': {smoking_essay_text}"
            self.persona_number += 1
            self.processed_data.append({
                "persona_number": self.persona_number,
                "code": code,
                "country": country,
                "l1": l1,
                "sex": sex,
                "age": age,
                "major": major,
                "proeficiency_level": proeficiency_level,
                "essays": essays
            })
            
        return self.processed_data

    def process_essays_uae(self):
        "Processes the UAE module metadata and integrates it with the existing processed data."
        metadata_uae_df = pd.read_excel(self.metadata_file_uae)

        for index, row in metadata_uae_df.iterrows():
            code = row['Learner']
            country = "United Arabic Emirates"
            l1 = "Arabic"
            sex = row['Sex']
            age = row['Age']
            major = row['Major']
            proeficiency_level = row['Prof Lev'] if len(row['Prof Lev']) == 4 else "B2_0"

            # getting essay files from essays folder
            filename_1 = f"W_UAE_PTJ0{code[3:]}_{proeficiency_level}.txt"
            part_time_essay_file_path = os.path.join(self.essays_folder, filename_1)
            if os.path.exists(part_time_essay_file_path):
                with open(part_time_essay_file_path, 'r', encoding='utf-8') as f:
                    part_time_essay_text = f.read()
            
            filename_2 = f"W_UAE_SMK0{code[3:]}_{proeficiency_level}.txt"
            smoking_essay_file_path = os.path.join(self.essays_folder, filename_2)
            if os.path.exists(smoking_essay_file_path):
                with open(smoking_essay_file_path, 'r', encoding='utf-8') as f:
                    smoking_essay_text = f.read()

            # define essays content
            if not os.path.exists(part_time_essay_file_path) and not os.path.exists(smoking_essay_file_path):
                print(f"Essay files for code {code} not found. Skipping. Path: {part_time_essay_file_path}")
                continue
            elif not os.path.exists(part_time_essay_file_path):
                essays = f"Essay about 'non-smoking at restaurants': {smoking_essay_text}"
            elif not os.path.exists(smoking_essay_file_path):
                essays = f"Essay about 'a part-time job for college students': {part_time_essay_text}"
            else:
                essays = f"Essay about 'a part-time job for college students': {part_time_essay_text}\n\nEssay about 'non-smoking at restaurants': {smoking_essay_text}"

            self.persona_number += 1
            self.processed_data.append({
                "persona_number": self.persona_number,
                "code": code,
                "country": country,
                "l1": l1,
                "sex": sex,
                "age": age,
                "major": major,
                "proeficiency_level": proeficiency_level,
                "essays": essays
            })

        return self.processed_data
            
    def save_processed_data(self):
        "Saves the processed data as a XLSX file."
        print(f"There are {len(self.processed_data)} personas processed.")
        df = pd.DataFrame(self.processed_data)
        df.to_excel(self.output_file, index=False)
