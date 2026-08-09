"""
Preprocessing script for the ICNALE dataset. It processes the essays and saves the processed data in a format suitable for training machine learning models.
"""
from pathlib import Path
from preprocessing.icnale import ICNALEPreProcessing

project_root = Path(__file__).resolve().parent.parent.parent
base_folder = project_root / 'data' / 'icnale' # path to the ICNALE dataset
processor = ICNALEPreProcessing(base_folder)
processor.process_essays()
processor.process_essays_uae()
processor.save_processed_data()