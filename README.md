# Overview

This repository contains the code and data of the **L1-APG** pipeline, generating API-oriented utterances reflecting linguistic patterns of non-native English speakers, as part of the research paper entitled: **Beyond Standard English: L1-Aware Benchmarks for Evaluating API-calling in LLM Agents**.

# Installation Instructions

To set up the environment, please follow these steps:

1. Download this repository
2. Create a conda virtual environment and activate it:
   ```bash
    conda env create -f environment.yml
    conda activate l1-apg
    ```

# ICNALE Dataset

## Data Preparation

**L1-APG** generates utterances based on real essays provided by the [ICNALE dataset](https://language.sakura.ne.jp/icnale/). The essays should be downloaded from the original source, and the following files should be placed in the `data/icnale/` directory:

- `ICNALE_Survey_202601.xlsx`: File containing metadata about the writers, including country, first language, proficiency level, etc.
- `ICNALE WE UAE Module Learner Profile.xlsx`: File containing metadata about the writers from UAE, released in a second version of the dataset.
- `written_essays`: Folder containing all the essays in .txt format provided by the original dataset. The following modules must be downloaded first: `Written Essays (WE) v2.6`, `Written Essays Plus (WEP) v0.7`, and `Written Essays UAE (WE_UAE) v1.0`. Then, copy all .txt files from the `0_Unclassified_Unmerged` folder to the `data/icnale/written_essays/` folder in this repository.

After placing the files in the correct directory, run the following pre-processing script:

```bash
python scripts/preprocessing/icnale_preprocessing.py
```

After running the above command, the `processed_icnale.xlsx` file will be created in the `data/icnale/` directory, containing the pre-processed essays and metadata required by the L1-APG pipeline in a single file.

We do not make the pre-processed file available in the repository due to the original dataset's license, which does not allow redistribution of the processed data. However, the pre-processing script is provided to allow users to generate the processed file on their own.

## Data Analysis

To perform an analysis of the linguistic patterns of the essays from ICNALE, run the following script:

```bash
python scripts/preprocessing/icnale_linguistic_analysis.py
```

This script computes a 21-dimensional linguistic feature vector for each essay based on the features described in the paper and then computes the average cosine distance between essays from the same country and essays from different countries. An image representing a heatmap of distances between countries is generated and saved in the `data/icnale/` directory as `country_distance_matrix.png`, and the distance matrix is also saved as an Excel file named `country_distance_matrix.xlsx` in the same directory.

# Data Generation

The configuration file must be set with the dataset paths and LLM generation settings. The configuration file is located at `config/utterance_generation.yaml`. Paths should be updated according to the local setup.

It is also necessary to create a `.env` file in the project root directory with the following content:

```
OPENAI_API_KEY=<openai_api_key>
DEEPINFRA_API_KEY=<deepinfra_api_key>
```

The API documentation can be found in the `data/apis` directory, which contains the API specifications used for generating the utterances.

## L1-APG Pipeline

We provide a script to generate data using the L1-APG pipeline, which generates API-oriented utterances by conditioning the generation on the essays from the ICNALE dataset. Implementation details about this pipeline can be found in the `src/data_generation/l1_apg` folder. 

To generate utterances using the L1-APG pipeline, execute the following command:

```bash
python scripts/data_generation/l1-apg_generation.py
```

This will save a csv file with the name of the LLM used to generate the data under the `data/generated_dataset/full/l1-apg` directory, which contains the generated utterances and their corresponding metadata, such as the country of the persona, the original essay, etc.

## P-ALP Pipeline

We provide a script to generate data using the P-ALP pipeline, which generates API-oriented utterances by extending the [ToolAlpaca](https://github.com/tangqiaoyu/ToolAlpaca) prompt by conditioning the generation on persona descriptions sampled from [PersonaHub](https://github.com/tencent-ailab/persona-hub). Implementation details about this pipeline can be found in the `src/data_generation/persona_toolalpaca` folder.

This pipeline requires the [PersonaHub](https://github.com/tencent-ailab/persona-hub) dataset. Download it from the original source and place the `persona.jsonl` file in the `data/personahub/` directory.

To generate utterances using the P-ALP pipeline, execute the following command:

```bash
python scripts/data_generation/persona_toolalpaca.py
```

This will save a csv file with the name of the LLM used to generate the data under the `data/generated_dataset/full/toolalpaca_persona` directory, which contains the generated utterances and their corresponding metadata, such as the persona description used for conditioning, etc.

## ALP Pipeline

We provide a script to generate data using the ALP pipeline, which generates API-oriented utterances by using the [ToolAlpaca](https://github.com/tangqiaoyu/ToolAlpaca) prompt without any persona conditioning. Implementation details about this pipeline can be found in the `src/data_generation/toolalpaca` folder.

To generate utterances using the ALP pipeline, execute the following command:

```bash
python scripts/data_generation/toolalpaca.py
```

This will save a csv file with the name of the LLM used to generate the data under the `data/generated_dataset/full/toolalpaca` directory, which contains the generated utterances and their corresponding metadata.

# Evaluation

We evaluate the generated datasets according to three perspectives: (i) linguistic patterns, which evaluate whether L1-APG utterances show distinct linguistic patterns across countries; (ii) linguistic diversity, which evaluates the lexical, syntactic, and semantic diversity of the generated utterances; and (iii) LLM agent evaluation, which evaluates whether LLM agents can correctly call the APIs described in the utterances.

## Evaluation Requirements

- The dataset root in `config/evaluation.yaml` must point to `data/generated_dataset` (or an equivalent local path with the same structure).
- The full evaluation workflow requires `data/generated_dataset/test/{generation_method}.csv` files, because agent evaluation uses the test split.
- Agent evaluation consumes precomputed agent outputs from `results/agent/{generation_method}/{agent_model}.jsonl`.
- Linguistic diversity loads a sentence-transformer with `device='cuda'` in `scripts/evaluation/full_evaluation.py`, so a CUDA-capable environment is required for the default full evaluation pipeline.

Note: This repository already provides the generated datasets and the corresponding LLM agent output files. Therefore, evaluation can be executed directly on the provided artifacts, without regenerating data or rerunning agent inference.

## Recommended Execution Order

1. Generate datasets with the data generation scripts.
2. Generate agent outputs on the test split:

```bash
python scripts/agent/run_agent.py
```

3. Run the full evaluation pipeline:

```bash
python scripts/evaluation/full_evaluation.py
```

This will run all three evaluations and save the results as a multi-sheet spreadsheet at `results/results.xlsx`, with one sheet per evaluation type. A JSON summary is also saved under `results/evaluation/`. The LLMs and generation strategies to evaluate can be configured in `config/full_evaluation.yaml`.

Note: `scripts/evaluation/full_evaluation.py` does not generate agent outputs automatically. Agent outputs must be created first with `scripts/agent/run_agent.py`.

It is also possible to run each evaluation separately by executing the corresponding script in the `scripts/evaluation/` directory:

- **Linguistic patterns** (L1-APG only):
  ```bash
  python scripts/evaluation/linguistic_patterns.py
  ```
- **Linguistic diversity**:
  ```bash
  python scripts/evaluation/linguistic_diversity.py
  ```
- **LLM agent evaluation**:
  ```bash
  python scripts/evaluation/agent_evaluation.py
  ```

When running a script individually, the configuration file `config/evaluation.yaml` must be set with the LLM used for data generation, the generation strategy, and the paths to the datasets.


# Agent Implementation

We implement an LLM agent to solve API-related utterances. The agent implementation can be seen in `src/agent` folder.


## APIs

We generated Python functions that implement all APIs defined in `data/apis`. 

Some APIs require API keys. Therefore, to test the agent under all possible APIs, create a `.env` file in the project root directory with the following content:

```
WEATHER_API_KEY="<weather_api_key>"
TMDB_API_KEY="<tmdb_api_key>"
GEOPIFY_API_KEY="<geopify_api_key>"
FOOD_API_KEY="<food_api_key>"
```

More information about the APIs can be found in the following links: [Weather](https://www.weatherapi.com/docs/), [TMDB](https://developer.themoviedb.org/reference/intro/getting-started), [Geopify](https://apidocs.geoapify.com/docs/routing/#quick-start) and [Food](https://fdc.nal.usda.gov/api-guide).

## Models

We deployed LLM agents locally using vLLM and HuggingFace. The agent definition to be used must be configured in `config/evaluation.yaml`. 

To test an agent working after deploying it locally or using external APIs, use the following script: 

```bash
python scripts/agent/test_agent.py
```


# Reproducing Results

Due to the stochastic nature of LLM generation, it is not possible to exactly reproduce the generated utterances. However, by running the generation and evaluation scripts as described above, similar trends and relative performance across LLMs and generation strategies should be observed.

To ensure reproducibility of the reported results, generated datasets are provided under `data/generated_dataset/full/`, and split files are provided under `data/generated_dataset/train/` and `data/generated_dataset/test/`. Precomputed LLM agent outputs are also provided under `results/agent/`. These artifacts were generated using a fixed random seed and are intended to support direct evaluation.

By running the evaluation scripts on these provided datasets and agent outputs, the results reported in the paper can be reproduced directly. Execute:

```bash
python scripts/evaluation/full_evaluation.py
```

If regenerating agent outputs is desired (for example, when changing the agent model or serving endpoint), run:

```bash
python scripts/agent/run_agent.py
```

The results will be saved as `results/results.xlsx`, which should contain the same results as reported in the paper for the corresponding LLMs and generation strategies.