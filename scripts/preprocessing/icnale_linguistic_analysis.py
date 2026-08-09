import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from pathlib import Path
from scipy.spatial.distance import cosine
from sklearn.preprocessing import StandardScaler
from evaluation.stylometric_metrics import extract_features_batch, fit_feature_vocabulary, load_spacy_model
from rich.console import Console

console = Console()
RNG_SEED = 42
N_PERMUTATIONS = 1000
PROGRESS_EVERY = 10

# read the ICNALE processed dataset and remove countries with few samples (Mongolia and Sri Lanka):
project_root = Path(__file__).resolve().parent.parent.parent
train_df = pd.read_excel(project_root / "data/icnale/processed_icnale.xlsx")
train_df = train_df[~train_df["country"].isin(["Mongolia", "Sri Lanka"])]
console.log(train_df["country"].value_counts())

# organising essays and labels
essays = train_df["essays"].tolist()
labels = train_df["country"].tolist()
countries = sorted(set(labels))

# computing vectors
nlp = load_spacy_model(model_name="en_core_web_sm", require_gpu=True)
# vocab = fit_feature_vocabulary(essays, nlp_model=nlp)
vectors = extract_features_batch(essays, nlp_model=nlp, use_cfg=False)
console.log(vectors.shape)
console.log(vectors[0])

# standardising features
scaler = StandardScaler()
vectors = scaler.fit_transform(vectors)

# computing distances within and between countries
d_within_countries = []
d_between_countries = []
country_pairs = {(c1, c2): [] for c1 in countries for c2 in countries}
pair_i = []
pair_j = []
pair_d = []

for i in range(len(vectors)):
    for j in range(i + 1, len(vectors)):
        d = cosine(vectors[i], vectors[j])
        c1, c2 = labels[i], labels[j]
        country_pairs[(c1, c2)].append(d)
        country_pairs[(c2, c1)].append(d)
        pair_i.append(i)
        pair_j.append(j)
        pair_d.append(d)
        
        if c1 == c2:
            d_within_countries.append(d)
        else:
            d_between_countries.append(d)

# computing average distance matrix
matrix = pd.DataFrame(index=countries, columns=countries, dtype=float)
for c1 in countries:
    for c2 in countries:
        values = country_pairs[(c1, c2)]
        matrix.loc[c1, c2] = np.mean(values) if values else 0.0

d_within = np.array(d_within_countries).mean()
d_between = np.array(d_between_countries).mean()
delta_observed = d_between - d_within

# Permutation test on country labels: preserves the pairwise dependence structure
# while testing whether the observed between-vs-within separation is larger than chance.
rng = np.random.default_rng(RNG_SEED)
labels_array = np.array(labels)
pair_i = np.array(pair_i, dtype=np.int32)
pair_j = np.array(pair_j, dtype=np.int32)
pair_d = np.array(pair_d, dtype=np.float64)

perm_deltas = np.empty(N_PERMUTATIONS, dtype=np.float64)
perm_start = time.perf_counter()
for b in range(N_PERMUTATIONS):
    permuted_labels = labels_array[rng.permutation(len(labels_array))]
    within_mask = permuted_labels[pair_i] == permuted_labels[pair_j]
    between_mask = ~within_mask
    perm_within = pair_d[within_mask].mean()
    perm_between = pair_d[between_mask].mean()
    perm_deltas[b] = perm_between - perm_within

    if (b + 1) % PROGRESS_EVERY == 0 or (b + 1) == N_PERMUTATIONS:
        elapsed = time.perf_counter() - perm_start
        avg_per_perm = elapsed / (b + 1)
        remaining = avg_per_perm * (N_PERMUTATIONS - (b + 1))
        console.log(
            f"Permutation {b + 1}/{N_PERMUTATIONS} "
            f"(elapsed: {elapsed:.1f}s, eta: {remaining:.1f}s)"
        )

p_value_perm = (1.0 + np.sum(perm_deltas >= delta_observed)) / (N_PERMUTATIONS + 1.0)

console.log("Mean within-country distance:", d_within)
console.log("Mean between-country distance:", d_between)
console.log("Observed separation (between - within):", delta_observed)
console.log("Permutation test p-value (one-sided, between > within):", p_value_perm)

# saving the matrix and plot it as a heatmap.
matrix.to_excel(project_root / "data/icnale/country_distance_matrix.xlsx")

plt.figure(figsize=(14, 10))
plt.imshow(matrix.values)
plt.colorbar(label="Cosine distance")
plt.xticks(range(len(matrix.columns)), matrix.columns, rotation=60, ha="right", fontsize=12)
plt.yticks(range(len(matrix.index)), matrix.index, fontsize=12)
plt.title("Country Distance Matrix", fontsize=14)
plt.tight_layout()
plt.savefig(project_root / "data/icnale/country_distance_matrix.png", dpi=300)
plt.close()