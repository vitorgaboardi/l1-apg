import logging
import re
import zss
import numpy as np
from nltk import word_tokenize, pos_tag, ngrams
from nltk import pos_tag, Tree
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer, util
from .utils import get_syntax_pattern, compute_tree_edit_distance
from typing import List
from itertools import combinations


_AVAILABLE_METRICS = {
    "TW":  lambda utterances, model: total_words(utterances),
    "UNIQ1": lambda utterances, model: unique_n_ratio(utterances, n=1),
    "UNIQ2": lambda utterances, model: unique_n_ratio(utterances, n=2),
    "UNIQ3": lambda utterances, model: unique_n_ratio(utterances, n=3),
    "SSM": lambda utterances, model: syntax_similarity_mean(utterances),
    "SB":  lambda utterances, model: self_bleu(utterances),
    "PSS": lambda utterances, model: pairwise_semantic_similarity(utterances, model),
    "TVN": lambda utterances, model: total_verbs_nouns(utterances),
    "USP": lambda utterances, model: unique_syntax_patterns(utterances),
}

DEFAULT_METRICS = ["TW", "UNIQ1", "UNIQ2", "UNIQ3", "SSM", "SB", "PSS"]

def compute_metrics(utterances, model, metrics: list[str] | None = None):
    """
    Compute diversity metrics for a list of utterances.

    Args:
        utterances: List of utterance strings
        model: SentenceTransformer model (required for PSS and CSS)
        metrics: List of metric keys to compute. Defaults to DEFAULT_METRICS.
                 Available: TW, TTR, UNIQ1, UNIQ2, UNIQ3, U3G, SSM, SB, PSS, UW, TVN, UVN, USP, CSS
    """
    if metrics is None:
        metrics = DEFAULT_METRICS

    unknown = [m for m in metrics if m not in _AVAILABLE_METRICS]
    if unknown:
        raise ValueError(f"Unknown metrics: {unknown}. Available: {list(_AVAILABLE_METRICS.keys())}")

    return {m: _AVAILABLE_METRICS[m](utterances, model) for m in metrics}


def unique_n_ratio(utterances: List[str], n: int = 1) -> float:
    """Unique-n: ratio of unique n-grams to total n-grams across utterances."""
    if n <= 0:
        raise ValueError("n must be >= 1")

    all_ngrams = []
    for utt in utterances:
        words = word_tokenize(utt.lower())
        if len(words) < n:
            continue
        all_ngrams.extend(list(ngrams(words, n)))

    total = len(all_ngrams)
    if total == 0:
        return 0.0

    unique_total = len(set(all_ngrams))
    return round(unique_total / total, 4)


def total_words(utterances: List[str]) -> int:
    """TW: Computes the total number of words in a list of utterances."""
    cleaned_utterances = [re.sub(r'[^\w\s]', '', utt) for utt in utterances]
    single_words = []
    for utt in cleaned_utterances:
        for word in utt.split():
            single_words.append(word.lower())

    total_word = len(single_words)
    return total_word


def total_verbs_nouns(utterances: List[str]) -> int:
    """TVN: Computes the total number of nouns and verbs in a list of utterances"""
    cleaned_utterances = [re.sub(r'[^\w\s]', '', utt) for utt in utterances]
    single_words = [word.lower() for sent in cleaned_utterances for word in word_tokenize(sent)]
    tagged_words = pos_tag(single_words)
    nouns_verbs = [word for word, tag in tagged_words if tag.startswith('N') or tag.startswith('V')]

    return len(nouns_verbs)


def self_bleu(utterances: List[str], n_gram = 4) -> float:
    """SB: Computes the average self-BLEU score for a list of utterances."""
    smoothing = SmoothingFunction().method1
    tokenized_utterances = [word_tokenize(utt.lower()) for utt in utterances]
    weights = tuple([1.0 / n_gram] * n_gram)

    bleu_scores = []
    for i in range(len(tokenized_utterances)):
        candidate = tokenized_utterances[i]
        references = tokenized_utterances[:i] + tokenized_utterances[i+1:]
        score = sentence_bleu(references, candidate, weights=weights, smoothing_function=smoothing)
        bleu_scores.append(score)

    return round(sum(bleu_scores) / len(bleu_scores), 4) if bleu_scores else 0.0


def unique_syntax_patterns(utterances: List[str]) -> int:
    """USP: Computes the number of unique syntax patterns in a set of utterances."""
    syntax_patterns = []
    for utt in utterances:
        pattern = get_syntax_pattern(utt, max_level=3)
        syntax_patterns.append(pattern)

    unique_syntax_patterns = set(syntax_patterns)
    return len(unique_syntax_patterns)


def syntax_similarity_mean(utterances: list[str]) -> float: 
    """SSM: Computes the syntax similarity mean defined by the paper 'Controllable Paraphrase Generation with a Syntactic Exemplar'.
    We compute the value pairwise between all sets of utterances and return the average."""
    syntax_patterns = []
    ted_scores = []
    
    for utt in utterances:
        pattern = get_syntax_pattern(utt, max_level=3)
        syntax_patterns.append(pattern)

    for tree1, tree2 in combinations(syntax_patterns, 2):
        ted = compute_tree_edit_distance(tree1, tree2)
        ted_scores.append(ted)

    if not ted_scores:
        return 0.0

    s_mean = np.mean(ted_scores)

    return round(float(s_mean), 4)


def pairwise_semantic_similarity(utterances: List[str], model: SentenceTransformer) -> float:
    """PSS: Computes the average pairwise cosine similarity between all utterances. 
    Returns 1 - the value, so the higher the value, more diverse it is."""
    if len(utterances) < 2:
        return 0.0

    # keep on GPU via tensor; normalize_embeddings=True makes dot product == cosine sim
    embeddings = model.encode(utterances, batch_size=64, convert_to_tensor=True, normalize_embeddings=True)
    sim_matrix = util.cos_sim(embeddings, embeddings)  # (N, N)
    N = sim_matrix.shape[0]
    upper_sum = (sim_matrix.triu(diagonal=1)).sum().item()
    n_pairs = N * (N - 1) / 2
    mean_sim = upper_sum / n_pairs

    return round(float(1 - mean_sim), 4)

