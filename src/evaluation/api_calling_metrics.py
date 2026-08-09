"""
Definitions of metrics for evaluating API-calling agents.
    - Execution Success Rate (ESR), defined as the proportion of predicted API calls that execute successfully (e.g., return an HTTP 200 response);
    - API Selection Precision (ASP), defined as the proportion of correctly predicted API endpoints out of all predicted endpoints;
    - Semantic Exact Match (SEM), defined as the proportion of cases where predicted API calls match ground-truth using semantic similarity (threshold 0.98) for parameter values;
"""

from __future__ import annotations
import ast
import json
from typing import Any
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize sentence transformer model (lightweight and fast)
_EMBEDDING_MODEL = None
SEMANTIC_THRESHOLD = 0.95

def _get_embedding_model() -> SentenceTransformer:
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        _EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    return _EMBEDDING_MODEL


def compute_instance_metrics(
    predicted_calls: Any,
    execution_results: Any,
    ground_truth_calls: Any,
) -> dict[str, float]:
    """Convenience helper that returns all per-instance metrics."""
    return {
        "esr": execution_success_rate(predicted_calls, execution_results),
        "asp": api_selection_precision(predicted_calls, ground_truth_calls),
        "sem": semantic_exact_match(predicted_calls, ground_truth_calls),
    }


def execution_success_rate(predicted_calls: Any, execution_results: Any) -> float:
    """
    Per-instance Execution Success Rate (ESR).

    ESR = (# predicted calls with successful execution) / (# predicted calls)
    Success is defined as status_code == 200.
    """
    calls = _to_list_of_calls(predicted_calls)
    results = execution_results if isinstance(execution_results, list) else []

    total_predicted = len(calls)
    if total_predicted == 0:
        return 0.0

    successes = 0
    for idx in range(total_predicted):
        if idx >= len(results):
            continue
        item = results[idx]
        if isinstance(item, dict) and item.get("status_code") == 200:
            successes += 1

    return successes / total_predicted


def api_selection_precision(predicted_calls: Any, ground_truth_calls: Any) -> float:
    """
    Per-instance API Selection Precision (ASP).

    Proportional metric:
    ASP = (# correctly predicted endpoints) / (# predicted endpoints)
    
    An endpoint is correct if it matches any ground-truth endpoint.
    Returns 0.0 if no predictions are made.
    """
    predicted = _to_list_of_calls(predicted_calls)
    ground_truth = _to_list_of_calls(ground_truth_calls)

    if not predicted:
        return 0.0
    
    if not ground_truth:
        return 0.0

    gt_names = {_extract_name_and_args(gt)[0] for gt in ground_truth}
    
    correct_predictions = 0
    for pred in predicted:
        pred_name = _extract_name_and_args(pred)[0]
        if pred_name in gt_names:
            correct_predictions += 1

    return correct_predictions / len(predicted)


def semantic_exact_match(predicted_calls: Any, ground_truth_calls: Any) -> float:
    """
    Per-instance Semantic Exact Match (SEM).

    Returns 1.0 if any predicted API call semantically matches a ground-truth call:
    - Same endpoint name (exact match)
    - ALL parameter values have cosine similarity >= SEMANTIC_THRESHOLD (0.98)
    
    This is stricter than argument_accuracy (requires all params to match),
    but more flexible than exact_match (uses semantic similarity for values).
    """
    predicted = _to_list_of_calls(predicted_calls)
    ground_truth = _to_list_of_calls(ground_truth_calls)

    if not predicted or not ground_truth:
        return 0.0

    # Build GT calls by endpoint name
    gt_by_name: dict[str, list[dict[str, Any]]] = {}
    for gt_call in ground_truth:
        name, args = _extract_name_and_args(gt_call)
        if name not in gt_by_name:
            gt_by_name[name] = []
        gt_by_name[name].append(args)

    # Check if any predicted call has a perfect semantic match
    for pred_call in predicted:
        pred_name, pred_args = _extract_name_and_args(pred_call)
        
        if pred_name not in gt_by_name:
            continue
        
        # Check against all GT calls with same endpoint
        for gt_args in gt_by_name[pred_name]:
            if _is_semantic_exact_match(pred_args, gt_args):
                return 1.0
    
    return 0.0
    

def _to_list_of_calls(raw_calls: Any) -> list[dict[str, Any]]:
    """
    Parse API calls into a normalized list of dictionaries.

    Supported input formats:
    - list[dict]
    - dict (single call)
    - JSON string
    - Python-literal string (e.g. from CSV with single quotes)
    - None/empty -> []
    """
    if raw_calls is None:
        return []

    parsed = raw_calls
    if isinstance(raw_calls, str):
        text = raw_calls.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except Exception:
            try:
                parsed = ast.literal_eval(text)
            except Exception:
                return []

    if isinstance(parsed, dict):
        parsed = [parsed]

    if not isinstance(parsed, list):
        return []

    calls: list[dict[str, Any]] = []
    for item in parsed:
        if isinstance(item, dict):
            calls.append(item)
    return calls


def _normalize_endpoint_name(name: Any) -> str:
    if name is None:
        return ""
    return str(name).strip().lower()


def _normalize_json_like(value: Any) -> Any:
    """Recursively normalize dict/list/scalar values for stable equality checks."""
    if isinstance(value, dict):
        return {k: _normalize_json_like(value[k]) for k in sorted(value)}
    if isinstance(value, list):
        return [_normalize_json_like(v) for v in value]
    if isinstance(value, float):
        return round(value, 10)
    return value


def _extract_name_and_args(call: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    name = _normalize_endpoint_name(call.get("name") or call.get("api_method_name"))
    args = call.get("arguments", {})
    if not isinstance(args, dict):
        args = {}
    return name, _normalize_json_like(args)


def _compute_semantic_similarity(val1: Any, val2: Any) -> float:
    """
    Compute semantic similarity between two values using embeddings.
    
    Returns:
        - 1.0 if exact match (after normalization)
        - Cosine similarity [0.0, 1.0] if both are non-empty strings
        - 0.0 if types differ or comparison not possible
    """
    # Normalize first
    norm_val1 = _normalize_json_like(val1)
    norm_val2 = _normalize_json_like(val2)
    
    # Check exact equality first (fast path)
    if norm_val1 == norm_val2:
        return 1.0
    
    # Convert to strings for embedding
    str1 = _value_to_string(norm_val1)
    str2 = _value_to_string(norm_val2)
    
    # If either is empty, no semantic match possible
    if not str1 or not str2:
        return 0.0
    
    # Compute embeddings and cosine similarity
    model = _get_embedding_model()
    embeddings = model.encode([str1, str2])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    
    return float(similarity)


def _is_semantic_exact_match(pred_args: dict[str, Any], gt_args: dict[str, Any]) -> bool:
    """
    Check if ALL parameters match semantically with similarity >= SEMANTIC_THRESHOLD.
    
    Requires:
    - Exact same set of keys
    - All values have semantic similarity >= SEMANTIC_THRESHOLD
    """
    # Must have exact same keys
    if set(pred_args.keys()) != set(gt_args.keys()):
        return False
    
    # If both empty, it's a match
    if not pred_args and not gt_args:
        return True
    
    # All values must meet threshold
    for key in pred_args.keys():
        pred_val = pred_args[key]
        gt_val = gt_args[key]
        
        similarity = _compute_semantic_similarity(pred_val, gt_val)
        if similarity < SEMANTIC_THRESHOLD:
            return False
    
    return True


def _value_to_string(value: Any) -> str:
    """
    Convert a value to a string representation suitable for embedding.
    """
    if isinstance(value, str):
        return value.strip()
    elif isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    elif value is None:
        return ""
    else:
        return str(value)
