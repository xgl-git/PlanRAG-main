
import re
import string
from typing import List, Iterable, Dict, Optional
from collections import Counter


def normalize_answer(s: Optional[str]) -> str:
    if s is None:
        s = ""
    s = s.strip()
    s = re.sub(f"[{re.escape(string.punctuation)}]", " ", s)
    s = re.sub(r"[，。！？；：、“”‘’]", " ", s)
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', " ", s, flags=re.IGNORECASE)
    s = " ".join(s.split())
    return s


def get_tokens(s: str) -> List[str]:
    if re.search(r'[\u4e00-\u9fff]', s):
        return list(s.replace(" ", ""))
    return s.split()


def em_score(prediction: Optional[str], ground_truths: Iterable[str]) -> int:
    norm_pred = normalize_answer(prediction)
    for gt in ground_truths:
        if norm_pred == normalize_answer(gt):
            return 1
    return 0


accuracy_score = em_score


def f1_score(prediction: Optional[str], ground_truths: Iterable[str]) -> float:
    norm_pred = normalize_answer(prediction)
    pred_tokens = get_tokens(norm_pred)
    best_f1 = 0.0
    for gt in ground_truths:
        truth_tokens = get_tokens(normalize_answer(gt))
        if not pred_tokens and not truth_tokens:
            f1 = 1.0
        elif not pred_tokens or not truth_tokens:
            f1 = 0.0
        else:
            pred_counter = Counter(pred_tokens)
            truth_counter = Counter(truth_tokens)
            common = pred_counter & truth_counter
            num_same = sum(common.values())
            if num_same == 0:
                f1 = 0.0
            else:
                precision = num_same / len(pred_tokens)
                recall = num_same / len(truth_tokens)
                f1 = 2 * precision * recall / (precision + recall)
        best_f1 = max(best_f1, f1)
    return best_f1


def hits_at_k(predictions: List[List[str]], ground_truths: List[Iterable[str]], k: int = 1) -> float:
    assert len(predictions) == len(ground_truths)
    n = len(predictions)
    if n == 0:
        return 0.0
    hits = 0
    for pred_list, gt in zip(predictions, ground_truths):
        topk = pred_list[:k]
        gt_norm = [normalize_answer(g) for g in gt]
        if any(normalize_answer(p) in gt_norm for p in topk):
            hits += 1
    return 100.0 * hits / n

def evaluate_batch(predictions: List[Optional[str]],
                   list_of_ground_truths: List[Iterable[str]],
                   predictions_topk: List[List[str]] = None,
                   topk_list: List[int] = [1, 3]) -> Dict[str, float]:
    assert len(predictions) == len(list_of_ground_truths)
    n = len(predictions)
    if n == 0:
        return {"accuracy": 0.0, "em": 0.0, "f1": 0.0, **{f"hits@{k}": 0.0 for k in topk_list}}

    acc_total, em_total, f1_total = 0, 0, 0.0
    for p, gts in zip(predictions, list_of_ground_truths):
        p_safe = "" if p is None else p
        acc_total += accuracy_score(p_safe, gts)
        em_total += em_score(p_safe, gts)
        f1_total += f1_score(p_safe, gts)

    metrics = {
        "accuracy": 100.0 * acc_total / n,
        "em": 100.0 * em_total / n,
        "f1": 100.0 * f1_total / n
    }
    if predictions_topk is not None:
        for k in topk_list:
            metrics[f"hits@{k}"] = hits_at_k(predictions_topk, list_of_ground_truths, k=k)
    return metrics
