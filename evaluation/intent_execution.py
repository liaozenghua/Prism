# intent_execution_metrics.py
# Metrics for the "Intent Execution" experiment:
# - BLEU
# - Intent Execution Faithfulness (Faithful)
# - Unnecessary Sub-tasks rate (US)
# - General Sub-tasks rate (GS)
# - Tool Invocations per Sub-task (TI)
#
# Metric definitions follow Appendix C.1 of your paper:
# BLEU; Faithful = proportion of final outputs that fully satisfy clarified intent elements;
# US = percent of sub-tasks regarded as unnecessary; GS = percent too-general sub-tasks;
# TI = average tool invocations per sub-task. :contentReference[oaicite:1]{index=1}

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple
import math
import re
import json
from pathlib import Path
import argparse


# ----------------------------
# Text normalization utilities
# ----------------------------

def _normalize_text(s: str) -> str:
    s = s or ""
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def _tokenize(s: str) -> List[str]:
    s = re.sub(r"[^a-z0-9]+", " ", _normalize_text(s))
    return [t for t in s.split() if t]


# ----------------------------
# Simple BLEU implementation
# ----------------------------

def _ngram_counts(tokens: List[str], n: int) -> Dict[Tuple[str, ...], int]:
    return {tuple(tokens[i:i+n]): tokens[i:i+n].count(tokens[i]) for i in range(len(tokens)-n+1)} if len(tokens)>=n else {}

def _prec_clip(candidate: List[str], reference: List[str], n: int) -> Tuple[int, int]:
    """Clipped n-gram precision numerator/denominator."""
    from collections import Counter
    c_ngrams = Counter(tuple(candidate[i:i+n]) for i in range(len(candidate)-n+1))
    r_ngrams = Counter(tuple(reference[i:i+n]) for i in range(len(reference)-n+1))
    if not c_ngrams:
        return 0, 0
    clip = 0
    for g, c in c_ngrams.items():
        clip += min(c, r_ngrams.get(g, 0))
    return clip, sum(c_ngrams.values())

def sentence_bleu(candidate: str, reference: str, max_n: int = 4, smooth: bool = True) -> float:
    """
    BLEU-1..N with brevity penalty; single reference (adequate for our experiment reporting). :contentReference[oaicite:2]{index=2}
    """
    c_tok, r_tok = _tokenize(candidate), _tokenize(reference)
    if not c_tok:
        return 0.0

    # Modified n-gram precisions
    p_logs = []
    for n in range(1, max_n + 1):
        num, den = _prec_clip(c_tok, r_tok, n)
        if den == 0:
            p = 0.0
        else:
            if num == 0 and smooth:
                # Chen-Cherry Smoothing 1: add 1 to num & den
                p = (num + 1) / (den + 1)
            else:
                p = num / den
        if p == 0.0:
            return 0.0
        p_logs.append(math.log(p))

    geo_mean = math.exp(sum(p_logs) / max_n)

    # Brevity Penalty
    c_len, r_len = len(c_tok), len(r_tok)
    if c_len > r_len:
        bp = 1.0
    else:
        bp = math.exp(1 - r_len / max(c_len, 1))

    return bp * geo_mean


# ----------------------------
# Data model
# ----------------------------

@dataclass
class SubTask:
    """One agent sub-task."""
    name: str
    tool_invocations: int = 0
    # tag ∈ {"ok","unnecessary","general"}; extend as needed
    tag: str = "ok"

@dataclass
class ExecutionCase:
    """
    One evaluation case for intent execution.
    - reference: reference output text (for BLEU)
    - prediction: model final output text
    - subtasks: the sub-task plan executed by the agent (with tool calls)
    - faithful: optional labeled boolean; if None, we fall back to heuristic_judge
    - required_elements: (optional) the clarified intent elements that must be satisfied
    """
    reference: str
    prediction: str
    subtasks: List[SubTask] = field(default_factory=list)
    faithful: Optional[bool] = None
    required_elements: List[str] = field(default_factory=list)


# ----------------------------
# Metrics
# ----------------------------

def bleu_macro(cases: Sequence[ExecutionCase], max_n: int = 4) -> float:
    """Macro-average BLEU over cases. :contentReference[oaicite:3]{index=3}"""
    if not cases:
        return 0.0
    scores = [sentence_bleu(c.prediction, c.reference, max_n=max_n) for c in cases]
    return sum(scores) / len(scores)

def intent_execution_faithfulness(
    cases: Sequence[ExecutionCase],
    judge_fn: Optional[Callable[[ExecutionCase], bool]] = None,
) -> float:
    """
    Proportion of final outputs judged to fully satisfy the clarified intent elements. :contentReference[oaicite:4]{index=4}
    If ExecutionCase.faithful is provided, use it directly; otherwise call judge_fn.
    Default judge_fn = heuristic over required_elements containment.
    """
    if not cases:
        return 0.0

    def _default_judge(case: ExecutionCase) -> bool:
        if not case.required_elements:
            # If no elements are provided, fallback to a conservative False
            return False
        pred = _normalize_text(case.prediction)
        # A simple deterministic proxy: each required element string must be reflected
        # (substring / token overlap) in the prediction.
        for elem in case.required_elements:
            e = _normalize_text(elem)
            if not e:
                continue
            if e not in pred:
                # allow token-overlap fallback
                etoks, ptoks = set(_tokenize(e)), set(_tokenize(pred))
                if not etoks or len(etoks & ptoks) / max(len(etoks), 1) < 0.6:
                    return False
        return True

    judge = judge_fn or _default_judge
    positives = 0
    for c in cases:
        if c.faithful is not None:
            positives += 1 if c.faithful else 0
        else:
            positives += 1 if judge(c) else 0
    return positives / len(cases)

def unnecessary_subtasks_rate(cases: Sequence[ExecutionCase]) -> float:
    """Percent of sub-tasks regarded as unnecessary. :contentReference[oaicite:5]{index=5}"""
    total = sum(len(c.subtasks) for c in cases)
    if total == 0:
        return 0.0
    unnec = sum(1 for c in cases for s in c.subtasks if (s.tag or "").lower() == "unnecessary")
    return unnec / total

def general_subtasks_rate(cases: Sequence[ExecutionCase]) -> float:
    """Percent of sub-tasks that are too general (not user-specific). :contentReference[oaicite:6]{index=6}"""
    total = sum(len(c.subtasks) for c in cases)
    if total == 0:
        return 0.0
    gen = sum(1 for c in cases for s in c.subtasks if (s.tag or "").lower() == "general")
    return gen / total

def tool_invocations_per_subtask(cases: Sequence[ExecutionCase]) -> float:
    """Average tool invocations per sub-task. Reflects execution efficiency. :contentReference[oaicite:7]{index=7}"""
    subtasks = [s for c in cases for s in c.subtasks]
    if not subtasks:
        return 0.0
    return sum(max(0, int(s.tool_invocations)) for s in subtasks) / len(subtasks)

def compute_all_execution_metrics(
    cases: Sequence[ExecutionCase],
    judge_fn: Optional[Callable[[ExecutionCase], bool]] = None,
    bleu_max_n: int = 4,
) -> Dict[str, float]:
    return {
        "BLEU": bleu_macro(cases, max_n=bleu_max_n),
        "Faithful": intent_execution_faithfulness(cases, judge_fn=judge_fn),
        "US": unnecessary_subtasks_rate(cases),
        "GS": general_subtasks_rate(cases),
        "TI": tool_invocations_per_subtask(cases),
    }


# ----------------------------
# JSONL I/O and CLI
# ----------------------------

def _case_from_obj(o: dict) -> ExecutionCase:
    subtasks = [SubTask(
        name=s.get("name",""),
        tool_invocations=int(s.get("tool_invocations", 0)),
        tag=(s.get("tag") or "ok")
    ) for s in (o.get("subtasks") or [])]

    return ExecutionCase(
        reference=o.get("reference",""),
        prediction=o.get("prediction",""),
        subtasks=subtasks,
        faithful=o.get("faithful", None),
        required_elements=list(o.get("required_elements") or []),
    )

def _read_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def _write_json(path: Path, obj: dict):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def main():
    parser = argparse.ArgumentParser(description="Compute Intent Execution metrics.")
    parser.add_argument("--input", required=True, help="Path to JSONL file of execution cases.")
    parser.add_argument("--output", help="Optional path to write metrics JSON; print to stdout if omitted.")
    parser.add_argument("--bleu_max_n", type=int, default=4, help="Max n-gram for BLEU (default 4).")
    args = parser.parse_args()

    cases = [_case_from_obj(o) for o in _read_jsonl(Path(args.input))]
    metrics = compute_all_execution_metrics(cases, bleu_max_n=args.bleu_max_n)

    if args.output:
        _write_json(Path(args.output), metrics)
        print(f"Wrote metrics to {args.output}")
    else:
        print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
