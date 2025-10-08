from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Set
import math
import json
import re
from collections import defaultdict, Counter


def _normalize_text(s: str) -> str:
    """Lightweight normalization for string matching."""
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"[^\w\s]", "", s)  # remove punctuation
    return s


def _token_set(s: str) -> Set[str]:
    return set(_normalize_text(s).split())


def default_semantic_match(a: str, b: str, jaccard_threshold: float = 0.6) -> bool:
    """
    Cheap, deterministic proxy for semantic matching using token Jaccard.
    Returns True if the token Jaccard similarity >= threshold or one contains the other.
    """
    if not a or not b:
        return False
    na, nb = _normalize_text(a), _normalize_text(b)
    if na in nb or nb in na:
        return True
    A, B = _token_set(a), _token_set(b)
    if not A or not B:
        return False
    inter = len(A & B)
    union = len(A | B)
    j = inter / union
    return j >= jaccard_threshold


@dataclass
class ClarificationQuestion:
    """A single clarification question asked by the model in a given turn."""
    text: str
    # options shown to the user for this question (could be empty if free-form)
    options: List[str] = field(default_factory=list)
    # Optional label of which intent element this question targets (for dependency checks)
    element: Optional[str] = None


@dataclass
class ClarificationTurn:
    """One turn of clarification containing one or more questions in parallel (table style)."""
    questions: List[ClarificationQuestion]


@dataclass
class InteractionSample:
    """
    One full interaction for a single instruction.
    - predicted_vague: whether the system judged the instruction as vague (needs clarification)
    - gold_vague: ground truth for vagueness
    - turns: the clarification turns (K may be 0 if no clarification)
    - gold_underlying_questions: the canon set of underlying fact-level clarifications
      that SHOULD be asked for this intent (used for Intents Cover Rate)
    - dependency_graph: (optional) prerequisite mapping among elements for Logical Conflict Rate
        e.g., {"activities": {"destination","travel dates", "budget"}}
      Question.element should refer to a node in this graph.
    - element_resolution_order: (optional) order in which elements become resolved by user answers.
      If omitted, we infer order by first time the model asks about the element.
    """
    predicted_vague: bool
    gold_vague: bool
    turns: List[ClarificationTurn]

    gold_underlying_questions: List[str] = field(default_factory=list)

    dependency_graph: Dict[str, Set[str]] = field(default_factory=dict)
    element_resolution_order: Optional[List[str]] = None


# -------------------- Metric Implementations (Appendix B.3) --------------------

def vagueness_judgement_accuracy(samples: Sequence[InteractionSample]) -> float:
    """
    Accuracy of the system's vagueness judgement across all instructions.
    """
    if not samples:
        return 0.0
    correct = sum(1 for s in samples if bool(s.predicted_vague) == bool(s.gold_vague))
    return correct / len(samples)


def intents_cover_rate(
    samples: Sequence[InteractionSample],
    match_fn: Callable[[str, str], bool] = default_semantic_match,
) -> float:
    """
    Percentage of underlying fact clarification questions (gold) that are covered by the
    model's asked questions during the interaction.
    (Appendix B.3: Intents Cover Rate)
    """
    covered_ratios: List[float] = []
    for s in samples:
        if not s.gold_underlying_questions:
            continue
        asked = [q.text for t in s.turns for q in t.questions]
        covered = set()
        for gold_q in s.gold_underlying_questions:
            for asked_q in asked:
                if match_fn(gold_q, asked_q):
                    covered.add(gold_q)
                    break
        covered_ratios.append(len(covered) / len(s.gold_underlying_questions))
    return sum(covered_ratios) / len(covered_ratios) if covered_ratios else 0.0


def average_interaction_turns(samples: Sequence[InteractionSample]) -> float:
    """
    Average number of interaction turns per instruction.
    (Appendix B.3: Average Interaction Turns)
    """
    if not samples:
        return 0.0
    return sum(len(s.turns) for s in samples) / len(samples)


def average_questions_per_turn(samples: Sequence[InteractionSample]) -> float:
    """
    Average number of clarification questions per turn.
    (Appendix B.3: Average Questions Per Turn)
    """
    counts: List[int] = []
    for s in samples:
        for t in s.turns:
            counts.append(len(t.questions))
    return sum(counts) / len(counts) if counts else 0.0


def options_presenting_rate(samples: Sequence[InteractionSample]) -> float:
    """
    Percentage of clarification questions accompanied by any referential options.
    (Appendix B.3: Options Presenting Rate)
    """
    total_q = 0
    with_options = 0
    for s in samples:
        for t in s.turns:
            for q in t.questions:
                total_q += 1
                if q.options and any(_normalize_text(opt) for opt in q.options):
                    with_options += 1
    return with_options / total_q if total_q else 0.0


def options_reasonable_rate(
    samples: Sequence[InteractionSample],
    # Optional hook to judge whether an option is reasonable for a question.
    # If not provided, we use a heuristic: an option is reasonable if it shares
    # some token overlap with the question and is not empty.
    is_reasonable_option: Optional[Callable[[str, str], bool]] = None,
) -> float:
    """
    Percentage of referential options that are considered reasonable.
    (Appendix B.3: Options Reasonable Rate)
    """
    def _default_is_reasonable(question: str, option: str) -> bool:
        if not option.strip():
            return False
        # simple overlap heuristic
        A, B = _token_set(question), _token_set(option)
        return len(A & B) > 0 or len(B) <= 3  # allow short categorical labels

    judge = is_reasonable_option or _default_is_reasonable

    total_opts = 0
    reasonable = 0
    for s in samples:
        for t in s.turns:
            for q in t.questions:
                for opt in q.options:
                    total_opts += 1
                    if judge(q.text, opt):
                        reasonable += 1
    return reasonable / total_opts if total_opts else 0.0


def average_options_per_question(samples: Sequence[InteractionSample]) -> float:
    """
    Average number of options provided per clarification question.
    (Appendix B.3: Average Options Per Query)
    """
    counts: List[int] = []
    for s in samples:
        for t in s.turns:
            for q in t.questions:
                counts.append(len([o for o in q.options if _normalize_text(o)]))
    return sum(counts) / len(counts) if counts else 0.0


def logical_conflict_rate(samples: Sequence[InteractionSample]) -> float:
    """
    Proportion of clarification questions that violate prerequisite dependencies among
    intent elements, i.e., a question about element E is asked before ALL of its
    prerequisites have been resolved.
    (Appendix B.3: Logical Conflict Rate)
    """
    total_q = 0
    conflicts = 0

    for s in samples:
        # Determine the order in which elements are resolved
        if s.element_resolution_order is not None:
            resolved_order = [e for e in s.element_resolution_order if e]
        else:
            # infer by first time the element appears in a question
            first_seen = {}
            step = 0
            for t in s.turns:
                step += 1
                for q in t.questions:
                    if q.element and q.element not in first_seen:
                        first_seen[q.element] = step
            resolved_order = [e for e, _ in sorted(first_seen.items(), key=lambda kv: kv[1])]

        resolved_set: Set[str] = set()
        dep = {k: set(v) for k, v in s.dependency_graph.items()} if s.dependency_graph else {}

        # Walk through turns, check each question
        for t in s.turns:
            for q in t.questions:
                total_q += 1
                e = q.element
                if not e or e not in dep:
                    # if no dependency info, we consider it non-conflicting
                    continue
                prereqs = dep.get(e, set())
                if not prereqs.issubset(resolved_set):
                    conflicts += 1
            # After the turn, assume elements asked in this turn become resolved
            for q in t.questions:
                if q.element:
                    resolved_set.add(q.element)

    return conflicts / total_q if total_q else 0.0


# -------------------- Convenience: Compute All Metrics --------------------

def compute_all_metrics(
    samples: Sequence[InteractionSample],
    match_fn: Callable[[str, str], bool] = default_semantic_match,
    is_reasonable_option: Optional[Callable[[str, str], bool]] = None,
) -> Dict[str, float]:
    return {
        "vagueness_judgement_accuracy": vagueness_judgement_accuracy(samples),
        "intents_cover_rate": intents_cover_rate(samples, match_fn=match_fn),
        "average_interaction_turns": average_interaction_turns(samples),
        "average_questions_per_turn": average_questions_per_turn(samples),
        "options_presenting_rate": options_presenting_rate(samples),
        "options_reasonable_rate": options_reasonable_rate(samples, is_reasonable_option=is_reasonable_option),
        "average_options_per_question": average_options_per_question(samples),
        "logical_conflict_rate": logical_conflict_rate(samples),
    }


# -------------------- Example Usage --------------------

if __name__ == "__main__":
    from pprint import pprint

    EXAMPLE_DATA = [
        InteractionSample(
            predicted_vague=True,
            gold_vague=True,
            gold_underlying_questions=[
                "What is your destination?",
                "What are your travel dates?",
                "What is your budget range?",
            ],
            dependency_graph={
                "transportation": {"destination", "travel dates", "budget"},
                "activities": {"destination", "travel dates", "budget"},
                "accommodation": {"destination", "travel dates", "budget"},
            },
            turns=[
                ClarificationTurn(
                    questions=[
                        ClarificationQuestion(
                            text="What is your destination?",
                            options=["Paris", "Okinawa", "Hokkaido"],
                            element="destination",
                        ),
                        ClarificationQuestion(
                            text="What are your travel dates?",
                            options=["Dec 15-18", "Dec 19-22", "Other"],
                            element="travel dates",
                        ),
                        ClarificationQuestion(
                            text="What is your budget range?",
                            options=["$1K-$3K", "$3K-$8K", "$8K+"],
                            element="budget",
                        ),
                    ]
                ),
                ClarificationTurn(
                    questions=[
                        ClarificationQuestion(
                            text="Which activities interest you?",
                            options=["whale watching", "diving", "cultural relics"],
                            element="activities",
                        )
                    ]
                ),
            ],
        )
    ]

    pprint(compute_all_metrics(EXAMPLE_DATA))
