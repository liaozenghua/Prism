# cognitive_load_metrics.py
# Metrics for Cognitive Load experiment.
# Behavioral: duration (s), total token count
# Subjective: overall rating (1-10), per-2-turn ratings trajectory
# Physiological (optional EEG): mean & std of all-band PSD with 10s window, 1s step
#
# References to your paper:
# - Measures & protocol (time, token count, overall & multi-turn ratings): Fig. 3 and Sec. 5.3.1.  [WWW_2026.pdf]
# - EEG preprocessing (0.1–40 Hz band-limit, 48–52 Hz notch, 10s window/1s step): Appendix D.1. [WWW_2026.pdf]

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import argparse
import json
import math
import csv
import numpy as np

# --------------------
# Data structures
# --------------------

@dataclass
class Turn:
    idx: int
    # ISO8601 or unix seconds. If per-turn timestamps are absent, use session-level start/end.
    timestamp: Optional[float] = None  # seconds since epoch (preferred) or relative
    role: Optional[str] = None         # "user" or "assistant"
    tokens: Optional[int] = None       # tokens in this turn (prompt+completion or per-speaker)
    rating: Optional[float] = None     # subjective rating provided at this turn (1-10)

@dataclass
class InteractionSession:
    session_id: str
    turns: List[Turn] = field(default_factory=list)
    start_time: Optional[float] = None  # seconds since epoch
    end_time: Optional[float] = None
    overall_rating: Optional[float] = None  # final 1-10
    # If ratings are every N turns (e.g., 2), set here; default tries to infer.
    rating_every_n_turns: Optional[int] = None

# --------------------
# Behavioral metrics
# --------------------

def compute_duration_seconds(sess: InteractionSession) -> float:
    # Prefer explicit start/end; else infer from first/last turn timestamps.
    if sess.start_time is not None and sess.end_time is not None:
        return max(0.0, float(sess.end_time - sess.start_time))
    ts = [t.timestamp for t in sess.turns if t.timestamp is not None]
    if len(ts) >= 2:
        return max(0.0, float(max(ts) - min(ts)))
    return 0.0

def compute_token_count(sess: InteractionSession) -> int:
    # Sum over turns; missing tokens treated as 0
    return int(sum(int(t.tokens or 0) for t in sess.turns))

# --------------------
# Subjective metrics
# --------------------

def infer_rating_stride(sess: InteractionSession) -> int:
    # Try to infer every-k-turn rating cadence from indexes containing ratings.
    idxs = [t.idx for t in sess.turns if t.rating is not None]
    if len(idxs) >= 2:
        diffs = [b - a for a, b in zip(idxs[:-1], idxs[1:]) if b > a]
        if diffs:
            # return the most common difference
            vals, counts = np.unique(diffs, return_counts=True)
            return int(vals[np.argmax(counts)])
    return int(sess.rating_every_n_turns or 2)  # default: every 2 turns

def collect_multiturn_ratings(sess: InteractionSession) -> List[Tuple[int, float]]:
    stride = infer_rating_stride(sess)
    pairs = []
    for t in sess.turns:
        if t.rating is not None:
            pairs.append((t.idx, float(t.rating)))
    # Optionally resample to exact cadence if needed; here we just return provided points.
    return pairs

def summarize_multiturn_ratings(ratings: List[Tuple[int, float]]) -> Dict[str, float]:
    if not ratings:
        return {"count": 0, "mean": 0.0, "std": 0.0, "first": 0.0, "last": 0.0, "slope_per_turn": 0.0}
    xs = np.array([i for i, _ in ratings], dtype=float)
    ys = np.array([v for _, v in ratings], dtype=float)
    # Linear trend as slope (rating change per turn)
    A = np.vstack([xs, np.ones_like(xs)]).T
    slope, _ = np.linalg.lstsq(A, ys, rcond=None)[0]
    return {
        "count": float(len(ys)),
        "mean": float(np.mean(ys)),
        "std": float(np.std(ys, ddof=0)),
        "first": float(ys[0]),
        "last": float(ys[-1]),
        "slope_per_turn": float(slope),
    }

# --------------------
# EEG PSD (physiological)
# --------------------

def _try_scipy():
    try:
        import scipy.signal as sps
        return sps
    except Exception:
        return None

def bandlimit_fft(x: np.ndarray, fs: float, f_lo: float = 0.1, f_hi: float = 40.0, notch_50: bool = True) -> np.ndarray:
    """
    Lightweight frequency-domain band-limit: FFT -> zero bins outside [f_lo, f_hi]
    and optionally notch 50 Hz (48-52 Hz). Not as good as IIR/FIR but works without SciPy.
    """
    n = len(x)
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0/fs)
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    if notch_50:
        mask &= ~((freqs >= 48.0) & (freqs <= 52.0))
    X_filtered = X * mask
    x_rec = np.fft.irfft(X_filtered, n=n)
    return x_rec

def welch_psd(signal: np.ndarray, fs: float, win_sec: float = 10.0, step_sec: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sliding-window Welch PSD over the entire frequency range.
    Returns (times, freqs, PSD_matrix) where PSD_matrix shape = (num_windows, num_freqs)
    """
    sps = _try_scipy()
    nperseg = int(win_sec * fs)
    step = int(step_sec * fs)
    if nperseg <= 0 or step <= 0 or len(signal) < nperseg:
        # Single-shot Welch
        if sps is not None:
            freqs, Pxx = sps.welch(signal, fs=fs, nperseg=max(16, int(2**np.floor(np.log2(len(signal))))), noverlap=0)
        else:
            # naive periodogram
            freqs = np.fft.rfftfreq(len(signal), d=1.0/fs)
            X = np.fft.rfft(signal * np.hanning(len(signal)))
            Pxx = (1.0 / (fs * (np.abs(np.hanning(len(signal)))**2).sum())) * (np.abs(X) ** 2)
        return np.array([0.0]), freqs, Pxx[None, :]

    times = []
    psd_list = []
    for start in range(0, len(signal) - nperseg + 1, step):
        seg = signal[start:start + nperseg]
        if sps is not None:
            freqs, Pxx = sps.welch(seg, fs=fs, nperseg=nperseg, noverlap=0)
        else:
            freqs = np.fft.rfftfreq(nperseg, d=1.0/fs)
            X = np.fft.rfft(seg * np.hanning(nperseg))
            Pxx = (1.0 / (fs * (np.abs(np.hanning(nperseg))**2).sum())) * (np.abs(X) ** 2)
        psd_list.append(Pxx)
        times.append((start + nperseg / 2) / fs)
    return np.array(times), freqs, np.vstack(psd_list)

def eeg_allband_psd_stats(
    eeg_csv_path: Path,
    fs: float,
    win_sec: float = 10.0,
    step_sec: float = 1.0,
    apply_bandlimit: bool = True,
    channel_cols: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Load EEG CSV (columns: timestamp, ch1, ch2, ...). Average channels across brain,
    (optionally) band-limit to 0.1–40 Hz with 48–52 Hz notch, then compute sliding-window Welch PSD.
    Report mean and std of all-band PSD over time, matching Fig.4 description.
    """
    # Load CSV
    with eeg_csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return {"psd_mean": 0.0, "psd_std": 0.0}

    # Pick channels
    if channel_cols is None:
        # Heuristic: all numeric fields except 'timestamp'
        sample = rows[0]
        channel_cols = [k for k,v in sample.items() if k.lower() != "timestamp"]
    # Build matrix [T x C]
    mat = []
    for r in rows:
        vals = []
        for c in channel_cols:
            try:
                vals.append(float(r[c]))
            except Exception:
                vals.append(0.0)
        mat.append(vals)
    X = np.asarray(mat, dtype=float)  # [T, C]

    # Average channels (common average reference proxy)
    sig = np.mean(X, axis=1)

    # Band-limit (optional, Appendix D.1)
    if apply_bandlimit:
        sig = bandlimit_fft(sig, fs=fs, f_lo=0.1, f_hi=40.0, notch_50=True)

    times, freqs, PSD = welch_psd(sig, fs=fs, win_sec=win_sec, step_sec=step_sec)
    # All-band PSD per window = mean across frequency bins
    psd_allband = PSD.mean(axis=1)  # [num_windows]
    return {"psd_mean": float(np.mean(psd_allband)), "psd_std": float(np.std(psd_allband, ddof=0))}

# --------------------
# JSONL I/O and top-level computation
# --------------------

def _session_from_obj(o: Dict[str, Any]) -> InteractionSession:
    turns = []
    for t in o.get("turns", []):
        turns.append(Turn(
            idx=int(t.get("idx", 0)),
            timestamp=(float(t["timestamp"]) if t.get("timestamp") is not None else None),
            role=t.get("role"),
            tokens=(int(t["tokens"]) if t.get("tokens") is not None else None),
            rating=(float(t["rating"]) if t.get("rating") is not None else None),
        ))
    return InteractionSession(
        session_id=str(o.get("session_id", "")),
        turns=sorted(turns, key=lambda x: x.idx),
        start_time=(float(o["start_time"]) if o.get("start_time") is not None else None),
        end_time=(float(o["end_time"]) if o.get("end_time") is not None else None),
        overall_rating=(float(o["overall_rating"]) if o.get("overall_rating") is not None else None),
        rating_every_n_turns=(int(o["rating_every_n_turns"]) if o.get("rating_every_n_turns") is not None else None),
    )

def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out

def compute_cognitive_load_metrics(
    sessions: List[InteractionSession],
    eeg_csv_path: Optional[Path] = None,
    eeg_fs: Optional[float] = None,
) -> Dict[str, Any]:
    # Aggregate behavioral & subjective per-session, then macro-average
    durations = [compute_duration_seconds(s) for s in sessions]
    tokens = [compute_token_count(s) for s in sessions]
    overall = [s.overall_rating for s in sessions if s.overall_rating is not None]
    # Multi-turn ratings: summarize each session then average summaries
    mt_summaries = [summarize_multiturn_ratings(collect_multiturn_ratings(s)) for s in sessions]
    def _avg(key: str, coll: List[Dict[str, float]]) -> float:
        vals = [d[key] for d in coll if d.get("count", 0) > 0]
        return float(np.mean(vals)) if vals else 0.0

    result = {
        "behavioral": {
            "duration_sec_mean": float(np.mean(durations)) if durations else 0.0,
            "duration_sec_std": float(np.std(durations, ddof=0)) if durations else 0.0,
            "token_count_mean": float(np.mean(tokens)) if tokens else 0.0,
            "token_count_std": float(np.std(tokens, ddof=0)) if tokens else 0.0,
        },
        "subjective": {
            "overall_rating_mean": float(np.mean(overall)) if overall else 0.0,
            "overall_rating_std": float(np.std(overall, ddof=0)) if overall else 0.0,
            # Multi-turn trajectory summary (macro-averaged across sessions)
            "multiturn": {
                "mean": _avg("mean", mt_summaries),
                "std": _avg("std", mt_summaries),
                "first": _avg("first", mt_summaries),
                "last": _avg("last", mt_summaries),
                "slope_per_turn": _avg("slope_per_turn", mt_summaries),
                "count_mean": _avg("count", mt_summaries),
            }
        }
    }

    if eeg_csv_path is not None and eeg_fs is not None:
        eeg_stats = eeg_allband_psd_stats(Path(eeg_csv_path), fs=float(eeg_fs))
        result["physiological"] = eeg_stats

    return result

# --------------------
# CLI
# --------------------

def main():
    p = argparse.ArgumentParser(description="Compute Cognitive Load experiment metrics.")
    p.add_argument("--sessions_jsonl", required=True, help="Path to JSONL with interaction sessions.")
    p.add_argument("--eeg_csv", help="Optional EEG CSV (timestamp, ch1, ch2, ...).")
    p.add_argument("--eeg_fs", type=float, help="Sampling rate (Hz) for EEG CSV.")
    p.add_argument("--output", help="Write metrics JSON here; print if omitted.")
    args = p.parse_args()

    sessions = [_session_from_obj(o) for o in _read_jsonl(Path(args.sessions_jsonl))]
    out = compute_cognitive_load_metrics(
        sessions,
        eeg_csv_path=Path(args.eeg_csv) if args.eeg_csv else None,
        eeg_fs=args.eeg_fs,
    )

    js = json.dumps(out, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(js, encoding="utf-8")
        print(f"Wrote metrics to {args.output}")
    else:
        print(js)

if __name__ == "__main__":
    main()
