"""Genetic Algorithm optimizer for risk-score weights and flag thresholds.

Optimizes the 6 weights in risk_scorer.WEIGHTS and the 6 flag thresholds
in risk_scorer._FLAG_RULES to maximise insider-detection quality using
existing scored CSVs -- no model retraining required.

Chromosome layout (12 genes):
  genes[0:6]   raw weights for [lstm_p95, after_hours, bcc_usage,
                                 file_exfil, usb_activity, multi_pc]
               normalised to sum=1 during fitness evaluation
  genes[6:12]  flag thresholds (one per signal, clipped to [0.10, 0.95])

Fitness = 0.8 * F1@50 + 0.2 * flag_coverage@50
  - F1@50          : standard F1 for top-50 risk-ranked users vs ground truth
  - flag_coverage@50: fraction of top-50 users with >= 1 behavioural flag

Outputs
-------
models/ga_optimized_config.json   weights + thresholds + run metadata
models/ga_optimization_report.json generation-by-generation history

Usage
-----
    python colab/ga_optimizer.py            # run with defaults
    python colab/ga_optimizer.py --quick    # 20 generations (smoke test)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path bootstrap -- works locally and on Colab
# ---------------------------------------------------------------------------
REPO_DIR = Path(os.environ.get("DLP_REPO", str(Path(__file__).resolve().parent.parent)))
sys.path.insert(0, str(REPO_DIR))
from config import CLEANED_DIR, MODELS_DIR  # noqa: E402

INSIDERS_CSV = REPO_DIR / "archive" / "answers" / "answers" / "insiders.csv"
LSTM_CSV     = CLEANED_DIR / "email_user_daily_lstm_scored.csv"
IFOREST_CSV  = CLEANED_DIR / "email_user_daily_scored.csv"

GA_CONFIG_PATH = MODELS_DIR / "ga_optimized_config.json"
GA_REPORT_PATH = MODELS_DIR / "ga_optimization_report.json"

# ---------------------------------------------------------------------------
# Signal and weight metadata (order must match chromosome gene order)
# ---------------------------------------------------------------------------
SIGNAL_NAMES = [
    "lstm_p95",
    "after_hours",
    "bcc_usage",
    "file_exfil",
    "usb_activity",
    "multi_pc",
]
SIGNAL_NORM_COLS = [
    "lstm_p95_norm",
    "after_hours_norm",
    "bcc_usage_norm",
    "file_exfil_norm",
    "usb_activity_norm",
    "multi_pc_norm",
]
FLAG_SIGNAL_COLS = SIGNAL_NORM_COLS  # thresholds apply in the same order

BASELINE_WEIGHTS = [0.50, 0.15, 0.10, 0.10, 0.10, 0.05]
BASELINE_THRESHOLDS = [0.70, 0.50, 0.50, 0.50, 0.50, 0.50]

# GA hyper-parameters
POP_SIZE        = 60
MAX_GENS        = 100
TOURNAMENT_K    = 5
CROSSOVER_PROB  = 0.5    # per-gene uniform crossover probability
MUTATION_SIGMA  = 0.04   # Gaussian noise std for each gene
MUTATION_RATE   = 0.25   # fraction of genes mutated per individual
ELITISM_N       = 4      # top individuals copied unchanged each generation
PATIENCE        = 25     # early-stop if best F1 does not improve for N gens

FITNESS_ALPHA   = 0.80   # weight on F1@K
FITNESS_BETA    = 0.20   # weight on flag coverage@K
TOP_K           = 50     # investigation-queue length for fitness evaluation

# Gene bounds
W_LOW, W_HIGH = 0.01, 1.0    # raw weight range (normalised later)
T_LOW, T_HIGH = 0.10, 0.95   # flag threshold range


# ===========================================================================
# Data loading
# ===========================================================================

def _minmax(series: pd.Series) -> pd.Series:
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - lo) / (hi - lo)


def load_data() -> tuple[np.ndarray, np.ndarray, set[str]]:
    """Return (norm_signals_matrix, user_ids, insider_users).

    norm_signals_matrix: shape (n_users, 6) — each column is one normalised
                         signal, ready for dot-product with the weight vector.
    user_ids            : numpy array of user strings (same order as rows).
    insider_users       : ground-truth set from insiders.csv.
    """
    for path in (INSIDERS_CSV, LSTM_CSV, IFOREST_CSV):
        if not path.exists():
            raise FileNotFoundError(
                f"Required file missing: {path}\n"
                "Run the full pipeline first to produce scored CSVs."
            )

    # Ground truth
    ins = pd.read_csv(INSIDERS_CSV)
    insider_users: set[str] = set(ins.loc[ins["dataset"] == 4.2, "user"].unique())

    # LSTM: per-user p95 score
    lstm_df = pd.read_csv(LSTM_CSV, usecols=["user", "lstm_score",
                                              "lstm_risk_severity", "dataset_split"])
    lstm_df = lstm_df[lstm_df["lstm_risk_severity"] != "undetermined"]

    lstm_user = (
        lstm_df.groupby("user")
        .agg(score_p95=("lstm_score",
                        lambda x: float(np.percentile(x.dropna(), 95))
                                  if x.dropna().size else 0.0),
             dataset_split=("dataset_split", lambda x: x.mode().iloc[0]))
        .reset_index()
    )

    # Behavioural signals from the IF-scored CSV (has logon/file/USB columns)
    idf = pd.read_csv(IFOREST_CSV, usecols=[
        "user", "after_hours_ratio", "bcc_ratio",
        "file_to_removable", "file_total",
        "usb_connect_count", "after_hours_logons",
        "logon_count", "unique_logon_pcs",
    ])

    beh = idf.groupby("user").agg(
        after_hours_rate=("after_hours_ratio", "mean"),
        bcc_rate=("bcc_ratio", "mean"),
        total_file_exfil=("file_to_removable", "sum"),
        total_file_ops=("file_total", "sum"),
        total_usb=("usb_connect_count", "sum"),
        total_ah_logons=("after_hours_logons", "sum"),
        total_logons=("logon_count", "sum"),
        max_unique_pcs=("unique_logon_pcs", "max"),
    ).reset_index()
    beh["file_exfil_rate"] = beh["total_file_exfil"] / (beh["total_file_ops"] + 1)
    beh["ah_logon_rate"]   = beh["total_ah_logons"]  / (beh["total_logons"] + 1)

    # Merge LSTM scores + behavioural signals
    df = lstm_user.merge(beh, on="user", how="left").fillna(0)

    # Normalise each signal independently across all users
    df["lstm_p95_norm"]     = _minmax(df["score_p95"])
    df["after_hours_norm"]  = _minmax(df["after_hours_rate"])
    df["bcc_usage_norm"]    = _minmax(df["bcc_rate"])
    df["file_exfil_norm"]   = _minmax(df["file_exfil_rate"])
    df["usb_activity_norm"] = _minmax(df["total_usb"])
    df["multi_pc_norm"]     = _minmax(df["max_unique_pcs"])

    norm_matrix = df[SIGNAL_NORM_COLS].values.astype(np.float32)
    user_ids    = df["user"].values

    return norm_matrix, user_ids, insider_users


# ===========================================================================
# Fitness evaluation (vectorised)
# ===========================================================================

def _f1_at_k(ranked_users: np.ndarray,
             insider_set: set[str],
             k: int) -> float:
    """Compute F1 for the top-k users in ranked_users."""
    top = set(ranked_users[:k])
    tp  = len(top & insider_set)
    if tp == 0:
        return 0.0
    prec = tp / k
    rec  = tp / len(insider_set)
    return 2 * prec * rec / (prec + rec)


def evaluate_chromosome(
    chrom: np.ndarray,
    norm_matrix: np.ndarray,
    user_ids: np.ndarray,
    insider_set: set[str],
    k: int = TOP_K,
) -> float:
    """Compute fitness for a single chromosome.

    chrom[0:6]  raw weights (will be normalised)
    chrom[6:12] flag thresholds
    """
    # Normalise weights to sum = 1
    raw_w = np.clip(chrom[:6], W_LOW, W_HIGH)
    w = raw_w / raw_w.sum()

    # Risk score = weighted sum of normalised signals
    risk_scores = norm_matrix @ w          # shape (n_users,)

    # Rank users by risk score descending
    order = np.argsort(-risk_scores)
    ranked = user_ids[order]

    # Primary: F1 at top-K
    f1 = _f1_at_k(ranked, insider_set, k)

    # Secondary: flag coverage (fraction of top-K with >= 1 flag triggered)
    thresholds = np.clip(chrom[6:12], T_LOW, T_HIGH)
    top_k_signals = norm_matrix[order[:k]]          # (k, 6)
    flags_triggered = (top_k_signals >= thresholds).any(axis=1)
    coverage = flags_triggered.mean()

    return float(FITNESS_ALPHA * f1 + FITNESS_BETA * coverage)


def evaluate_population(
    pop: np.ndarray,
    norm_matrix: np.ndarray,
    user_ids: np.ndarray,
    insider_set: set[str],
    k: int = TOP_K,
) -> np.ndarray:
    """Fitness for every individual in the population (shape (pop_size,))."""
    return np.array([
        evaluate_chromosome(chrom, norm_matrix, user_ids, insider_set, k)
        for chrom in pop
    ], dtype=np.float64)


# ===========================================================================
# GA operators
# ===========================================================================

def _random_chromosome(rng: np.random.Generator) -> np.ndarray:
    weights    = rng.uniform(W_LOW, W_HIGH, 6)
    thresholds = rng.uniform(T_LOW, T_HIGH, 6)
    return np.concatenate([weights, thresholds])


def _tournament_select(pop: np.ndarray,
                       fitness: np.ndarray,
                       rng: np.random.Generator) -> np.ndarray:
    """Select one individual via tournament selection."""
    idx = rng.integers(0, len(pop), TOURNAMENT_K)
    winner = idx[np.argmax(fitness[idx])]
    return pop[winner].copy()


def _crossover(p1: np.ndarray, p2: np.ndarray,
               rng: np.random.Generator) -> np.ndarray:
    """Uniform crossover."""
    mask  = rng.random(len(p1)) < CROSSOVER_PROB
    child = np.where(mask, p1, p2)
    return child


def _mutate(chrom: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Gaussian mutation on a random subset of genes."""
    out  = chrom.copy()
    mask = rng.random(len(out)) < MUTATION_RATE
    out[mask] += rng.normal(0, MUTATION_SIGMA, mask.sum())
    # Clip back to bounds
    out[:6]  = np.clip(out[:6],  W_LOW, W_HIGH)
    out[6:]  = np.clip(out[6:],  T_LOW, T_HIGH)
    return out


# ===========================================================================
# Main GA loop
# ===========================================================================

def run_ga(
    norm_matrix: np.ndarray,
    user_ids: np.ndarray,
    insider_set: set[str],
    pop_size: int   = POP_SIZE,
    max_gens: int   = MAX_GENS,
    seed: int       = 42,
    verbose: bool   = True,
) -> tuple[np.ndarray, list[dict]]:
    """Run the GA and return (best_chromosome, generation_history).

    history is a list of dicts with keys: gen, best_fitness, mean_fitness.
    """
    rng = np.random.default_rng(seed)

    # Seed the population with the baseline chromosome (good starting point)
    baseline = np.array(BASELINE_WEIGHTS + BASELINE_THRESHOLDS, dtype=np.float64)
    pop = np.array([
        baseline if i == 0 else _random_chromosome(rng)
        for i in range(pop_size)
    ], dtype=np.float64)

    fitness  = evaluate_population(pop, norm_matrix, user_ids, insider_set)
    history: list[dict] = []

    best_fitness  = float(fitness.max())
    best_chrom    = pop[fitness.argmax()].copy()
    no_improve    = 0

    if verbose:
        print(f"GA | pop={pop_size}  max_gens={max_gens}  "
              f"elitism={ELITISM_N}  patience={PATIENCE}")
        print(f"    baseline fitness = {best_fitness:.4f}\n")
        print(f"  {'Gen':>4}  {'Best':>8}  {'Mean':>8}  {'Change':>8}")
        print("  " + "-" * 36)

    for gen in range(1, max_gens + 1):
        # Sort by fitness (descending)
        order   = np.argsort(-fitness)
        pop     = pop[order]
        fitness = fitness[order]

        # Elitism: carry best individuals forward
        new_pop = [pop[i].copy() for i in range(ELITISM_N)]

        # Fill the rest via selection + crossover + mutation
        while len(new_pop) < pop_size:
            p1 = _tournament_select(pop, fitness, rng)
            p2 = _tournament_select(pop, fitness, rng)
            child = _crossover(p1, p2, rng)
            child = _mutate(child, rng)
            new_pop.append(child)

        pop     = np.array(new_pop, dtype=np.float64)
        fitness = evaluate_population(pop, norm_matrix, user_ids, insider_set)

        gen_best = float(fitness.max())
        gen_mean = float(fitness.mean())

        if gen_best > best_fitness + 1e-6:
            delta       = gen_best - best_fitness
            best_fitness = gen_best
            best_chrom  = pop[fitness.argmax()].copy()
            no_improve  = 0
        else:
            delta      = 0.0
            no_improve += 1

        history.append({
            "gen": gen,
            "best_fitness": round(gen_best, 6),
            "mean_fitness": round(gen_mean, 6),
        })

        if verbose and (gen % 10 == 0 or gen <= 5 or no_improve == 0):
            marker = " *" if delta > 0 else ""
            print(f"  {gen:>4}  {gen_best:>8.4f}  {gen_mean:>8.4f}  {delta:>+8.4f}{marker}")

        if no_improve >= PATIENCE:
            if verbose:
                print(f"\n  Early stop: no improvement for {PATIENCE} generations.")
            break

    if verbose:
        print(f"\n  Final best fitness = {best_fitness:.4f}\n")

    return best_chrom, history


# ===========================================================================
# Build detailed results for the best chromosome
# ===========================================================================

def build_results(
    best_chrom: np.ndarray,
    norm_matrix: np.ndarray,
    user_ids: np.ndarray,
    insider_set: set[str],
    k: int = TOP_K,
) -> dict:
    """Compute per-signal breakdown and F1 metrics for the best chromosome."""
    raw_w = np.clip(best_chrom[:6], W_LOW, W_HIGH)
    w     = raw_w / raw_w.sum()
    thresholds = np.clip(best_chrom[6:12], T_LOW, T_HIGH)

    risk_scores = norm_matrix @ w
    order       = np.argsort(-risk_scores)
    ranked      = user_ids[order]

    top_k_set = set(ranked[:k])
    tp  = len(top_k_set & insider_set)
    fp  = k - tp
    fn  = len(insider_set) - tp
    prec   = tp / k if k > 0 else 0.0
    recall = tp / len(insider_set) if insider_set else 0.0
    f1     = (2 * prec * recall / (prec + recall)) if (prec + recall) > 0 else 0.0

    top_k_signals = norm_matrix[order[:k]]
    coverage = float((top_k_signals >= thresholds).any(axis=1).mean())

    # Baseline comparison
    bw = np.array(BASELINE_WEIGHTS, dtype=np.float64)
    baseline_scores = norm_matrix @ bw
    baseline_order  = np.argsort(-baseline_scores)
    baseline_top    = set(user_ids[baseline_order[:k]])
    baseline_tp     = len(baseline_top & insider_set)
    b_prec   = baseline_tp / k
    b_recall = baseline_tp / len(insider_set)
    b_f1     = (2 * b_prec * b_recall / (b_prec + b_recall)) if (b_prec + b_recall) > 0 else 0.0

    return {
        "optimised_weights":    {n: float(round(v, 6)) for n, v in zip(SIGNAL_NAMES, w.tolist())},
        "optimised_thresholds": {n: float(round(v, 4)) for n, v in zip(SIGNAL_NAMES, thresholds.tolist())},
        "metrics_at_k": {
            "k": k, "tp": tp, "fp": fp, "fn": fn,
            "precision": round(prec, 4),
            "recall":    round(recall, 4),
            "f1":        round(f1, 4),
            "flag_coverage": round(coverage, 4),
        },
        "baseline_metrics_at_k": {
            "k": k, "tp": baseline_tp,
            "precision": round(b_prec, 4),
            "recall":    round(b_recall, 4),
            "f1":        round(b_f1, 4),
        },
        "improvement": {
            "delta_f1":       round(f1 - b_f1, 4),
            "delta_precision": round(prec - b_prec, 4),
            "delta_recall":   round(recall - b_recall, 4),
        },
    }


# ===========================================================================
# Save outputs
# ===========================================================================

def save_outputs(
    best_chrom: np.ndarray,
    history: list[dict],
    results: dict,
    pop_size: int,
    max_gens: int,
    elapsed: float,
) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    raw_w = np.clip(best_chrom[:6], W_LOW, W_HIGH)
    w     = raw_w / raw_w.sum()
    thresholds = np.clip(best_chrom[6:12], T_LOW, T_HIGH)

    config = {
        "_description": (
            "GA-optimised weights and flag thresholds for risk_scorer.py. "
            "Loaded automatically by risk_scorer when this file is present."
        ),
        "weights":     {n: float(round(v, 6)) for n, v in zip(SIGNAL_NAMES, w.tolist())},
        "thresholds":  {n: float(round(v, 4)) for n, v in zip(SIGNAL_NAMES, thresholds.tolist())},
        "fitness":     round(results["metrics_at_k"]["f1"] * FITNESS_ALPHA
                             + results["metrics_at_k"]["flag_coverage"] * FITNESS_BETA, 6),
        "f1_at_k":    results["metrics_at_k"]["f1"],
        "k":           results["metrics_at_k"]["k"],
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    GA_CONFIG_PATH.write_text(json.dumps(config, indent=2))

    report = {
        "ga_parameters": {
            "pop_size":       pop_size,
            "max_gens":       max_gens,
            "tournament_k":   TOURNAMENT_K,
            "crossover_prob": CROSSOVER_PROB,
            "mutation_sigma": MUTATION_SIGMA,
            "mutation_rate":  MUTATION_RATE,
            "elitism_n":      ELITISM_N,
            "patience":       PATIENCE,
            "fitness_alpha":  FITNESS_ALPHA,
            "fitness_beta":   FITNESS_BETA,
            "top_k":          TOP_K,
        },
        "baseline_weights":    dict(zip(SIGNAL_NAMES, BASELINE_WEIGHTS)),
        "baseline_thresholds": dict(zip(SIGNAL_NAMES, BASELINE_THRESHOLDS)),
        "results":             results,
        "elapsed_seconds":     round(elapsed, 1),
        "convergence_history": history,
    }
    GA_REPORT_PATH.write_text(json.dumps(report, indent=2))

    print(f"  Config  -> {GA_CONFIG_PATH}")
    print(f"  Report  -> {GA_REPORT_PATH}")


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="GA weight + threshold optimiser")
    parser.add_argument("--quick", action="store_true",
                        help="Run 20 generations (smoke test / CI)")
    parser.add_argument("--pop", type=int, default=POP_SIZE)
    parser.add_argument("--gens", type=int, default=MAX_GENS)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    max_gens = 20 if args.quick else args.gens
    pop_size = args.pop

    print("=" * 60)
    print("  GA Optimizer — UEBA+DLP Risk Scoring System")
    print("=" * 60)
    print(f"\n  Loading data from {CLEANED_DIR.name}/...")
    norm_matrix, user_ids, insider_set = load_data()
    n_users    = len(user_ids)
    n_insiders = len(insider_set & set(user_ids))
    print(f"  Users: {n_users}  |  Insiders: {n_insiders}  |  Signals: {len(SIGNAL_NAMES)}\n")

    t0 = time.time()
    best_chrom, history = run_ga(
        norm_matrix, user_ids, insider_set,
        pop_size=pop_size, max_gens=max_gens,
        seed=args.seed, verbose=True,
    )
    elapsed = time.time() - t0

    results = build_results(best_chrom, norm_matrix, user_ids, insider_set)

    print("  Optimised weights:")
    for name, val in results["optimised_weights"].items():
        baseline = dict(zip(SIGNAL_NAMES, BASELINE_WEIGHTS))[name]
        print(f"    {name:<15} {val:.4f}  (baseline {baseline:.2f})")

    print("\n  Optimised flag thresholds:")
    for name, val in results["optimised_thresholds"].items():
        baseline = dict(zip(SIGNAL_NAMES, BASELINE_THRESHOLDS))[name]
        print(f"    {name:<15} {val:.4f}  (baseline {baseline:.2f})")

    m  = results["metrics_at_k"]
    bm = results["baseline_metrics_at_k"]
    im = results["improvement"]
    print(f"\n  Performance at K={m['k']}:")
    print(f"    {'':20} {'GA':>8}  {'Baseline':>8}  {'Delta':>8}")
    print(f"    {'Precision':<20} {m['precision']:>8.4f}  {bm['precision']:>8.4f}  {im['delta_precision']:>+8.4f}")
    print(f"    {'Recall':<20} {m['recall']:>8.4f}  {bm['recall']:>8.4f}  {im['delta_recall']:>+8.4f}")
    print(f"    {'F1':<20} {m['f1']:>8.4f}  {bm['f1']:>8.4f}  {im['delta_f1']:>+8.4f}")
    print(f"    {'TP / {0} insiders'.format(n_insiders):<20} {m['tp']:>8}  {bm['tp']:>8}")
    print(f"    {'Flag coverage@K':<20} {m['flag_coverage']:>8.4f}")
    print(f"\n  Elapsed: {elapsed:.1f}s")

    print("\n  Saving outputs...")
    save_outputs(best_chrom, history, results, pop_size, max_gens, elapsed)
    print("\nDone.\n")


if __name__ == "__main__":
    main()
