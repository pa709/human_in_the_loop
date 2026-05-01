
import argparse
import csv
import json
import math
from collections import defaultdict


def wilson_ci(successes, n, z=1.96):
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", help="rollout_results.json")
    ap.add_argument("--csv", default=None, help="optional CSV output path")
    args = ap.parse_args()

    with open(args.results) as f:
        results = json.load(f)

    # group by model
    by_model = defaultdict(list)
    for r in results:
        by_model[r["model"]].append(r)

    # table
    model_order = ["mlp", "lstm_tf", "lstm_ss"]
    model_labels = {
        "mlp":     "MLP            ",
        "lstm_tf": "LSTM-TF (AR)   ",
        "lstm_ss": "LSTM-SS (AR)   ",
    }

    print()
    print("=" * 72)
    print("Rollout Success Rate (Test 1: Training-Set Replay, Approximate Reset)")
    print("=" * 72)
    print(f"{'Model':<16} {'Successes':>10} {'Rate':>8} {'95% CI (Wilson)':>25} {'Errors':>8}")
    print("-" * 72)

    rows_for_csv = []
    for m in model_order:
        if m not in by_model:
            continue
        rs = by_model[m]
        n = len(rs)
        n_err = sum(1 for r in rs if r["error"] is not None)
        n_success = sum(1 for r in rs if r["success"])
        p, lo, hi = wilson_ci(n_success, n)

        print(f"{model_labels[m]:<16} {n_success:>4}/{n:<5} "
              f"{100*p:>6.1f}%  [{100*lo:>5.1f}% - {100*hi:>5.1f}%]"
              f"  {n_err:>6}")

        rows_for_csv.append({
            "model": m,
            "n": n,
            "successes": n_success,
            "rate": p,
            "ci_low": lo,
            "ci_high": hi,
            "errors": n_err,
        })

    print("=" * 72)
    print()

    # per-demo breakdown: for each demo, which models succeeded
    print("Per-demo breakdown (S = success, . = fail, E = error):")
    print()
    demos = sorted({r["demo"] for r in results}, key=lambda x: int(x.split("_")[1]))
    header = "Demo         " + " ".join(f"{m[:7]:>7}" for m in model_order if m in by_model)
    print(header)
    print("-" * len(header))
    for d in demos:
        cells = []
        for m in model_order:
            if m not in by_model:
                continue
            match = [r for r in by_model[m] if r["demo"] == d]
            if not match:
                cells.append(f"{'-':>7}")
            else:
                r = match[0]
                if r["error"] is not None:
                    cells.append(f"{'E':>7}")
                elif r["success"]:
                    cells.append(f"{'S':>7}")
                else:
                    cells.append(f"{'.':>7}")
        print(f"{d:<13}" + " ".join(cells))
    print()

    # optional CSV
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["model", "n", "successes", "rate", "ci_low", "ci_high", "errors"])
            w.writeheader()
            for row in rows_for_csv:
                w.writerow(row)
        print(f"Summary CSV: {args.csv}")


if __name__ == "__main__":
    main()
