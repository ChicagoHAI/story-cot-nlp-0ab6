import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2


def load_results(path: str) -> pd.DataFrame:
    df = pd.read_json(path, lines=True)
    return df


def bootstrap_diff(pivot: pd.DataFrame, a: str, b: str, n_boot: int = 1000, seed: int = 42) -> Tuple[float, Tuple[float, float]]:
    rng = np.random.default_rng(seed)
    diffs = pivot[a].astype(int) - pivot[b].astype(int)
    boot_means = []
    for _ in range(n_boot):
        sample = rng.choice(diffs, size=len(diffs), replace=True)
        boot_means.append(sample.mean())
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return float(diffs.mean()), (float(lo), float(hi))


def mcnemar_test(pivot: pd.DataFrame, a: str, b: str) -> Dict[str, float]:
    both = pivot[[a, b]].dropna()
    b01 = int(((both[a] == 0) & (both[b] == 1)).sum())
    b10 = int(((both[a] == 1) & (both[b] == 0)).sum())
    if b01 + b10 == 0:
        return {"b01": b01, "b10": b10, "stat": 0.0, "p_value": 1.0}
    stat = (abs(b01 - b10) - 1) ** 2 / (b01 + b10)
    p_value = float(chi2.sf(stat, 1))
    return {"b01": b01, "b10": b10, "stat": float(stat), "p_value": p_value}


def summarize(df: pd.DataFrame, save_dir: str = "results") -> Dict[str, Dict]:
    summary: Dict[str, Dict] = {}
    plots_dir = Path(save_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for dataset in df["dataset"].unique():
        df_ds = df[df["dataset"] == dataset]
        pivot = df_ds.pivot_table(index="sample_id", columns="style", values="correct", aggfunc="first")
        acc = df_ds.groupby("style")["correct"].mean().sort_values(ascending=False)

        # bootstrap story vs cot if available
        if "story" in pivot.columns and "cot" in pivot.columns:
            mean_diff, ci = bootstrap_diff(pivot[["story", "cot"]].dropna(), "story", "cot")
            summary[f"{dataset}_story_vs_cot"] = {"mean_diff": mean_diff, "ci95": ci}
            summary[f"{dataset}_mcnemar"] = mcnemar_test(pivot, "story", "cot")

        # visualize accuracy
        plt.figure(figsize=(6, 4))
        sns.barplot(x=acc.index, y=acc.values, palette="viridis")
        plt.ylim(0, 1)
        plt.ylabel("Accuracy")
        plt.title(f"Accuracy by style - {dataset}")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plot_path = plots_dir / f"{dataset}_accuracy.png"
        plt.savefig(plot_path, dpi=200)
        plt.close()

        # token costs
        cost = df_ds.groupby("style")[["usage_prompt_tokens", "usage_completion_tokens"]].mean()
        summary[f"{dataset}_accuracy"] = acc.to_dict()
        summary[f"{dataset}_token_usage_mean"] = cost.to_dict()

    metrics_path = Path(save_dir) / "analysis.json"
    with metrics_path.open("w") as f:
        json.dump(summary, f, indent=2)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Story CoT experiment outputs.")
    parser.add_argument("--raw_path", default="results/raw_outputs.jsonl", help="Path to raw outputs JSONL.")
    parser.add_argument("--save_dir", default="results", help="Directory to save analysis and plots.")
    args = parser.parse_args()

    df_results = load_results(args.raw_path)
    stats = summarize(df_results, save_dir=args.save_dir)
    print(json.dumps(stats, indent=2))
