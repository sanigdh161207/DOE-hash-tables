import os
import csv
from typing import Dict, Any, Optional

def calculate_heuristic_score(results_dir: str = "results") -> Dict[str, Any]:
    """
    Calculates a transparent, reproducible 1-10 heuristic quality score reading actual result CSV files.
    """
    quality_csv = os.path.join(results_dir, "quality_baseline_vs_cosine.csv")
    
    precision = 0.0
    hit_rate = 0.0
    coverage = 0.0
    diversity = 0.0
    latency_sec = 0.001

    if os.path.exists(quality_csv):
        with open(quality_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("recommender") in ["Cosine Recommender", "Weighted Cosine", "cosine"]:
                    precision = float(row.get("precision@k", 0.0))
                    hit_rate = float(row.get("hit_rate@k", 0.0))
                    coverage = float(row.get("coverage", 0.0))
                    diversity = float(row.get("diversity", 0.0))
                    latency_sec = float(row.get("latency_sec", 0.001))
                    break

    # Rubric scoring (out of 10 for each dimension)
    score_precision = min(10.0, (precision / 0.40) * 10.0)
    score_hit_rate = min(10.0, (hit_rate / 0.70) * 10.0)
    score_coverage = min(10.0, (coverage / 0.80) * 10.0)
    score_diversity = min(10.0, (diversity / 0.80) * 10.0)
    
    # Latency score: < 1ms -> 10 pts, < 5ms -> 8 pts, < 10ms -> 6 pts
    if latency_sec <= 0.001:
        score_latency = 10.0
    elif latency_sec <= 0.005:
        score_latency = 8.5
    elif latency_sec <= 0.01:
        score_latency = 7.0
    else:
        score_latency = 5.0

    # Weighted final aggregate score
    weights = {
        "precision": 0.25,
        "hit_rate": 0.20,
        "coverage": 0.20,
        "diversity": 0.15,
        "latency": 0.20
    }

    final_score = (
        score_precision * weights["precision"] +
        score_hit_rate * weights["hit_rate"] +
        score_coverage * weights["coverage"] +
        score_diversity * weights["diversity"] +
        score_latency * weights["latency"]
    )

    final_score = round(min(10.0, max(1.0, final_score)), 2)

    return {
        "final_score": final_score,
        "dimension_scores": {
            "Precision@5 Score": round(score_precision, 2),
            "Hit-Rate@5 Score": round(score_hit_rate, 2),
            "Coverage Score": round(score_coverage, 2),
            "Diversity Score": round(score_diversity, 2),
            "Latency Score": round(score_latency, 2)
        },
        "raw_measured_values": {
            "precision@5": precision,
            "hit_rate@5": hit_rate,
            "coverage": coverage,
            "diversity": diversity,
            "latency_sec": latency_sec
        }
    }

if __name__ == "__main__":
    res = calculate_heuristic_score()
    print("============================================================")
    print(f"AUTOMATED HEURISTIC QUALITY SCORE: {res['final_score']} / 10.0")
    print("============================================================")
    for dim, sc in res["dimension_scores"].items():
        print(f"  - {dim:<20}: {sc:.2f} / 10.0")
    print("\nRaw Measured Values:")
    for k, v in res["raw_measured_values"].items():
        print(f"  - {k:<20}: {v}")
    print("============================================================")
