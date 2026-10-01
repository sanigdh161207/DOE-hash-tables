import os
import matplotlib
matplotlib.use("Agg")  # Always use headless backend — no display server required
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Dict, Union, Any

DARK_BACKGROUND = "#1A1A1A"
CARD_BACKGROUND = "#242424"
PRIMARY_BLUE = "#1F6AA5"
SECONDARY_RED = "#D62728"
SECONDARY_ACCENT = "#FF7F0E"
GREEN_ACCENT = "#2CA02C"
PURPLE_ACCENT = "#9467BD"
TEXT_COLOR = "#E0E0E0"
GRID_COLOR = "#333333"

def setup_plot_style(ax, title: str, xlabel: str, ylabel: str):
    ax.set_facecolor(CARD_BACKGROUND)
    ax.figure.patch.set_facecolor(DARK_BACKGROUND)
    ax.grid(True, color=GRID_COLOR, linestyle="--", alpha=0.5)
    ax.set_title(title, color=TEXT_COLOR, fontsize=11, pad=12, fontweight="bold")
    ax.set_xlabel(xlabel, color=TEXT_COLOR, fontsize=9, labelpad=8)
    ax.set_ylabel(ylabel, color=TEXT_COLOR, fontsize=9, labelpad=8)
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)

def plot_lookup_benchmark(data: Union[List[Tuple[int, float, int, int]], Dict[str, List[Tuple[int, float, int, int]]]]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 3.5), dpi=100)
    setup_plot_style(ax, "Dataset Size vs. Lookup Time", "Dataset Size (Users)", "Avg Lookup Time (microseconds)")
    
    if isinstance(data, dict):
        max_y = 1.0
        for strat, results in data.items():
            sizes = [item[0] for item in results]
            times_us = [item[1] * 1_000_000 for item in results]
            color = PRIMARY_BLUE if strat == "Separate Chaining" else SECONDARY_RED
            ax.plot(sizes, times_us, marker="o", color=color, linewidth=2.0, markersize=5, label=strat)
            if times_us:
                max_y = max(max_y, max(times_us))
        ax.legend(facecolor=CARD_BACKGROUND, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, fontsize=8)
        ax.set_ylim(0, max_y * 1.3)
    else:
        sizes = [item[0] for item in data]
        times_us = [item[1] * 1_000_000 for item in data]
        ax.plot(sizes, times_us, marker="o", color=PRIMARY_BLUE, linewidth=2.5, markersize=6, label="Hash Table")
        ax.set_ylim(0, max(times_us) * 1.3 if times_us else 10)
        
    fig.tight_layout()
    return fig

def plot_collision_experiment(data: Union[List[Tuple[float, float, float]], Dict[str, List[Tuple[float, float, float, int, dict]]]]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 3.5), dpi=100)
    setup_plot_style(ax, "Load Factor vs. Collision Rate", "Load Factor (N / M)", "Collision Rate (%)")
    
    if isinstance(data, dict):
        strats = list(data.keys())
        lf_keys = [str(item[0]) for item in data[strats[0]]]
        x = np.arange(len(lf_keys))
        width = 0.35
        
        for i, strat in enumerate(strats):
            rates = [item[1] * 100 for item in data[strat]]
            color = PRIMARY_BLUE if strat == "Separate Chaining" else SECONDARY_RED
            offset = -width/2 if i == 0 else width/2
            bars = ax.bar(x + offset, rates, width, color=color, label=strat, edgecolor=GRID_COLOR)
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f"{height:.0f}%",
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 2),
                            textcoords="offset points",
                            ha="center", va="bottom", color=TEXT_COLOR, fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(lf_keys)
        ax.legend(facecolor=CARD_BACKGROUND, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, fontsize=8)
        all_rates = []
        for s in strats:
            all_rates.extend([item[1] * 100 for item in data[s]])
        ax.set_ylim(0, max(all_rates) * 1.2 if all_rates else 100)
    else:
        load_factors = [item[0] for item in data]
        collision_rates = [item[1] * 100 for item in data]
        colors = [PRIMARY_BLUE if cr < 20 else SECONDARY_ACCENT for cr in collision_rates]
        bars = ax.bar([str(lf) for lf in load_factors], collision_rates, color=colors, width=0.5, edgecolor=GRID_COLOR)
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.1f}%",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center", va="bottom", color=TEXT_COLOR, fontsize=8)
        ax.set_ylim(0, max(collision_rates) * 1.2 if collision_rates else 100)
        
    fig.tight_layout()
    return fig

def plot_load_factor_vs_lookup_time(data: Dict[str, List[Tuple[float, float, float, int, dict]]]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 3.5), dpi=100)
    setup_plot_style(ax, "Load Factor vs. Lookup Time", "Load Factor (N / M)", "Avg Lookup Time (microseconds)")
    
    max_y = 1.0
    for strat, results in data.items():
        lfs = [item[0] for item in results]
        times_us = [item[2] * 1_000_000 for item in results]
        color = PRIMARY_BLUE if strat == "Separate Chaining" else SECONDARY_RED
        ax.plot(lfs, times_us, marker="o", color=color, linewidth=2.0, markersize=5, label=strat)
        if times_us:
            max_y = max(max_y, max(times_us))
            
    ax.legend(facecolor=CARD_BACKGROUND, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, fontsize=8)
    ax.set_ylim(0, max_y * 1.3)
    fig.tight_layout()
    return fig

def plot_quality_comparison(baseline_metrics: Dict[str, float], cosine_metrics: Dict[str, float]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.5, 3.5), dpi=100)
    setup_plot_style(ax, "Baseline vs. Cosine Recommender Quality", "Metric", "Score (0.0 to 1.0)")
    
    metrics = ["precision@k", "recall@k", "hit_rate@k", "ndcg@k", "coverage", "diversity"]
    labels = ["P@5", "R@5", "HR@5", "NDCG@5", "Cov", "Div"]
    
    b_vals = [baseline_metrics.get(m, 0.0) for m in metrics]
    c_vals = [cosine_metrics.get(m, 0.0) for m in metrics]
    
    x = np.arange(len(labels))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, b_vals, width, label="Baseline (Random)", color=SECONDARY_RED, edgecolor=GRID_COLOR)
    bars2 = ax.bar(x + width/2, c_vals, width, label="Cosine Similarity", color=PRIMARY_BLUE, edgecolor=GRID_COLOR)
    
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.annotate(f"{height:.2f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", color=TEXT_COLOR, fontsize=7)
                    
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.15)
    ax.legend(facecolor=CARD_BACKGROUND, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, fontsize=8)
    fig.tight_layout()
    return fig

def plot_cold_start_experiment(cold_start_rows: List[Dict[str, Any]]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.5, 3.5), dpi=100)
    setup_plot_style(ax, "Cold Start Performance by Seed Count", "Number of Seed Items", "Hit Rate @ 5")
    
    seeds = sorted(list(set(r["n_seed"] for r in cold_start_rows)))
    strategies = sorted(list(set(r["strategy"] for r in cold_start_rows)))
    
    colors = {"Random Baseline": SECONDARY_RED, "Popularity Fallback": SECONDARY_ACCENT, "Seed-Based Cosine": PRIMARY_BLUE}
    
    for strat in strategies:
        vals = [next((r["hit_rate@k"] for r in cold_start_rows if r["n_seed"] == s and r["strategy"] == strat), 0.0) for s in seeds]
        ax.plot(seeds, vals, marker="o", linewidth=2.0, label=strat, color=colors.get(strat, PURPLE_ACCENT))
        
    ax.set_xticks(seeds)
    ax.set_ylim(0, 1.15)
    ax.legend(facecolor=CARD_BACKGROUND, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, fontsize=8)
    fig.tight_layout()
    return fig

def plot_weighted_cosine_comparison(rows: List[Dict[str, Any]]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.5, 3.5), dpi=100)
    setup_plot_style(ax, "Plain Cosine vs IDF-Weighted Cosine", "Metric", "Score / Bias")
    
    strats = [r["strategy"] for r in rows]
    metrics = ["precision@k", "ndcg@k", "coverage", "diversity"]
    
    x = np.arange(len(metrics))
    width = 0.25
    colors = [SECONDARY_RED, PRIMARY_BLUE, GREEN_ACCENT]
    
    for idx, r in enumerate(rows):
        vals = [r.get(m, 0.0) for m in metrics]
        offset = (idx - 1) * width
        bars = ax.bar(x + offset, vals, width, label=r["strategy"], color=colors[idx % len(colors)], edgecolor=GRID_COLOR)
        
    ax.set_xticks(x)
    ax.set_xticklabels(["P@5", "NDCG@5", "Coverage", "Diversity"])
    ax.set_ylim(0, 1.15)
    ax.legend(facecolor=CARD_BACKGROUND, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, fontsize=7)
    fig.tight_layout()
    return fig

def plot_sparse_memory_vs_users(rows: List[Dict[str, Any]]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.5, 3.5), dpi=100)
    setup_plot_style(ax, "Dense vs Sparse Memory (1000 Movies)", "Users", "Memory (MB)")
    
    users = [r["users"] for r in rows if r["movies"] == 1000]
    dense_mb = [r["dense_mem_bytes"] / (1024**2) for r in rows if r["movies"] == 1000]
    csr_mb = [r["csr_mem_bytes"] / (1024**2) for r in rows if r["movies"] == 1000]
    
    ax.plot(users, dense_mb, marker="o", color=SECONDARY_RED, linewidth=2.0, label="Dense Float64")
    ax.plot(users, csr_mb, marker="s", color=PRIMARY_BLUE, linewidth=2.0, label="Sparse CSR")
    
    ax.set_yscale("log")
    ax.set_ylabel("Memory in MB (Log Scale)")
    ax.legend(facecolor=CARD_BACKGROUND, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, fontsize=8)
    fig.tight_layout()
    return fig

def plot_extendible_vs_rehash(rows: List[Dict[str, Any]]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.5, 3.5), dpi=100)
    setup_plot_style(ax, "Extendible Hashing vs. Full Rehashing", "Number of Insertions", "Keys Moved During Rehash / Split")
    
    sizes = [r["dataset_size"] for r in rows]
    rehash_moved = [r["rehash_entries_moved"] for r in rows]
    extendible_moved = [r["extendible_entries_moved"] for r in rows]
    
    ax.plot(sizes, rehash_moved, marker="o", color=SECONDARY_RED, linewidth=2.0, label="Full Rehashing (Chaining)")
    ax.plot(sizes, extendible_moved, marker="s", color=PRIMARY_BLUE, linewidth=2.0, label="Extendible Bucket Splitting")
    
    ax.legend(facecolor=CARD_BACKGROUND, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, fontsize=8)
    fig.tight_layout()
    return fig

def plot_scaling_loglog(rows: List[Dict[str, Any]], slopes: Dict[str, float]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.5, 3.5), dpi=100)
    setup_plot_style(ax, "Empirical Scaling Log-Log Plot", "Users (Log Scale)", "Total Time sec (Log Scale)")
    
    strats = sorted(list(set(r["recommender"] for r in rows)))
    colors = {"baseline": SECONDARY_RED, "cosine": PRIMARY_BLUE, "weighted_cosine": SECONDARY_ACCENT, "sparse": GREEN_ACCENT}
    
    for strat in strats:
        sub = [r for r in rows if r["recommender"] == strat]
        sub.sort(key=lambda x: x["users"])
        x_vals = [r["users"] for r in sub]
        y_vals = [r["total_time_median_sec"] for r in sub]
        slope = slopes.get(strat, 1.0)
        
        ax.plot(x_vals, y_vals, marker="o", label=f"{strat} (slope={slope:.2f})", color=colors.get(strat, PURPLE_ACCENT))
        
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(facecolor=CARD_BACKGROUND, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, fontsize=8)
    fig.tight_layout()
    return fig
