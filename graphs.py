import matplotlib
matplotlib.use("TkAgg")  # Ensure TkAgg backend is set for embedding in Tkinter
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Dict, Union

# Custom styling to match application dark mode
DARK_BACKGROUND = "#1A1A1A"
CARD_BACKGROUND = "#242424"
PRIMARY_BLUE = "#1F6AA5"
SECONDARY_RED = "#D62728"   # Red accent for Linear Probing comparison
SECONDARY_ACCENT = "#FF7F0E" # Orange for highlights
TEXT_COLOR = "#E0E0E0"
GRID_COLOR = "#333333"

def setup_plot_style(ax, title: str, xlabel: str, ylabel: str):
    """
    Applies custom styling to the matplotlib axis to blend with the dark UI theme.
    """
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
    """
    Generates a line plot showing Dataset Size vs. Average Lookup Time.
    Supports comparison plotting for both Separate Chaining and Linear Probing.
    """
    fig, ax = plt.subplots(figsize=(5, 3.5), dpi=100)
    setup_plot_style(ax, "Dataset Size vs. Lookup Time", "Dataset Size (Users)", "Avg Lookup Time (microseconds)")
    
    if isinstance(data, dict):
        # Comparison mode
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
        # Legacy list fallback
        sizes = [item[0] for item in data]
        times_us = [item[1] * 1_000_000 for item in data]
        ax.plot(sizes, times_us, marker="o", color=PRIMARY_BLUE, linewidth=2.5, markersize=6, label="Hash Table")
        ax.set_ylim(0, max(times_us) * 1.3 if times_us else 10)
        
    fig.tight_layout()
    return fig

def plot_collision_experiment(data: Union[List[Tuple[float, float, float]], Dict[str, List[Tuple[float, float, float, int, dict]]]]) -> plt.Figure:
    """
    Generates a bar plot showing Load Factor vs. Collision Rate.
    """
    fig, ax = plt.subplots(figsize=(5, 3.5), dpi=100)
    setup_plot_style(ax, "Load Factor vs. Collision Rate", "Load Factor (N / M)", "Collision Rate (%)")
    
    if isinstance(data, dict):
        # Comparison mode
        strats = list(data.keys())
        # We assume load factors are identical across both runs
        lf_keys = [str(item[0]) for item in data[strats[0]]]
        x = np.arange(len(lf_keys))
        width = 0.35
        
        # Plot bars side by side
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
        # Legacy list fallback
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
    """
    Generates a line plot comparing Load Factor vs. Average Lookup Time for both strategies.
    """
    fig, ax = plt.subplots(figsize=(5, 3.5), dpi=100)
    setup_plot_style(ax, "Load Factor vs. Lookup Time", "Load Factor (N / M)", "Avg Lookup Time (microseconds)")
    
    max_y = 1.0
    for strat, results in data.items():
        lfs = [item[0] for item in results]
        # Item 2 is lookup time
        times_us = [item[2] * 1_000_000 for item in results]
        color = PRIMARY_BLUE if strat == "Separate Chaining" else SECONDARY_RED
        ax.plot(lfs, times_us, marker="o", color=color, linewidth=2.0, markersize=5, label=strat)
        if times_us:
            max_y = max(max_y, max(times_us))
            
    ax.legend(facecolor=CARD_BACKGROUND, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, fontsize=8)
    ax.set_ylim(0, max_y * 1.3)
    
    fig.tight_layout()
    return fig
