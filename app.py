import tkinter as tk
from tkinter import messagebox
import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time
import random
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

# Import custom modules
from data_generator import generate_user_data, format_first_n_users
from recommender import HashTable, RecommenderSystem, HashTableChaining, HashTableLinearProbing
from simulator import PerformanceSimulator
import graphs

# Set appearance mode and default color theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# ==========================================
# MODERN DESIGN TOKENS & PALETTE
# ==========================================
COLOR_BG_ROOT = "#0C0E14"          # Deep obsidian black
COLOR_BG_SIDEBAR = "#12151F"       # Sleek sidebar charcoal
COLOR_BG_MAIN = "#0F121C"          # Main panel background
COLOR_BG_CARD = "#171B26"          # Standard card background
COLOR_BG_CARD_ELEVATED = "#1E2333" # Hover / elevated card background
COLOR_BG_INPUT = "#131622"         # Input background
COLOR_BORDER = "#252B3B"           # Subtle card border
COLOR_BORDER_HOVER = "#3B4259"     # Card border on hover
COLOR_ACCENT_PRIMARY = "#6366F1"   # Electric Indigo / Violet
COLOR_ACCENT_HOVER = "#4F46E5"     # Darker Indigo
COLOR_ACCENT_CYAN = "#00E5FF"      # Cyber Cyan
COLOR_ACCENT_CYAN_DIM = "#00838F"
COLOR_SUCCESS = "#10B981"          # Emerald Green
COLOR_WARNING = "#F59E0B"          # Amber / Orange
COLOR_DANGER = "#EF4444"           # Coral Red
COLOR_TEXT_PRIMARY = "#F8FAFC"     # Bright white
COLOR_TEXT_SECONDARY = "#94A3B8"   # Slate gray
COLOR_TEXT_MUTED = "#64748B"       # Dim gray


class AnimatedActionButton(ctk.CTkButton):
    """
    Enhanced interactive button with:
    - Dynamic hover animations
    - Border glowing transitions
    - Real-time event broadcasting to Output Section preview
    """
    def __init__(
        self,
        master,
        action_id: str,
        title: str,
        description: str,
        hover_handler=None,
        leave_handler=None,
        accent_color: str = COLOR_ACCENT_PRIMARY,
        hover_accent: str = COLOR_ACCENT_HOVER,
        **kwargs
    ):
        self.action_id = action_id
        self.action_title = title
        self.action_description = description
        self.accent_color = accent_color
        self.hover_handler = hover_handler
        self.leave_handler = leave_handler
        self._default_border = COLOR_BORDER

        super().__init__(
            master,
            text=title,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color=kwargs.pop("fg_color", COLOR_BG_CARD_ELEVATED),
            hover_color=kwargs.pop("hover_color", hover_accent),
            border_color=COLOR_BORDER,
            border_width=1.5,
            corner_radius=8,
            height=36,
            **kwargs
        )

    def _on_enter(self, event=None):
        super()._on_enter(event)
        try:
            self.configure(border_color=self.accent_color)
        except Exception:
            pass
        if self.hover_handler:
            self.hover_handler(self.action_title, self.action_description, self.accent_color)

    def _on_leave(self, event=None):
        super()._on_leave(event)
        try:
            self.configure(border_color=self._default_border)
        except Exception:
            pass
        if self.leave_handler:
            self.leave_handler()


class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Configure window settings
        self.title("Hash Table Recommender Simulator")
        self.geometry("1260x820")
        self.minsize(1100, 700)
        self.configure(fg_color=COLOR_BG_ROOT)

        # Application state
        self.dataset_size = 100
        self.table_size_mode = "Auto"  # "Auto" or "Custom"
        self.custom_table_size = 150
        self.selected_load_factor = 0.75
        self.dataset: List[Tuple[int, List[int]]] = []
        self.hash_table: Optional[HashTable] = None
        self.recommender: Optional[RecommenderSystem] = None
        self.selected_strategy = "Separate Chaining"
        self.experiment_mode = "Reproducible Experiment"

        # Beacon & Animation state
        self._beacon_state = False
        self._hover_active = False

        # Create UI layout
        self.setup_ui()

        # Start background pulsing beacon for live terminal
        self._pulse_beacon()

        # Initial welcome display
        self.write_console(
            "======================================================================\n"
            "   HASH TABLE RECOMMENDER SIMULATOR — SYSTEM INITIALIZED\n"
            "======================================================================\n"
            "Welcome! Select a dataset size and click 'Generate Dataset' to start.\n"
            "Hover over any control button to preview execution output details."
        )

    # ==========================================
    # UI SETUP & ARCHITECTURE
    # ==========================================
    def setup_ui(self):
        self.grid_columnconfigure(0, weight=0, minsize=320)  # Left sidebar
        self.grid_columnconfigure(1, weight=1)               # Right content panel
        self.grid_rowconfigure(0, weight=1)

        # ----------------------------------------------------
        # 1. SIDEBAR CONTROLS
        # ----------------------------------------------------
        self.sidebar_frame = ctk.CTkFrame(
            self,
            width=320,
            corner_radius=0,
            fg_color=COLOR_BG_SIDEBAR,
            border_color=COLOR_BORDER,
            border_width=1
        )
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        self.sidebar_frame.grid_propagate(False)

        # Header Title with Glowing Logo Badge
        self.header_frame = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        self.header_frame.pack(fill="x", padx=18, pady=(18, 12))

        self.logo_badge = ctk.CTkLabel(
            self.header_frame,
            text="⬡",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color=COLOR_ACCENT_CYAN
        )
        self.logo_badge.pack(side="left", padx=(0, 8))

        self.app_title_box = ctk.CTkFrame(self.header_frame, fg_color="transparent")
        self.app_title_box.pack(side="left", fill="x", expand=True)

        self.app_title = ctk.CTkLabel(
            self.app_title_box,
            text="HASH SIMULATOR",
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color=COLOR_TEXT_PRIMARY
        )
        self.app_title.pack(anchor="w")

        self.app_subtitle = ctk.CTkLabel(
            self.app_title_box,
            text="Interactive Recommendation Engine",
            font=ctk.CTkFont(size=10),
            text_color=COLOR_TEXT_SECONDARY
        )
        self.app_subtitle.pack(anchor="w")

        # Separator line
        self.sep_head = ctk.CTkFrame(self.sidebar_frame, height=1, fg_color=COLOR_BORDER)
        self.sep_head.pack(fill="x", padx=16, pady=(0, 10))

        # Mode Selection Card
        self.mode_card = ctk.CTkFrame(self.sidebar_frame, fg_color=COLOR_BG_CARD, corner_radius=10, border_color=COLOR_BORDER, border_width=1)
        self.mode_card.pack(fill="x", padx=16, pady=4)

        self.mode_label = ctk.CTkLabel(
            self.mode_card,
            text="Experiment Mode",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=COLOR_TEXT_PRIMARY
        )
        self.mode_label.pack(padx=12, pady=(8, 4), anchor="w")

        self.mode_segmented = ctk.CTkSegmentedButton(
            self.mode_card,
            values=["Reproducible Experiment", "Fresh Random Dataset"],
            command=self.on_mode_change,
            selected_color=COLOR_ACCENT_PRIMARY,
            selected_hover_color=COLOR_ACCENT_HOVER,
            unselected_color=COLOR_BG_INPUT,
            font=ctk.CTkFont(size=10, weight="bold")
        )
        self.mode_segmented.set("Reproducible Experiment")
        self.mode_segmented.pack(padx=10, pady=(0, 10), fill="x")

        # Parameters Container (Accordion / Config Card)
        self.config_card = ctk.CTkFrame(self.sidebar_frame, fg_color=COLOR_BG_CARD, corner_radius=10, border_color=COLOR_BORDER, border_width=1)
        self.config_card.pack(fill="x", padx=16, pady=6)

        # 1. Dataset Size
        self.ds_label = ctk.CTkLabel(self.config_card, text="Dataset Size (Users):", font=ctk.CTkFont(size=11, weight="bold"), text_color=COLOR_TEXT_SECONDARY)
        self.ds_label.pack(padx=12, pady=(8, 2), anchor="w")
        self.ds_option = ctk.CTkOptionMenu(
            self.config_card,
            values=["10", "50", "100", "500", "1000"],
            command=self.on_dataset_size_change,
            fg_color=COLOR_BG_CARD_ELEVATED,
            button_color=COLOR_ACCENT_PRIMARY,
            button_hover_color=COLOR_ACCENT_HOVER,
            dropdown_fg_color=COLOR_BG_CARD_ELEVATED
        )
        self.ds_option.set("100")
        self.ds_option.pack(padx=12, pady=(0, 8), fill="x")

        # 2. Collision Strategy
        self.strat_label = ctk.CTkLabel(self.config_card, text="Collision Strategy:", font=ctk.CTkFont(size=11, weight="bold"), text_color=COLOR_TEXT_SECONDARY)
        self.strat_label.pack(padx=12, pady=(2, 2), anchor="w")
        self.strat_segmented = ctk.CTkSegmentedButton(
            self.config_card,
            values=["Separate Chaining", "Linear Probing"],
            command=self.on_strategy_change,
            selected_color=COLOR_ACCENT_PRIMARY,
            selected_hover_color=COLOR_ACCENT_HOVER,
            unselected_color=COLOR_BG_INPUT,
            font=ctk.CTkFont(size=10, weight="bold")
        )
        self.strat_segmented.set("Separate Chaining")
        self.strat_segmented.pack(padx=12, pady=(0, 8), fill="x")

        # 3. Table Size Mode & Target Load Factor (Compact 2-col)
        self.grid_params = ctk.CTkFrame(self.config_card, fg_color="transparent")
        self.grid_params.pack(fill="x", padx=12, pady=(0, 10))
        self.grid_params.grid_columnconfigure(0, weight=1)
        self.grid_params.grid_columnconfigure(1, weight=1)

        self.ts_label = ctk.CTkLabel(self.grid_params, text="Table Size:", font=ctk.CTkFont(size=11, weight="bold"), text_color=COLOR_TEXT_SECONDARY)
        self.ts_label.grid(row=0, column=0, sticky="w", pady=(0, 2))
        self.ts_mode = ctk.CTkSegmentedButton(
            self.grid_params,
            values=["Auto", "Custom"],
            command=self.on_table_size_mode_change,
            selected_color=COLOR_ACCENT_PRIMARY,
            selected_hover_color=COLOR_ACCENT_HOVER,
            unselected_color=COLOR_BG_INPUT,
            font=ctk.CTkFont(size=9, weight="bold")
        )
        self.ts_mode.set("Auto")
        self.ts_mode.grid(row=1, column=0, sticky="ew", padx=(0, 4))

        self.lf_label = ctk.CTkLabel(self.grid_params, text="Target LF:", font=ctk.CTkFont(size=11, weight="bold"), text_color=COLOR_TEXT_SECONDARY)
        self.lf_label.grid(row=0, column=1, sticky="w", pady=(0, 2), padx=(4, 0))
        self.lf_segmented = ctk.CTkSegmentedButton(
            self.grid_params,
            values=["0.25", "0.50", "0.75", "0.90"],
            command=self.on_load_factor_change,
            selected_color=COLOR_ACCENT_PRIMARY,
            selected_hover_color=COLOR_ACCENT_HOVER,
            unselected_color=COLOR_BG_INPUT,
            font=ctk.CTkFont(size=9, weight="bold")
        )
        self.lf_segmented.set("0.75")
        self.lf_segmented.grid(row=1, column=1, sticky="ew", padx=(4, 0))

        # Custom Table Size Entry (initially hidden)
        self.custom_ts_entry = ctk.CTkEntry(
            self.config_card,
            placeholder_text="Enter Custom Table Size",
            fg_color=COLOR_BG_INPUT,
            border_color=COLOR_BORDER
        )
        self.custom_ts_entry.insert(0, "150")

        # Action Buttons Scroll Frame
        self.btn_scroll = ctk.CTkScrollableFrame(
            self.sidebar_frame,
            fg_color="transparent",
            scrollbar_button_color=COLOR_BORDER,
            scrollbar_button_hover_color=COLOR_BORDER_HOVER
        )
        self.btn_scroll.pack(fill="both", expand=True, padx=12, pady=(4, 8))

        # Benchmark Advanced Options Card
        self.bench_expander = ctk.CTkFrame(self.btn_scroll, fg_color=COLOR_BG_CARD, corner_radius=8, border_color=COLOR_BORDER, border_width=1)
        self.bench_expander.pack(fill="x", padx=4, pady=(2, 8))

        self.sizes_lbl = ctk.CTkLabel(self.bench_expander, text="Benchmark Sizes (csv):", font=ctk.CTkFont(size=10, weight="bold"), text_color=COLOR_TEXT_SECONDARY)
        self.sizes_lbl.pack(anchor="w", padx=8, pady=(4, 0))
        self.sizes_entry = ctk.CTkEntry(self.bench_expander, font=ctk.CTkFont(size=10), height=24, fg_color=COLOR_BG_INPUT, border_color=COLOR_BORDER)
        self.sizes_entry.insert(0, "10, 50, 100, 500, 1000")
        self.sizes_entry.pack(fill="x", padx=8, pady=(1, 4))

        self.lfs_lbl = ctk.CTkLabel(self.bench_expander, text="Collision Load Factors (csv):", font=ctk.CTkFont(size=10, weight="bold"), text_color=COLOR_TEXT_SECONDARY)
        self.lfs_lbl.pack(anchor="w", padx=8, pady=(2, 0))
        self.lfs_entry = ctk.CTkEntry(self.bench_expander, font=ctk.CTkFont(size=10), height=24, fg_color=COLOR_BG_INPUT, border_color=COLOR_BORDER)
        self.lfs_entry.insert(0, "0.25, 0.50, 0.75, 0.90")
        self.lfs_entry.pack(fill="x", padx=8, pady=(1, 4))

        self.numpy_size_lbl = ctk.CTkLabel(self.bench_expander, text="NumPy Benchmark Size:", font=ctk.CTkFont(size=10, weight="bold"), text_color=COLOR_TEXT_SECONDARY)
        self.numpy_size_lbl.pack(anchor="w", padx=8, pady=(2, 0))
        self.numpy_size_entry = ctk.CTkEntry(self.bench_expander, font=ctk.CTkFont(size=10), height=24, fg_color=COLOR_BG_INPUT, border_color=COLOR_BORDER)
        self.numpy_size_entry.insert(0, "5000")
        self.numpy_size_entry.pack(fill="x", padx=8, pady=(1, 4))

        self.bench_strat_lbl = ctk.CTkLabel(self.bench_expander, text="Benchmark Mode:", font=ctk.CTkFont(size=10, weight="bold"), text_color=COLOR_TEXT_SECONDARY)
        self.bench_strat_lbl.pack(anchor="w", padx=8, pady=(2, 0))
        self.bench_strat_menu = ctk.CTkOptionMenu(
            self.bench_expander,
            values=["Separate Chaining", "Linear Probing", "Both"],
            font=ctk.CTkFont(size=10),
            height=24,
            fg_color=COLOR_BG_CARD_ELEVATED,
            button_color=COLOR_ACCENT_PRIMARY
        )
        self.bench_strat_menu.set("Both")
        self.bench_strat_menu.pack(fill="x", padx=8, pady=(1, 6))

        # Action Buttons with Animated Hover & Live Output Section Connection
        self.btn_gen_ds = AnimatedActionButton(
            self.btn_scroll,
            action_id="gen_dataset",
            title="Generate Dataset",
            description="Synthesizes user interaction profiles using reproducible or dynamic seeds.",
            hover_handler=self.on_action_hover_enter,
            leave_handler=self.on_action_hover_leave,
            accent_color=COLOR_ACCENT_CYAN,
            command=self.click_generate_dataset
        )
        self.btn_gen_ds.pack(pady=3, fill="x", padx=4)

        self.btn_insert = AnimatedActionButton(
            self.btn_scroll,
            action_id="insert_table",
            title="Insert Into Hash Table",
            description="Hashes records, builds buckets, resolves collisions, and runs animated insertion trace.",
            hover_handler=self.on_action_hover_enter,
            leave_handler=self.on_action_hover_leave,
            accent_color="#38BDF8",
            command=self.click_insert_table
        )
        self.btn_insert.pack(pady=3, fill="x", padx=4)

        self.btn_lookup = AnimatedActionButton(
            self.btn_scroll,
            action_id="lookup_bench",
            title="Run Lookup Benchmark",
            description="Measures O(1) direct memory retrieval times across scaling dataset sizes.",
            hover_handler=self.on_action_hover_enter,
            leave_handler=self.on_action_hover_leave,
            accent_color=COLOR_ACCENT_PRIMARY,
            command=self.click_run_lookup_benchmark
        )
        self.btn_lookup.pack(pady=3, fill="x", padx=4)

        self.btn_collision = AnimatedActionButton(
            self.btn_scroll,
            action_id="collision_test",
            title="Run Collision Test",
            description="Evaluates collision frequencies and probe chain lengths across varied load factors.",
            hover_handler=self.on_action_hover_enter,
            leave_handler=self.on_action_hover_leave,
            accent_color=COLOR_WARNING,
            command=self.click_run_collision_test
        )
        self.btn_collision.pack(pady=3, fill="x", padx=4)

        self.btn_numpy = AnimatedActionButton(
            self.btn_scroll,
            action_id="compare_numpy",
            title="Compare NumPy",
            description="Compares optimized vectorized C-routines against standard Python list counting.",
            hover_handler=self.on_action_hover_enter,
            leave_handler=self.on_action_hover_leave,
            accent_color="#A855F7",
            command=self.click_compare_numpy
        )
        self.btn_numpy.pack(pady=3, fill="x", padx=4)

        # Removed 'Week 6' beside Cosine Similarity as requested
        self.btn_cosine = AnimatedActionButton(
            self.btn_scroll,
            action_id="cosine_recs",
            title="Cosine Similarity Recs",
            description="Computes user preference vector dot-products and predicts personalized recommendations.",
            hover_handler=self.on_action_hover_enter,
            leave_handler=self.on_action_hover_leave,
            accent_color="#00E5FF",
            fg_color="#1E293B",
            hover_accent="#0284C7",
            command=self.click_run_week6
        )
        self.btn_cosine.pack(pady=3, fill="x", padx=4)
        self.btn_week6 = self.btn_cosine  # Backwards compatibility alias

        self.btn_visualize = AnimatedActionButton(
            self.btn_scroll,
            action_id="show_table",
            title="Show Hash Table",
            description="Renders interactive memory buckets, collision chains, and user item counts.",
            hover_handler=self.on_action_hover_enter,
            leave_handler=self.on_action_hover_leave,
            accent_color=COLOR_SUCCESS,
            command=self.click_show_hash_table
        )
        self.btn_visualize.pack(pady=3, fill="x", padx=4)

        self.btn_graphs = AnimatedActionButton(
            self.btn_scroll,
            action_id="gen_graphs",
            title="Generate Graphs",
            description="Renders high-resolution matplotlib charts for lookup speed and collision curves.",
            hover_handler=self.on_action_hover_enter,
            leave_handler=self.on_action_hover_leave,
            accent_color="#F43F5E",
            command=self.click_generate_graphs
        )
        self.btn_graphs.pack(pady=3, fill="x", padx=4)

        self.btn_how_it_works = AnimatedActionButton(
            self.btn_scroll,
            action_id="edu_guide",
            title="How It Works (Guide)",
            description="Opens the visual educational walkthrough of hash functions, collisions, and load factors.",
            hover_handler=self.on_action_hover_enter,
            leave_handler=self.on_action_hover_leave,
            accent_color="#10B981",
            fg_color="#064E3B",
            hover_accent="#059669",
            command=self.click_how_it_works
        )
        self.btn_how_it_works.pack(pady=3, fill="x", padx=4)

        self.btn_reset = AnimatedActionButton(
            self.btn_scroll,
            action_id="reset_sim",
            title="Reset Simulator",
            description="Clears memory tables, resets statistical counters, and restores clean slate.",
            hover_handler=self.on_action_hover_enter,
            leave_handler=self.on_action_hover_leave,
            accent_color=COLOR_DANGER,
            fg_color="#7F1D1D",
            hover_accent="#DC2626",
            command=self.click_reset
        )
        self.btn_reset.pack(pady=(10, 4), fill="x", padx=4)

        # ----------------------------------------------------
        # 2. RIGHT MAIN CONTENT AREA
        # ----------------------------------------------------
        self.right_frame = ctk.CTkFrame(self, fg_color=COLOR_BG_MAIN, corner_radius=0)
        self.right_frame.grid(row=0, column=1, sticky="nsew", padx=0, pady=0)

        self.right_frame.grid_columnconfigure(0, weight=1)
        self.right_frame.grid_rowconfigure(0, weight=10) # Visualizer / Charts
        self.right_frame.grid_rowconfigure(1, weight=4)  # Step Trace & Stats
        self.right_frame.grid_rowconfigure(2, weight=6)  # Output Console Section

        # --- ROW 0: TABVIEW (VISUALIZER & PERFORMANCE CHARTS) ---
        self.content_tabview = ctk.CTkTabview(
            self.right_frame,
            fg_color=COLOR_BG_CARD,
            segmented_button_selected_color=COLOR_ACCENT_PRIMARY,
            segmented_button_selected_hover_color=COLOR_ACCENT_HOVER,
            segmented_button_unselected_color=COLOR_BG_INPUT,
            border_color=COLOR_BORDER,
            border_width=1,
            corner_radius=12
        )
        self.content_tabview.grid(row=0, column=0, sticky="nsew", padx=16, pady=(14, 6))

        self.tab_visualizer = self.content_tabview.add("Hash Table Visualizer")
        self.tab_graphs = self.content_tabview.add("Performance Charts")

        # Visualizer Tab Sub-layout
        self.tab_visualizer.grid_columnconfigure(0, weight=1)
        self.tab_visualizer.grid_rowconfigure(0, weight=1)
        self.tab_visualizer.grid_rowconfigure(1, weight=0)

        self.visualizer_scroll = ctk.CTkScrollableFrame(
            self.tab_visualizer,
            fg_color=COLOR_BG_MAIN,
            corner_radius=8,
            border_color=COLOR_BORDER,
            border_width=1
        )
        self.visualizer_scroll.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)

        # Search / Interactive Lookup Bar
        self.search_control_frame = ctk.CTkFrame(self.tab_visualizer, height=48, fg_color=COLOR_BG_CARD_ELEVATED, corner_radius=8)
        self.search_control_frame.grid(row=1, column=0, sticky="ew", padx=6, pady=(0, 6))

        self.search_lbl = ctk.CTkLabel(
            self.search_control_frame,
            text="Lookup User ID:",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=COLOR_TEXT_PRIMARY
        )
        self.search_lbl.pack(side="left", padx=(14, 8))

        self.search_entry = ctk.CTkEntry(
            self.search_control_frame,
            width=160,
            placeholder_text="e.g. 1005",
            fg_color=COLOR_BG_INPUT,
            border_color=COLOR_BORDER
        )
        self.search_entry.pack(side="left", padx=6)

        self.btn_search = ctk.CTkButton(
            self.search_control_frame,
            text="Search & Animate",
            command=self.click_search_animate,
            fg_color=COLOR_ACCENT_PRIMARY,
            hover_color=COLOR_ACCENT_HOVER,
            width=140,
            corner_radius=6,
            font=ctk.CTkFont(weight="bold")
        )
        self.btn_search.pack(side="left", padx=10)

        # Performance Charts Sub-layout
        self.tab_graphs.grid_columnconfigure(0, weight=1)
        self.tab_graphs.grid_columnconfigure(1, weight=1)
        self.tab_graphs.grid_columnconfigure(2, weight=1)
        self.tab_graphs.grid_rowconfigure(0, weight=1)

        self.graph_frame_1 = ctk.CTkFrame(self.tab_graphs, fg_color=COLOR_BG_MAIN, corner_radius=8, border_color=COLOR_BORDER, border_width=1)
        self.graph_frame_1.grid(row=0, column=0, sticky="nsew", padx=4, pady=6)
        self.graph_lbl_1 = ctk.CTkLabel(self.graph_frame_1, text="Lookup Performance Chart\n(Click 'Generate Graphs' to render)", font=ctk.CTkFont(size=11, slant="italic"), text_color=COLOR_TEXT_MUTED)
        self.graph_lbl_1.pack(expand=True)

        self.graph_frame_2 = ctk.CTkFrame(self.tab_graphs, fg_color=COLOR_BG_MAIN, corner_radius=8, border_color=COLOR_BORDER, border_width=1)
        self.graph_frame_2.grid(row=0, column=1, sticky="nsew", padx=4, pady=6)
        self.graph_lbl_2 = ctk.CTkLabel(self.graph_frame_2, text="Collision Rate Chart\n(Click 'Generate Graphs' to render)", font=ctk.CTkFont(size=11, slant="italic"), text_color=COLOR_TEXT_MUTED)
        self.graph_lbl_2.pack(expand=True)

        self.graph_frame_3 = ctk.CTkFrame(self.tab_graphs, fg_color=COLOR_BG_MAIN, corner_radius=8, border_color=COLOR_BORDER, border_width=1)
        self.graph_frame_3.grid(row=0, column=2, sticky="nsew", padx=4, pady=6)
        self.graph_lbl_3 = ctk.CTkLabel(self.graph_frame_3, text="Load Factor vs Lookup Chart\n(Click 'Generate Graphs' to render)", font=ctk.CTkFont(size=11, slant="italic"), text_color=COLOR_TEXT_MUTED)
        self.graph_lbl_3.pack(expand=True)

        # --- ROW 1: MIDDLE PANEL (STEP TRACE & STATISTICS CARDS) ---
        self.middle_frame = ctk.CTkFrame(self.right_frame, height=135, fg_color="transparent")
        self.middle_frame.grid(row=1, column=0, sticky="nsew", padx=16, pady=4)
        self.middle_frame.grid_columnconfigure(0, weight=3) # Step trace
        self.middle_frame.grid_columnconfigure(1, weight=2) # Stats cards
        self.middle_frame.grid_rowconfigure(0, weight=1)

        # 1. Step-by-Step Animation Panel
        self.anim_panel = ctk.CTkFrame(self.middle_frame, fg_color=COLOR_BG_CARD, corner_radius=10, border_color=COLOR_BORDER, border_width=1)
        self.anim_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 6), pady=0)

        # Trace Header & Stepper Badges
        self.anim_head_frame = ctk.CTkFrame(self.anim_panel, fg_color="transparent")
        self.anim_head_frame.pack(fill="x", padx=12, pady=(8, 2))

        self.anim_title = ctk.CTkLabel(
            self.anim_head_frame,
            text="⚡ Operation Step Trace",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=COLOR_ACCENT_CYAN
        )
        self.anim_title.pack(side="left")

        # 4-stage pipeline stepper badges
        self.stepper_frame = ctk.CTkFrame(self.anim_head_frame, fg_color="transparent")
        self.stepper_frame.pack(side="right")
        self.stepper_badges = {}
        for s_idx, s_name in enumerate(["1. Input", "2. Hash", "3. Traverse", "4. Done"]):
            lbl = ctk.CTkLabel(
                self.stepper_frame,
                text=s_name,
                font=ctk.CTkFont(size=9, weight="bold"),
                fg_color=COLOR_BG_INPUT,
                text_color=COLOR_TEXT_MUTED,
                corner_radius=4,
                padx=6,
                pady=1
            )
            lbl.pack(side="left", padx=2)
            self.stepper_badges[s_idx] = lbl

        self.anim_text = ctk.CTkLabel(
            self.anim_panel,
            text="No ongoing operation.\nUse search or insert to trace step-by-step hashing execution.",
            justify="left",
            font=ctk.CTkFont(family="Consolas", size=11),
            text_color=COLOR_TEXT_SECONDARY
        )
        self.anim_text.pack(fill="both", expand=True, padx=14, pady=(2, 8))

        # 2. Statistics Grid Cards
        self.stats_panel = ctk.CTkFrame(self.middle_frame, fg_color=COLOR_BG_CARD, corner_radius=10, border_color=COLOR_BORDER, border_width=1)
        self.stats_panel.grid(row=0, column=1, sticky="nsew", padx=(6, 0), pady=0)

        self.stats_title = ctk.CTkLabel(
            self.stats_panel,
            text="Table Metrics",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=COLOR_ACCENT_PRIMARY
        )
        self.stats_title.pack(anchor="w", padx=12, pady=(6, 2))

        self.stats_grid = ctk.CTkFrame(self.stats_panel, fg_color="transparent")
        self.stats_grid.pack(fill="both", expand=True, padx=8, pady=(0, 6))
        self.stats_grid.grid_columnconfigure((0, 1, 2), weight=1)
        self.stats_grid.grid_rowconfigure((0, 1), weight=1)

        self.stats_labels = {}
        self.stats_cards = {}
        fields = [
            ("Users", "0", "[N]"),
            ("Table Size", "0", "[M]"),
            ("Collisions", "0", "[!]"),
            ("Load Factor", "0.00", "[LF]"),
            ("Avg Bucket", "0.00", "[Avg]"),
            ("Max Bucket", "0", "[Max]")
        ]

        for idx, (title, default_val, icon) in enumerate(fields):
            r = idx // 3
            c = idx % 3
            card = ctk.CTkFrame(self.stats_grid, fg_color=COLOR_BG_CARD_ELEVATED, corner_radius=6, border_color=COLOR_BORDER, border_width=1)
            card.grid(row=r, column=c, padx=3, pady=2, sticky="nsew")

            t_lbl = ctk.CTkLabel(card, text=f"{icon} {title}", font=ctk.CTkFont(size=9, weight="bold"), text_color=COLOR_TEXT_SECONDARY)
            t_lbl.pack(anchor="w", padx=6, pady=(3, 0))

            v_lbl = ctk.CTkLabel(card, text=default_val, font=ctk.CTkFont(size=12, weight="bold"), text_color=COLOR_TEXT_PRIMARY)
            v_lbl.pack(anchor="w", padx=6, pady=(0, 3))

            self.stats_labels[title + ":"] = v_lbl
            self.stats_cards[title] = card

        # --- ROW 2: OUTPUT CONSOLE WITH ANIMATED HOVER ACTION BANNER ---
        self.console_frame = ctk.CTkFrame(
            self.right_frame,
            fg_color=COLOR_BG_CARD,
            corner_radius=12,
            border_color=COLOR_BORDER,
            border_width=1.5
        )
        self.console_frame.grid(row=2, column=0, sticky="nsew", padx=16, pady=(6, 14))

        # Console Header Bar
        self.console_topbar = ctk.CTkFrame(self.console_frame, fg_color="transparent", height=32)
        self.console_topbar.pack(fill="x", padx=14, pady=(8, 4))

        # Beacon dot & Title
        self.beacon_dot = ctk.CTkLabel(
            self.console_topbar,
            text="●",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=COLOR_SUCCESS
        )
        self.beacon_dot.pack(side="left", padx=(0, 6))

        self.console_title = ctk.CTkLabel(
            self.console_topbar,
            text="TERMINAL LOG",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=COLOR_TEXT_PRIMARY
        )
        self.console_title.pack(side="left")

        # Quick Control Buttons on Header (Copy / Clear)
        self.btn_clear_log = ctk.CTkButton(
            self.console_topbar,
            text="Clear",
            font=ctk.CTkFont(size=10, weight="bold"),
            width=50,
            height=22,
            fg_color=COLOR_BG_CARD_ELEVATED,
            hover_color="#334155",
            corner_radius=4,
            command=self._clear_console
        )
        self.btn_clear_log.pack(side="right", padx=(4, 0))

        self.btn_copy_log = ctk.CTkButton(
            self.console_topbar,
            text="Copy Log",
            font=ctk.CTkFont(size=10, weight="bold"),
            width=65,
            height=22,
            fg_color=COLOR_BG_CARD_ELEVATED,
            hover_color="#334155",
            corner_radius=4,
            command=self._copy_console
        )
        self.btn_copy_log.pack(side="right", padx=(4, 4))

        # ANIMATED ACTION PREVIEW BANNER (Animates whenever a button is hovered)
        self.action_preview_banner = ctk.CTkFrame(
            self.console_frame,
            fg_color=COLOR_BG_CARD_ELEVATED,
            corner_radius=6,
            border_color=COLOR_BORDER,
            border_width=1,
            height=28
        )
        self.action_preview_banner.pack(fill="x", padx=14, pady=(0, 6))

        self.preview_badge = ctk.CTkLabel(
            self.action_preview_banner,
            text="● READY",
            font=ctk.CTkFont(size=9, weight="bold"),
            text_color=COLOR_SUCCESS,
            padx=8
        )
        self.preview_badge.pack(side="left", padx=(6, 4), pady=2)

        self.preview_text = ctk.CTkLabel(
            self.action_preview_banner,
            text="Hover over any action button to preview its execution details and output impact.",
            font=ctk.CTkFont(size=10),
            text_color=COLOR_TEXT_SECONDARY
        )
        self.preview_text.pack(side="left", padx=4, pady=2)

        # Rich Monospace Console Textbox
        self.console_textbox = ctk.CTkTextbox(
            self.console_frame,
            state="disabled",
            font=ctk.CTkFont(family="Consolas", size=11),
            fg_color=COLOR_BG_ROOT,
            border_color=COLOR_BORDER,
            border_width=1,
            corner_radius=8
        )
        self.console_textbox.pack(fill="both", expand=True, padx=14, pady=(0, 10))

        # Setup Rich Color Tags inside Tkinter Text widget
        self._setup_console_tags()

    def _setup_console_tags(self):
        try:
            tb = self.console_textbox._textbox
            tb.tag_config("cyan", foreground=COLOR_ACCENT_CYAN)
            tb.tag_config("indigo", foreground="#818CF8")
            tb.tag_config("emerald", foreground=COLOR_SUCCESS)
            tb.tag_config("amber", foreground=COLOR_WARNING)
            tb.tag_config("rose", foreground=COLOR_DANGER)
            tb.tag_config("dim", foreground=COLOR_TEXT_MUTED)
            tb.tag_config("bold", font=ctk.CTkFont(family="Consolas", size=11, weight="bold"))
        except Exception:
            pass

    # ==========================================
    # ANIMATIONS & HOVER INTERACTION
    # ==========================================
    def _pulse_beacon(self):
        """Pulsing animation for the terminal active status indicator."""
        self._beacon_state = not self._beacon_state
        if not self._hover_active:
            glow_color = COLOR_SUCCESS if self._beacon_state else "#059669"
            self.beacon_dot.configure(text_color=glow_color)
        self.after(750, self._pulse_beacon)

    def on_action_hover_enter(self, title: str, description: str, accent_color: str):
        """
        Triggered when cursor hovers over any action button:
        - Animates the Output Section border with a colored glow
        - Lights up the Action Preview Banner with contextual explanation
        """
        self._hover_active = True
        self.console_frame.configure(border_color=accent_color, border_width=2)
        self.preview_badge.configure(
            text=f"⚡ PREVIEW: {title.upper()}",
            text_color=accent_color
        )
        self.preview_text.configure(
            text=description,
            text_color=COLOR_TEXT_PRIMARY
        )
        self.beacon_dot.configure(text_color=accent_color)

    def on_action_hover_leave(self):
        """Restores output section styling when hover ends."""
        self._hover_active = False
        self.console_frame.configure(border_color=COLOR_BORDER, border_width=1.5)
        self.preview_badge.configure(
            text="● READY",
            text_color=COLOR_SUCCESS
        )
        self.preview_text.configure(
            text="Hover over any action button to preview its execution details and output impact.",
            text_color=COLOR_TEXT_SECONDARY
        )
        self.beacon_dot.configure(text_color=COLOR_SUCCESS)

    def _set_stepper_active(self, step_idx: int):
        """Animates the 4-step pipeline breadcrumb badges."""
        for idx, lbl in self.stepper_badges.items():
            if idx < step_idx:
                lbl.configure(fg_color="#064E3B", text_color=COLOR_SUCCESS) # Completed
            elif idx == step_idx:
                lbl.configure(fg_color=COLOR_ACCENT_PRIMARY, text_color=COLOR_TEXT_PRIMARY) # Active
            else:
                lbl.configure(fg_color=COLOR_BG_INPUT, text_color=COLOR_TEXT_MUTED)

    def _copy_console(self):
        """Copies console text to clipboard with temporary button animation."""
        content = self.console_textbox.get("1.0", "end-1c")
        self.clipboard_clear()
        self.clipboard_append(content)
        self.btn_copy_log.configure(text="✓ Copied!", fg_color=COLOR_SUCCESS)
        self.after(1600, lambda: self.btn_copy_log.configure(text="Copy Log", fg_color=COLOR_BG_CARD_ELEVATED))

    def _clear_console(self):
        """Clears console text."""
        self.write_console("[Console Log Cleared]", clear=True)

    # ==========================================
    # LOGGING & CONSOLE OUTPUT
    # ==========================================
    def write_console(self, text: str, clear: bool = False):
        self.console_textbox.configure(state="normal")
        if clear:
            self.console_textbox.delete("1.0", "end")
        
        # Colorize specific keywords using Tkinter text tags
        start_index = self.console_textbox.index("end-1c")
        self.console_textbox.insert("end", text + "\n")
        end_index = self.console_textbox.index("end-1c")

        try:
            tb = self.console_textbox._textbox
            # Format headers
            if "===" in text or "---" in text:
                tb.tag_add("dim", start_index, end_index)
            elif "Success" in text or "COMPLETED" in text or "OK" in text:
                tb.tag_add("emerald", start_index, end_index)
            elif "FAILURE" in text or "Error" in text:
                tb.tag_add("rose", start_index, end_index)
            elif "MODE A" in text or "MODE B" in text or "SIMULATOR" in text:
                tb.tag_add("cyan", start_index, end_index)
        except Exception:
            pass

        self.console_textbox.see("end")
        self.console_textbox.configure(state="disabled")

    def update_stats_ui(self):
        if not self.hash_table:
            return
        stats = self.hash_table.get_collision_statistics()
        self.stats_labels["Users:"].configure(text=str(stats["users"]))
        self.stats_labels["Table Size:"].configure(text=str(stats["table_size"]))
        self.stats_labels["Collisions:"].configure(
            text=str(stats["collisions"]),
            text_color=COLOR_WARNING if stats["collisions"] > 0 else COLOR_TEXT_PRIMARY
        )
        lf = stats['load_factor']
        self.stats_labels["Load Factor:"].configure(
            text=f"{lf:.2f}",
            text_color=COLOR_DANGER if lf > 0.85 else (COLOR_WARNING if lf > 0.70 else COLOR_SUCCESS)
        )
        if hasattr(self.hash_table, "TOMBSTONE"):
            self.stats_labels["Avg Bucket:"].configure(text="N/A")
            self.stats_labels["Max Bucket:"].configure(text="N/A")
        else:
            self.stats_labels["Avg Bucket:"].configure(text=f"{stats['avg_bucket_length']:.2f}")
            self.stats_labels["Max Bucket:"].configure(text=str(stats["max_bucket_length"]))

        # Pulse highlight on stats cards
        for card in self.stats_cards.values():
            card.configure(border_color=COLOR_ACCENT_CYAN)
        self.after(450, lambda: [card.configure(border_color=COLOR_BORDER) for card in self.stats_cards.values()])

    # ==========================================
    # UI CALLBACKS
    # ==========================================
    def on_mode_change(self, choice):
        self.experiment_mode = choice
        if choice == "Reproducible Experiment":
            self.write_console("\n[MODE A: Reproducible Experiment] Fixed seed (42) enabled for standard academic comparison.")
        else:
            self.write_console("\n[MODE B: Fresh Random Dataset] Dynamic seed enabled. Each click on 'Generate Dataset' will produce a new dataset.")

    def on_strategy_change(self, choice):
        self.selected_strategy = choice
        self.write_console(f"Collision strategy changed to {choice}.")
        if self.dataset:
            self.click_insert_table()

    def on_dataset_size_change(self, choice):
        self.dataset_size = int(choice)
        self.write_console(f"Target Dataset Size configured to {self.dataset_size}.")

    def on_table_size_mode_change(self, mode):
        self.table_size_mode = mode
        self.write_console(f"Table Size selection mode changed to {mode}.")
        if mode == "Custom":
            self.custom_ts_entry.pack(padx=12, pady=(0, 8), fill="x")
        else:
            self.custom_ts_entry.pack_forget()

    def on_load_factor_change(self, choice):
        self.selected_load_factor = float(choice)
        self.write_console(f"Target Load Factor configured to {self.selected_load_factor}.")

    # ==========================================
    # SIMULATOR ACTIONS & BENCHMARKS
    # ==========================================
    def click_generate_dataset(self):
        seed = 42 if self.experiment_mode == "Reproducible Experiment" else None
        self.dataset = generate_user_data(self.dataset_size, seed=seed)
        formatted_txt = format_first_n_users(self.dataset, 10)
        mode_str = "MODE A: Reproducible Experiment (Seed 42)" if seed == 42 else "MODE B: Fresh Random Dataset (Dynamic Seed)"
        self.write_console(f"\n" + "="*55 + f"\nDataset Generated Successfully! [{mode_str}]\n" + "="*55, clear=True)
        self.write_console(formatted_txt)

    def click_insert_table(self):
        if not self.dataset:
            messagebox.showwarning("No Dataset", "Please generate a dataset first before inserting into the Hash Table.")
            return

        # Determine table size
        if self.table_size_mode == "Auto":
            size = int(len(self.dataset) / self.selected_load_factor)
            if size % 2 == 0:
                size += 1
            self.write_console(f"\nAuto calculated Table Size based on Load Factor target ({self.selected_load_factor}): {size}")
        else:
            try:
                size = int(self.custom_ts_entry.get())
                if size <= 0:
                    raise ValueError
            except ValueError:
                messagebox.showerror("Invalid Size", "Please enter a valid positive integer for custom table size.")
                return

        strat_class = HashTableChaining if self.selected_strategy == "Separate Chaining" else HashTableLinearProbing
        self.hash_table = strat_class(size)
        self.recommender = RecommenderSystem(size, strategy_class=strat_class)

        self.write_console(f"Inserting {len(self.dataset)} records into Hash Table...")
        self.anim_title.configure(text="⚡ Insertion Animation & Trace", text_color=COLOR_ACCENT_CYAN)

        first_user = self.dataset[0]
        trace = self.hash_table.insert(first_user[0], first_user[1])

        def run_insertion_trace(step_idx=0):
            if step_idx < len(trace):
                step = trace[step_idx]
                if step["step"] == "input":
                    self._set_stepper_active(0)
                    self.anim_text.configure(text=f"Inserting Key (User ID): {step['val']}\nWaiting to process hash function...")
                elif step["step"] == "hash":
                    self._set_stepper_active(1)
                    self.anim_text.configure(text=f"Computing Hash Index:\nFormula: {step['formula']}\nTarget Index = {step['index']}")
                    self.highlight_visualizer_bucket(step["index"])
                elif step["step"] == "traverse":
                    self._set_stepper_active(2)
                    self.anim_text.configure(text=f"Traversing Chaining Index {step['index']}.\nExisting elements: {step['bucket']}\nOperation: Adding user to chaining list.")
                elif step["step"] == "probe":
                    self._set_stepper_active(2)
                    action_str = "Inserting here!" if step["action"] == "inserted" else "Probing next index..."
                    self.anim_text.configure(text=f"Probing Linear Index {step['index']}: slot is {step['state']}.\nProbe count: {step['probes']}\nAction: {action_str}")
                    self.highlight_visualizer_bucket(step["index"])
                elif step["step"] == "done":
                    self._set_stepper_active(3)
                    self.anim_text.configure(text=f"Successfully inserted User into index {step['index']}!")
                    self.highlight_visualizer_bucket(step["index"])
                self.after(700, lambda: run_insertion_trace(step_idx + 1))
            else:
                for user_id, movies in self.dataset[1:]:
                    self.hash_table.insert(user_id, movies)
                self.recommender.fit(self.dataset, movie_vocab=list(range(1, 101)))

                self.write_console(f"Successfully inserted all {len(self.dataset)} users into Hash Table.")
                self.update_stats_ui()
                self.click_show_hash_table()

        run_insertion_trace(0)

    def click_run_lookup_benchmark(self):
        sizes_str = self.sizes_entry.get().strip()
        try:
            sizes = [int(s.strip()) for s in sizes_str.split(",") if s.strip()]
            if not sizes:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid comma-separated list of integers for benchmark sizes.")
            return

        mode = self.bench_strat_menu.get()
        self.write_console(f"\nRunning Lookup Benchmark across varying dataset sizes ({', '.join(map(str, sizes))}) using Mode: {mode}...")
        self.update()

        results = PerformanceSimulator.run_lookup_benchmark(sizes, strategy=mode)

        for strat, res_list in results.items():
            self.write_console(f"\n--- Strategy: {strat} ---")
            self.write_console("-"*65)
            self.write_console(f"{'Dataset Size':<12} | {'Avg Lookup (sec)':<18} | {'Collisions':<12} | {'Memory (bytes)':<15}")
            self.write_console("-"*65)
            for size, t, col, mem in res_list:
                self.write_console(f"{size:<12} | {t:.9f} | {col:<12} | {mem:<15}")
            self.write_console("-"*65)

        self.write_console(
            "Observation:\n"
            "Even though dataset grows, lookup time remains almost constant.\n"
            "Reason: Hash tables directly compute the location instead of searching sequentially."
        )

    def click_run_collision_test(self):
        lfs_str = self.lfs_entry.get().strip()
        try:
            lfs = [float(f.strip()) for f in lfs_str.split(",") if f.strip()]
            if not lfs:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid comma-separated list of floats for load factors.")
            return

        mode = self.bench_strat_menu.get()
        self.write_console(f"\nRunning Collision Experiment across different Load Factors ({', '.join(map(str, lfs))}) using Mode: {mode}...")
        self.update()

        results = PerformanceSimulator.run_collision_experiment(lfs, strategy=mode)

        for strat, res_list in results.items():
            self.write_console(f"\n--- Strategy: {strat} ---")
            self.write_console("-"*85)
            if strat == "Linear Probing":
                self.write_console(f"{'Load Factor':<12} | {'Collision Rate':<16} | {'Avg Lookup (sec)':<18} | {'Memory (bytes)':<15} | {'Avg Search Probes':<15}")
                self.write_console("-"*85)
                for lf, rate, lookup_t, mem, extra in res_list:
                    self.write_console(f"{lf:<12.2f} | {rate*100:<15.1f}% | {lookup_t:.9f} | {mem:<15} | {extra.get('avg_search_probes', 0):.2f}")
            else:
                self.write_console(f"{'Load Factor':<12} | {'Collision Rate':<16} | {'Avg Lookup (sec)':<18} | {'Memory (bytes)':<15} | {'Avg Chain':<10} | {'Max Chain':<10}")
                self.write_console("-"*85)
                for lf, rate, lookup_t, mem, extra in res_list:
                    self.write_console(f"{lf:<12.2f} | {rate*100:<15.1f}% | {lookup_t:.9f} | {mem:<15} | {extra.get('avg_chain', 0):.2f} | {extra.get('max_chain', 0)}")
            self.write_console("-"*85)

        self.write_console(
            "Observation:\n"
            "Higher Load Factor -> Less Empty Space -> More Collisions -> Slower Search."
        )

    def click_compare_numpy(self):
        try:
            n_size = int(self.numpy_size_entry.get().strip())
            if n_size <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid positive integer for NumPy benchmark size.")
            return

        self.write_console(f"\nComparing Item Interaction Counting: Python List vs. NumPy (N={n_size})...")
        self.update()

        res = PerformanceSimulator.compare_numpy_performance(n_size)

        self.write_console("\n" + "-"*50)
        self.write_console("Implementation           | Avg Execution Time (sec)")
        self.write_console("-"*50)
        self.write_console(f"Python List Counting     | {res['python_list_time']:.9f}")
        self.write_console(f"NumPy bincount Vector    | {res['numpy_time']:.9f}")
        self.write_console(f"Hash Table Lookup (Ref)  | {res['hash_table_time']:.9f}")
        self.write_console("-"*50)

        self.write_console(
            f"Results Analysis:\n"
            f"NumPy Speedup vs List: {res['speedup_numpy']:.1f}x\n\n"
            f"Explanation:\n"
            f"- NumPy array manipulation uses highly optimized C routines (vectorized bincount).\n"
            f"- Python lists require traversing and updating elements one-by-one in python runtime memory."
        )

    def click_run_week6(self):
        """
        Executes Cosine Similarity Experiment vs Baseline (previously labeled Week 6).
        """
        is_reproducible = (self.experiment_mode == "Reproducible Experiment")
        mode_title = "MODE A — REPRODUCIBLE EXPERIMENT" if is_reproducible else "MODE B — FRESH DATA DEMO"

        if not self.dataset or not self.recommender:
            seed = 42 if is_reproducible else None
            self.write_console("\n" + "="*70)
            self.write_console(f"Running Cosine Similarity Recs: {mode_title} (Standalone N=500, M=100)...")
            self.write_console("="*70)
            self.update()
            res = PerformanceSimulator.run_week6_experiment(500, 100, top_n=5, seed=seed, mode_label=mode_title)
            self.write_console(f"\n--- {mode_title} Completed ---")
            self.write_console(f"Avg Baseline Time: {res['avg_baseline_time_sec']*1000:.4f} ms")
            self.write_console(f"Avg Cosine Time:   {res['avg_cosine_time_sec']*1000:.4f} ms")
            self.write_console(f"  * Hash Lookup:       {res['avg_lookup_time_sec']*1000:.4f} ms")
            self.write_console(f"  * Similarity Calc:   {res['avg_sim_calc_time_sec']*1000:.4f} ms")
            self.write_console(f"  * Rec Generation:    {res['avg_gen_time_sec']*1000:.4f} ms")
            self.write_console(f"Avg Top Similar User Score: {res['avg_max_similarity']:.4f}\n")
            return

        n_users = len(self.dataset)
        self.write_console("\n" + "="*70)
        self.write_console(f"Cosine Similarity Recs: {mode_title} (CURRENT HASH TABLE - {n_users} Users)")
        self.write_console("="*70)
        self.update()

        baseline_times = []
        cosine_times = []
        sim_calc_times = []
        lookup_times = []
        gen_times = []
        all_top_sim_scores = []
        user_ids = [uid for uid, _ in self.dataset]

        for uid in user_ids:
            b_recs, b_timing, _ = self.recommender.recommend_movies_baseline(uid, top_n=5)
            c_recs, c_timing, c_neighbors, _ = self.recommender.recommend_movies_cosine(uid, top_n=5, top_k_users=10)
            baseline_times.append(b_timing["total_time"])
            cosine_times.append(c_timing["total_time"])
            sim_calc_times.append(c_timing["similarity_calc_time"])
            lookup_times.append(c_timing["hash_lookup_time"])
            gen_times.append(c_timing["rec_gen_time"])
            if c_neighbors:
                all_top_sim_scores.append(c_neighbors[0][1])

        sample_indices = [0, n_users // 2, n_users - 1] if n_users >= 3 else list(range(n_users))
        sample_indices = list(dict.fromkeys(sample_indices))

        label_context = "Same 3 Benchmark Users" if is_reproducible else "3 Newly Sampled Users"
        self.write_console(f"\n--- Evaluation of {label_context} ---")
        self.write_console("-" * 70)
        for idx in sample_indices:
            target_uid, original_prefs = self.dataset[idx]
            b_recs, b_timing, _ = self.recommender.recommend_movies_baseline(target_uid, top_n=5)
            c_recs, c_timing, c_neighbors, _ = self.recommender.recommend_movies_cosine(target_uid, top_n=5, top_k_users=10)

            self.write_console(f"User ID: {target_uid} (Stored in Hash Table)")
            self.write_console(f"  Original Preferences:     {original_prefs}")
            self.write_console(f"  Baseline Recommendations: {b_recs} ({b_timing['total_time']*1000:.3f} ms)")
            self.write_console(f"  Cosine Recommendations:   {c_recs} ({c_timing['total_time']*1000:.3f} ms)")
            top_neighbors_info = [f"User {u[0]} (sim: {u[1]:.4f})" for u in c_neighbors[:5]]
            self.write_console(f"  Top Similar Neighbors:    {top_neighbors_info}")
            self.write_console("-" * 70)

        avg_base = float(np.mean(baseline_times))
        avg_cos = float(np.mean(cosine_times))
        avg_lookup = float(np.mean(lookup_times))
        avg_sim = float(np.mean(sim_calc_times))
        avg_gen = float(np.mean(gen_times))
        avg_max_sim = float(np.mean(all_top_sim_scores)) if all_top_sim_scores else 0.0

        self.write_console(f"\nSummary for Current Dataset ({mode_title}, N={n_users} Users):")
        self.write_console(f"  Avg Baseline Time:  {avg_base*1000:.4f} ms")
        self.write_console(f"  Avg Cosine Time:    {avg_cos*1000:.4f} ms")
        self.write_console(f"    * Hash Lookup:      {avg_lookup*1000:.4f} ms")
        self.write_console(f"    * Similarity Calc:  {avg_sim*1000:.4f} ms")
        self.write_console(f"    * Rec Generation:   {avg_gen*1000:.4f} ms")
        self.write_console(f"  Avg Top Similar User Score: {avg_max_sim:.4f}")
        self.write_console("="*70 + "\n")

    def click_show_hash_table(self):
        if not self.hash_table:
            messagebox.showwarning("Empty Table", "Hash Table has not been populated yet. Please insert a dataset first.")
            return

        self.content_tabview.set("Hash Table Visualizer")

        for widget in self.visualizer_scroll.winfo_children():
            widget.destroy()

        max_display = 30
        limit = min(self.hash_table.size, max_display)

        for idx in range(limit):
            bucket = self.hash_table.buckets[idx]
            is_collision = len(bucket) > 1
            border_c = COLOR_WARNING if is_collision else COLOR_BORDER

            row_frame = ctk.CTkFrame(
                self.visualizer_scroll,
                fg_color=COLOR_BG_CARD,
                border_color=border_c,
                border_width=1.5,
                corner_radius=8,
                height=52
            )
            row_frame.pack(fill="x", padx=6, pady=3)
            row_frame.pack_propagate(False)

            # Interactive hover glow on bucket card
            def _make_hover(f, def_bc):
                f.bind("<Enter>", lambda e: f.configure(border_color=COLOR_ACCENT_CYAN, border_width=2), add="+")
                f.bind("<Leave>", lambda e: f.configure(border_color=def_bc, border_width=1.5), add="+")
            _make_hover(row_frame, border_c)

            idx_lbl = ctk.CTkLabel(
                row_frame,
                text=f"Index {idx:02d}",
                width=80,
                font=ctk.CTkFont(family="Consolas", size=11, weight="bold"),
                text_color=COLOR_ACCENT_CYAN
            )
            idx_lbl.pack(side="left", padx=12)

            arrow_lbl = ctk.CTkLabel(row_frame, text="➔", font=ctk.CTkFont(size=13), text_color=COLOR_TEXT_MUTED)
            arrow_lbl.pack(side="left", padx=4)

            content_container = ctk.CTkFrame(row_frame, fg_color="transparent")
            content_container.pack(side="left", fill="both", expand=True, padx=6)

            if not bucket:
                empty_pill = ctk.CTkFrame(content_container, fg_color=COLOR_BG_INPUT, border_color=COLOR_BORDER, border_width=1, corner_radius=6, height=32)
                empty_pill.pack(side="left", pady=10)
                empty_lbl = ctk.CTkLabel(empty_pill, text="[ Empty Slot ]", font=ctk.CTkFont(size=9, slant="italic"), text_color=COLOR_TEXT_MUTED)
                empty_lbl.pack(padx=10, pady=4)
            else:
                if is_collision:
                    col_badge = ctk.CTkLabel(
                        content_container,
                        text=f"⚠️ {len(bucket)} keys",
                        font=ctk.CTkFont(size=9, weight="bold"),
                        text_color=COLOR_WARNING,
                        fg_color="#3B2610",
                        corner_radius=4,
                        padx=6,
                        pady=2
                    )
                    col_badge.pack(side="left", padx=(0, 6), pady=10)

                for i, (k, v) in enumerate(bucket):
                    node_color = COLOR_BG_CARD_ELEVATED if not is_collision else "#2D2218"
                    node_border = COLOR_BORDER_HOVER if not is_collision else COLOR_WARNING

                    pill = ctk.CTkFrame(content_container, fg_color=node_color, border_color=node_border, border_width=1, corner_radius=6, height=36)
                    pill.pack(side="left", pady=8, padx=2)
                    pill.pack_propagate(False)

                    txt_lbl = ctk.CTkLabel(pill, text=f"User {k}", font=ctk.CTkFont(weight="bold", size=10), text_color=COLOR_TEXT_PRIMARY)
                    txt_lbl.pack(padx=8, pady=(2, 0), anchor="w")

                    sub_lbl = ctk.CTkLabel(pill, text=f"{len(v)} items", font=ctk.CTkFont(size=8), text_color=COLOR_TEXT_SECONDARY)
                    sub_lbl.pack(padx=8, pady=(0, 2), anchor="w")

                    if i < len(bucket) - 1:
                        conn_lbl = ctk.CTkLabel(content_container, text="➔", font=ctk.CTkFont(size=11), text_color=COLOR_ACCENT_PRIMARY)
                        conn_lbl.pack(side="left", padx=2)

        if self.hash_table.size > max_display:
            more_lbl = ctk.CTkLabel(self.visualizer_scroll, text=f"... and {self.hash_table.size - max_display} more indices ...", font=ctk.CTkFont(slant="italic"), text_color=COLOR_TEXT_MUTED)
            more_lbl.pack(pady=10)

    def click_search_animate(self):
        if not self.hash_table:
            messagebox.showwarning("Empty Table", "Hash Table has not been populated yet. Please insert a dataset first.")
            return

        search_key_str = self.search_entry.get().strip()
        if not search_key_str:
            messagebox.showwarning("Input Required", "Please enter a User ID to search.")
            return

        try:
            search_key = int(search_key_str)
        except ValueError:
            messagebox.showerror("Invalid Input", "User ID must be an integer.")
            return

        val, trace = self.hash_table.search(search_key)
        self.anim_title.configure(text="Search/Lookup Trace", text_color=COLOR_ACCENT_CYAN)
        self.content_tabview.set("Hash Table Visualizer")

        def run_search_trace(step_idx=0):
            if step_idx < len(trace):
                step = trace[step_idx]
                if step["step"] == "input":
                    self._set_stepper_active(0)
                    self.anim_text.configure(text=f"Searching for User ID: {step['val']}\nInitializing hash calculation...", text_color=COLOR_TEXT_PRIMARY)
                elif step["step"] == "hash":
                    self._set_stepper_active(1)
                    self.anim_text.configure(text=f"Hashing Key to Index:\nFormula: {step['formula']}\nTarget Index = {step['index']}", text_color=COLOR_ACCENT_CYAN)
                    self.highlight_visualizer_bucket(step["index"])
                elif step["step"] == "compare":
                    self._set_stepper_active(2)
                    matched_str = "MATCHED!" if step["matched"] else "no match"
                    color = COLOR_SUCCESS if step["matched"] else COLOR_TEXT_PRIMARY
                    self.anim_text.configure(text=f"Scanning chaining list in Index {step['index']}:\nPosition {step['bucket_idx']}: Key {step['current_key']} vs Target {step['target_key']} -> {matched_str}", text_color=color)
                elif step["step"] == "probe_search":
                    self._set_stepper_active(2)
                    matched_str = "MATCHED!" if step["matched"] else "no match"
                    color = COLOR_SUCCESS if step["matched"] else COLOR_TEXT_PRIMARY
                    self.anim_text.configure(text=f"Probing Linear Index {step['index']}: slot is {step['state']}.\nProbe count: {step['probes']}\nMatch: {matched_str}", text_color=color)
                    self.highlight_visualizer_bucket(step["index"])
                elif step["step"] == "done":
                    self._set_stepper_active(3)
                    if step["found"]:
                        self.anim_text.configure(text=f"SUCCESS: Found User ID {search_key}!\nMovies liked: {step['value']}", text_color=COLOR_SUCCESS)
                        self.write_console(f"Search Success: User {search_key} prefers movies: {step['value']}")
                        if self.recommender:
                            b_recs, b_t, _ = self.recommender.recommend_movies_baseline(search_key, top_n=3)
                            c_recs, c_t, c_sims, _ = self.recommender.recommend_movies_cosine(search_key, top_n=3)
                            top_sim_info = f", Top Neighbor: {c_sims[0][0]} ({c_sims[0][1]:.2f})" if c_sims else ""
                            self.write_console(f"  * Baseline Recs: {b_recs} (Total: {b_t['total_time']*1000:.3f} ms)")
                            self.write_console(f"  * Cosine Recs:   {c_recs} (Total: {c_t['total_time']*1000:.3f} ms{top_sim_info})")
                    else:
                        self.anim_text.configure(text=f"FAILURE: User ID {search_key} not present in hash table.", text_color=COLOR_DANGER)
                        self.write_console(f"Search Failure: User {search_key} not found.")
                self.after(800, lambda: run_search_trace(step_idx + 1))

        run_search_trace(0)

    def highlight_visualizer_bucket(self, index: int):
        children = self.visualizer_scroll.winfo_children()
        if index < len(children) and index >= 0:
            row_frame = children[index]
            if isinstance(row_frame, ctk.CTkFrame):
                original_color = row_frame.cget("border_color")
                row_frame.configure(border_color=COLOR_ACCENT_CYAN, border_width=2.5)
                self.after(1200, lambda: row_frame.configure(border_color=original_color, border_width=1.5))

    def click_generate_graphs(self):
        sizes_str = self.sizes_entry.get().strip()
        lfs_str = self.lfs_entry.get().strip()
        try:
            sizes = [int(s.strip()) for s in sizes_str.split(",") if s.strip()]
            lfs = [float(f.strip()) for f in lfs_str.split(",") if f.strip()]
            if not sizes or not lfs:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid comma-separated lists of sizes and load factors.")
            return

        mode = self.bench_strat_menu.get()
        self.write_console(f"\nGenerating performance comparison charts for Mode: {mode}...")
        self.content_tabview.set("Performance Charts")

        for f in [self.graph_frame_1, self.graph_frame_2, self.graph_frame_3]:
            for w in f.winfo_children():
                w.destroy()

        lookup_data = PerformanceSimulator.run_lookup_benchmark(sizes, strategy=mode)
        lookup_plot_data = lookup_data if mode == "Both" else lookup_data[mode]
        fig1 = graphs.plot_lookup_benchmark(lookup_plot_data)
        canvas1 = FigureCanvasTkAgg(fig1, master=self.graph_frame_1)
        canvas1.draw()
        canvas1.get_tk_widget().pack(fill="both", expand=True)

        collision_data = PerformanceSimulator.run_collision_experiment(lfs, strategy=mode)
        collision_plot_data = collision_data if mode == "Both" else collision_data[mode]
        fig2 = graphs.plot_collision_experiment(collision_plot_data)
        canvas2 = FigureCanvasTkAgg(fig2, master=self.graph_frame_2)
        canvas2.draw()
        canvas2.get_tk_widget().pack(fill="both", expand=True)

        if mode == "Both":
            fig3 = graphs.plot_load_factor_vs_lookup_time(collision_data)
            canvas3 = FigureCanvasTkAgg(fig3, master=self.graph_frame_3)
            canvas3.draw()
            canvas3.get_tk_widget().pack(fill="both", expand=True)
        else:
            lbl = ctk.CTkLabel(self.graph_frame_3, text="Load Factor vs Lookup Chart\n(Select 'Both' mode to render comparison)", font=ctk.CTkFont(size=10, slant="italic"), text_color=COLOR_TEXT_MUTED)
            lbl.pack(expand=True)

        self.write_console("Charts generated successfully on the 'Performance Charts' tab.")

    def click_how_it_works(self):
        popup = EduPopup(self)
        popup.grab_set()

    def click_reset(self):
        self.dataset = []
        self.hash_table = None
        self.recommender = None
        self.write_console("\nSimulator state reset. Generating list cleared.", clear=True)
        self.anim_text.configure(text="No ongoing operation.\nUse search or insert to trace execution steps.", text_color=COLOR_TEXT_SECONDARY)
        self.anim_title.configure(text="⚡ Operation Step Trace", text_color=COLOR_ACCENT_CYAN)
        self._set_stepper_active(-1)

        for k in self.stats_labels:
            self.stats_labels[k].configure(text="0" if k != "Load Factor:" and k != "Avg Bucket:" else "0.00", text_color=COLOR_TEXT_PRIMARY)

        for widget in self.visualizer_scroll.winfo_children():
            widget.destroy()

        for f, lbl in [(self.graph_frame_1, "Lookup Performance Chart\n(Click 'Generate Graphs' to render)"), 
                        (self.graph_frame_2, "Collision Rate Chart\n(Click 'Generate Graphs' to render)"),
                        (self.graph_frame_3, "Load Factor vs Lookup Chart\n(Click 'Generate Graphs' to render)")]:
            for w in f.winfo_children():
                w.destroy()
            ctk.CTkLabel(f, text=lbl, font=ctk.CTkFont(size=10, slant="italic"), text_color=COLOR_TEXT_MUTED).pack(expand=True)


class EduPopup(ctk.CTkToplevel):
    """
    Educational popup window explaining Hash Table concepts visually with modern dark UI.
    """
    def __init__(self, parent):
        super().__init__(parent)
        self.title("How it Works: Hash Table Visual Guide")
        self.geometry("740x600")
        self.resizable(False, False)
        self.configure(fg_color=COLOR_BG_ROOT)

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        scroll = ctk.CTkScrollableFrame(self, fg_color=COLOR_BG_MAIN, corner_radius=10, border_color=COLOR_BORDER, border_width=1)
        scroll.grid(row=0, column=0, sticky="nsew", padx=16, pady=(16, 8))

        sections = [
            (
                "1. The Hash Function",
                "A Hash Function converts an arbitrary key (like a User ID) into an index\n"
                "within the memory bounds of the array.\n\n"
                "  User ID               Hash Formula               Memory Index\n"
                " ┌───────┐          ┌───────────────────┐          ┌──────────────┐\n"
                " │  105  │  ──────➔  │  105 % TableSize  │  ──────➔  │      5       │\n"
                " └───────┘          └───────────────────┘          └──────────────┘"
            ),
            (
                "2. Hashing Collisions & Chaining",
                "Collisions occur when two distinct keys yield the same index.\n"
                "Chaining handles this by storing multiple keys in a list (chain) at that index.\n\n"
                "  Keys (User IDs)       Target Index               Visual Chain\n"
                " ┌───────┐\n"
                " │  23   │ ──(23 % 10 = 3)──┐\n"
                " └───────┘                  │                      ┌───────────────┐\n"
                "                            ▼                      │ Index 3:      │\n"
                "                         Index 3   ──────────────➔ │  (23) ➔ (113) │\n"
                "                            ▲                      └───────────────┘\n"
                " ┌───────┐                  │\n"
                " │  113  │ ──(113 % 10 = 3)─┘\n"
                " └───────┘"
            ),
            (
                "3. Load Factor",
                "Load Factor represents how full the hash table is:\n\n"
                "               Total Inserted Keys (N)\n"
                " Load Factor = ───────────────────────\n"
                "                  Table Size (M)\n\n"
                "  - Higher Load Factor   = More full table, higher probability of collisions.\n"
                "  - Optimal Load Factor  = Typically between 0.70 and 0.75."
            ),
            (
                "4. Rehashing",
                "When the load factor exceeds a threshold, the table undergoes rehashing:\n\n"
                " Double Table Size      Recompute indices for       Collisions and\n"
                " (M ──➔ 2 * M)      ──➔ all existing keys       ──➔ chain lengths\n"
                "                        with new Table Size          decrease!"
            )
        ]

        for title, diagram in sections:
            card = ctk.CTkFrame(scroll, fg_color=COLOR_BG_CARD, border_color=COLOR_BORDER, border_width=1, corner_radius=8)
            card.pack(fill="x", padx=10, pady=8)

            lbl_t = ctk.CTkLabel(card, text=title, font=ctk.CTkFont(size=13, weight="bold"), text_color=COLOR_ACCENT_CYAN)
            lbl_t.pack(anchor="w", padx=16, pady=(10, 4))

            lbl_d = ctk.CTkLabel(card, text=diagram, font=ctk.CTkFont(family="Consolas", size=10), justify="left", text_color=COLOR_TEXT_PRIMARY)
            lbl_d.pack(anchor="w", padx=20, pady=(0, 14))

        btn_close = ctk.CTkButton(
            self,
            text="Close Guide",
            command=self.destroy,
            fg_color=COLOR_ACCENT_PRIMARY,
            hover_color=COLOR_ACCENT_HOVER,
            width=120,
            corner_radius=6
        )
        btn_close.grid(row=1, column=0, pady=(4, 12))


if __name__ == "__main__":
    app = App()
    app.mainloop()
