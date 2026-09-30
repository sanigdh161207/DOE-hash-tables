import tkinter as tk
from tkinter import messagebox
import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time
import datetime
import random
from typing import Dict, Any, List, Optional, Tuple

# Custom modules
from data_generator import generate_user_data, generate_structured_data, format_first_n_users, get_movie_genres
from recommender import HashTable, RecommenderSystem, HashTableChaining, HashTableLinearProbing
from sparse_recommender import SparseRecommender
from extendible_hashing import ExtendibleHashTable
from simulator import PerformanceSimulator
import graphs

# Set appearance mode and color theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# ── Theme Color Palettes ──
THEME_DARK = {
    "name": "dark",
    "sidebar_bg": "#18181C",
    "main_bg": "#121215",
    "card_bg": "#1A1A1E",
    "card_inner": "#202025",
    "card_alt": "#18181C",
    "separator": "#2A2A32",
    "accent": "#00E5FF",
    "btn_primary": "#1F6AA5",
    "btn_success": "#2E7D32",
    "btn_success_hover": "#1B5E20",
    "btn_danger": "#C62828",
    "btn_danger_hover": "#B71C1C",
    "text_primary": "#E0E0E0",
    "text_secondary": "#A0A0A0",
    "text_muted": "#666666",
    "border": "#333333",
    "console_bg": "#0D0E12",
    "console_header": "#18181C",
    "console_border": "#282932",
    "console_btn_bg": "#252733",
    "console_btn_hover": "#353846",
    "tag_text": "#D6DEEB",
    "tag_time": "#637777",
    "tag_header": "#00E5FF",
    "tag_success": "#00E676",
    "tag_error": "#FF5252",
    "tag_metric": "#FFD54F",
    "tag_step": "#B388FF",
    "pill_bg": "#2A2A32",
    "empty_pill_bg": "#202025",
}

THEME_LIGHT = {
    "name": "light",
    "sidebar_bg": "#F0F2F5",
    "main_bg": "#FFFFFF",
    "card_bg": "#F7F8FA",
    "card_inner": "#EBEDF0",
    "card_alt": "#F0F2F5",
    "separator": "#D0D5DD",
    "accent": "#0077B6",
    "btn_primary": "#1565C0",
    "btn_success": "#2E7D32",
    "btn_success_hover": "#1B5E20",
    "btn_danger": "#C62828",
    "btn_danger_hover": "#B71C1C",
    "text_primary": "#1A1A1A",
    "text_secondary": "#555555",
    "text_muted": "#999999",
    "border": "#C0C6D0",
    "console_bg": "#F8FAFC",
    "console_header": "#EDF2F7",
    "console_border": "#CBD5E1",
    "console_btn_bg": "#E2E8F0",
    "console_btn_hover": "#CBD5E1",
    "tag_text": "#1E293B",
    "tag_time": "#64748B",
    "tag_header": "#0284C7",
    "tag_success": "#16A34A",
    "tag_error": "#DC2626",
    "tag_metric": "#D97706",
    "tag_step": "#7C3AED",
    "pill_bg": "#E3E8EF",
    "empty_pill_bg": "#EBEDF0",
}


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # Configure window settings
        self.title("Hash Table Recommender Simulator — DOE Week 7 + 8")
        self.geometry("1300x820")
        self.resizable(True, True)
        
        # Application state
        self.dataset_size = 100
        self.table_size_mode = "Auto"
        self.custom_table_size = 150
        self.selected_load_factor = 0.75
        self.dataset: List[Tuple[int, List[int]]] = []
        self.hash_table: Optional[Any] = None
        self.recommender: Optional[RecommenderSystem] = None
        self.sparse_recommender: Optional[SparseRecommender] = None
        self.selected_strategy = "Separate Chaining"
        self.rec_mode = "IDF-Weighted Cosine"
        self.current_theme = THEME_DARK
        
        # Build UI layout
        self.setup_ui()
        
        # Welcome console message
        self.write_console(
            "=== Hash Table Recommender Simulator (DOE Week 7 + Week 8) ===\n"
            "Select dataset configuration and click 'Generate Dataset' to start."
        )

    def setup_ui(self):
        # Configure master layout: Sidebar (col 0) & Main Panel (col 1)
        self.grid_columnconfigure(0, weight=0, minsize=300)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # --- LEFT SIDEBAR PANEL ---
        self.sidebar_frame = ctk.CTkFrame(self, width=300, corner_radius=0, fg_color="#18181C")
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        self.sidebar_frame.grid_propagate(False)
        
        # App Header row with title + theme toggle
        self.header_row = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        self.header_row.pack(fill="x", padx=20, pady=(15, 5))

        self.app_title = ctk.CTkLabel(
            self.header_row,
            text="SIMULATOR CONTROLS",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="#00E5FF"
        )
        self.app_title.pack(side="left", anchor="w")

        # Light / Dark mode toggle
        self.theme_switch_var = ctk.StringVar(value="dark")
        self.theme_label = ctk.CTkLabel(
            self.header_row, text="☀",
            font=ctk.CTkFont(size=14), text_color="#FFD54F"
        )
        self.theme_label.pack(side="right", padx=(4, 0))
        self.theme_switch = ctk.CTkSwitch(
            self.header_row, text="", width=40,
            variable=self.theme_switch_var,
            onvalue="dark", offvalue="light",
            command=self.toggle_theme
        )
        self.theme_switch.pack(side="right", padx=4)
        self.theme_label_dark = ctk.CTkLabel(
            self.header_row, text="🌙",
            font=ctk.CTkFont(size=14)
        )
        self.theme_label_dark.pack(side="right", padx=(0, 4))
        
        # 1. Dataset Configuration
        self.ds_label = ctk.CTkLabel(self.sidebar_frame, text="Dataset Size (Users):", font=ctk.CTkFont(size=11, weight="bold"))
        self.ds_label.pack(padx=20, pady=(5, 2), anchor="w")
        self.ds_option = ctk.CTkOptionMenu(self.sidebar_frame, values=["10", "50", "100", "500", "1000"], command=self.on_dataset_size_change)
        self.ds_option.set("100")
        self.ds_option.pack(padx=20, pady=(0, 10), fill="x")
        
        # Data Structure / Strategy Selection
        self.strat_label = ctk.CTkLabel(self.sidebar_frame, text="Backend Data Structure:", font=ctk.CTkFont(size=11, weight="bold"))
        self.strat_label.pack(padx=20, pady=(5, 2), anchor="w")
        self.strat_segmented = ctk.CTkOptionMenu(
            self.sidebar_frame,
            values=["Separate Chaining", "Linear Probing", "Extendible Hashing"],
            command=self.on_strategy_change
        )
        self.strat_segmented.set("Separate Chaining")
        self.strat_segmented.pack(padx=20, pady=(0, 10), fill="x")

        # Table Sizing Mode
        self.ts_label = ctk.CTkLabel(self.sidebar_frame, text="Table Sizing:", font=ctk.CTkFont(size=11, weight="bold"))
        self.ts_label.pack(padx=20, pady=(5, 2), anchor="w")
        self.ts_mode = ctk.CTkSegmentedButton(self.sidebar_frame, values=["Auto", "Custom"], command=self.on_table_size_mode_change)
        self.ts_mode.set("Auto")
        self.ts_mode.pack(padx=20, pady=(0, 5), fill="x")
        
        self.custom_ts_entry = ctk.CTkEntry(self.sidebar_frame, placeholder_text="Enter Custom Table Size")
        self.custom_ts_entry.insert(0, "150")

        # Load Factor Target
        self.lf_label = ctk.CTkLabel(self.sidebar_frame, text="Target Load Factor:", font=ctk.CTkFont(size=11, weight="bold"))
        self.lf_label.pack(padx=20, pady=(5, 2), anchor="w")
        # Load Factor options stored on self so chart code can read them dynamically
        self.load_factor_options = [0.25, 0.50, 0.75, 0.90]
        self.lf_segmented = ctk.CTkSegmentedButton(
            self.sidebar_frame,
            values=[str(v) for v in self.load_factor_options],
            command=self.on_load_factor_change
        )
        self.lf_segmented.set("0.75")
        self.lf_segmented.pack(padx=20, pady=(0, 10), fill="x")

        # Separator line
        self.sep = ctk.CTkFrame(self.sidebar_frame, height=2, fg_color="#2A2A32")
        self.sep.pack(padx=20, pady=8, fill="x")

        # Scrollable Action Buttons
        self.btn_scroll = ctk.CTkScrollableFrame(self.sidebar_frame, fg_color="transparent")
        self.btn_scroll.pack(fill="both", expand=True, padx=10, pady=5)

        self.btn_gen_ds = ctk.CTkButton(self.btn_scroll, text="1. Generate Structured Data", command=self.click_generate_dataset, fg_color="#1F6AA5")
        self.btn_gen_ds.pack(pady=4, fill="x")

        self.btn_insert = ctk.CTkButton(self.btn_scroll, text="2. Populate Hash Table", command=self.click_insert_table, fg_color="#1F6AA5")
        self.btn_insert.pack(pady=4, fill="x")

        self.btn_visualize = ctk.CTkButton(self.btn_scroll, text="3. View Hash Table Grid", command=self.click_show_hash_table)
        self.btn_visualize.pack(pady=4, fill="x")

        self.btn_rec_lab = ctk.CTkButton(self.btn_scroll, text="4. Open Recommender Lab", command=self.open_recommender_lab, fg_color="#2E7D32", hover_color="#1B5E20")
        self.btn_rec_lab.pack(pady=4, fill="x")

        self.btn_lookup = ctk.CTkButton(self.btn_scroll, text="Run Lookup Benchmark", command=self.click_run_lookup_benchmark)
        self.btn_lookup.pack(pady=4, fill="x")

        self.btn_collision = ctk.CTkButton(self.btn_scroll, text="Run Collision Test", command=self.click_run_collision_test)
        self.btn_collision.pack(pady=4, fill="x")

        self.btn_numpy = ctk.CTkButton(self.btn_scroll, text="Compare NumPy bincount", command=self.click_compare_numpy)
        self.btn_numpy.pack(pady=4, fill="x")

        self.btn_graphs = ctk.CTkButton(self.btn_scroll, text="Generate Charts", command=self.click_generate_graphs)
        self.btn_graphs.pack(pady=4, fill="x")

        self.btn_how_it_works = ctk.CTkButton(self.btn_scroll, text="How It Works (Edu)", command=self.click_how_it_works)
        self.btn_how_it_works.pack(pady=4, fill="x")

        self.btn_reset = ctk.CTkButton(self.btn_scroll, text="Reset Simulator", fg_color="#C62828", hover_color="#B71C1C", command=self.click_reset)
        self.btn_reset.pack(pady=(12, 5), fill="x")

        # Hidden inputs for graph options compatibility
        self.sizes_entry = ctk.CTkEntry(self.btn_scroll)
        self.sizes_entry.insert(0, "10, 50, 100, 500, 1000")
        self.lfs_entry = ctk.CTkEntry(self.btn_scroll)
        self.lfs_entry.insert(0, "0.25, 0.50, 0.75, 0.90")
        self.numpy_size_entry = ctk.CTkEntry(self.btn_scroll)
        self.numpy_size_entry.insert(0, "5000")
        self.bench_strat_menu = ctk.CTkOptionMenu(self.btn_scroll, values=["Both"])
        self.bench_strat_menu.set("Both")

        # --- RIGHT MAIN CONTENT PANEL ---
        self.right_frame = ctk.CTkFrame(self, fg_color="#121215", corner_radius=0)
        self.right_frame.grid(row=0, column=1, sticky="nsew", padx=0, pady=0)
        self.right_frame.grid_columnconfigure(0, weight=1)
        self.right_frame.grid_rowconfigure(0, weight=3)  # Tabs area
        self.right_frame.grid_rowconfigure(1, weight=1)  # Stats panel
        self.right_frame.grid_rowconfigure(2, weight=1)  # Console log

        # Content Tabview
        self.content_tabview = ctk.CTkTabview(self.right_frame, fg_color="#1A1A1E")
        self.content_tabview.grid(row=0, column=0, sticky="nsew", padx=15, pady=(15, 5))

        self.tab_visualizer = self.content_tabview.add("Hash Table Visualizer")
        self.tab_recommender = self.content_tabview.add("Recommender Lab")
        self.tab_graphs = self.content_tabview.add("Performance Charts")

        # --- TAB 1: VISUALIZER ---
        self.tab_visualizer.grid_columnconfigure(0, weight=1)
        self.tab_visualizer.grid_rowconfigure(0, weight=1)
        self.tab_visualizer.grid_rowconfigure(1, weight=0)

        self.visualizer_scroll = ctk.CTkScrollableFrame(self.tab_visualizer, fg_color="#202025")
        self.visualizer_scroll.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        self.search_control_frame = ctk.CTkFrame(self.tab_visualizer, height=45, fg_color="#18181C")
        self.search_control_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=(0, 5))
        self.search_lbl = ctk.CTkLabel(self.search_control_frame, text="Lookup / Search User ID:", font=ctk.CTkFont(size=11, weight="bold"))
        self.search_lbl.pack(side="left", padx=10)
        self.search_entry = ctk.CTkEntry(self.search_control_frame, width=120, placeholder_text="e.g. 1005")
        self.search_entry.pack(side="left", padx=5)
        self.btn_search = ctk.CTkButton(self.search_control_frame, text="Search & Animate", command=self.click_search_animate, width=125)
        self.btn_search.pack(side="left", padx=8)

        self.speed_lbl = ctk.CTkLabel(self.search_control_frame, text="Anim Speed:", font=ctk.CTkFont(size=10, weight="bold"))
        self.speed_lbl.pack(side="left", padx=(12, 4))
        self.anim_speed_menu = ctk.CTkSegmentedButton(
            self.search_control_frame, values=["Fast", "Normal", "Slow"], width=130, height=24,
            font=ctk.CTkFont(size=9, weight="bold")
        )
        self.anim_speed_menu.set("Normal")
        self.anim_speed_menu.pack(side="left", padx=4)

        # --- TAB 2: RECOMMENDER LAB ---
        self.tab_recommender.grid_columnconfigure(0, weight=1)
        self.tab_recommender.grid_rowconfigure(0, weight=0)
        self.tab_recommender.grid_rowconfigure(1, weight=1)

        self.rec_control_frame = ctk.CTkFrame(self.tab_recommender, fg_color="#202025")
        self.rec_control_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)

        ctk.CTkLabel(self.rec_control_frame, text="Target User ID:", font=ctk.CTkFont(size=11, weight="bold")).pack(side="left", padx=8, pady=8)
        self.rec_user_entry = ctk.CTkEntry(self.rec_control_frame, width=110, placeholder_text="User ID")
        self.rec_user_entry.pack(side="left", padx=5, pady=8)

        ctk.CTkLabel(self.rec_control_frame, text="Top N:", font=ctk.CTkFont(size=11, weight="bold")).pack(side="left", padx=8, pady=8)
        self.rec_top_n_menu = ctk.CTkOptionMenu(self.rec_control_frame, values=["3", "5", "10"], width=70)
        self.rec_top_n_menu.set("5")
        self.rec_top_n_menu.pack(side="left", padx=5, pady=8)

        ctk.CTkLabel(self.rec_control_frame, text="Algorithm:", font=ctk.CTkFont(size=11, weight="bold")).pack(side="left", padx=8, pady=8)
        self.rec_algo_menu = ctk.CTkOptionMenu(
            self.rec_control_frame,
            values=["Baseline (Random)", "Cosine Similarity", "IDF-Weighted Cosine", "Hybrid Cold-Start", "Sparse CSR Cosine"],
            width=170
        )
        self.rec_algo_menu.set("IDF-Weighted Cosine")
        self.rec_algo_menu.pack(side="left", padx=5, pady=8)

        self.btn_gen_rec = ctk.CTkButton(self.rec_control_frame, text="Generate Recommendations", command=self.click_generate_recommendations, fg_color="#2E7D32")
        self.btn_gen_rec.pack(side="left", padx=10, pady=8)

        self.rec_output_frame = ctk.CTkScrollableFrame(self.tab_recommender, fg_color="#18181C")
        self.rec_output_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))

        # --- TAB 3: CHARTS ---
        self.tab_graphs.grid_columnconfigure(0, weight=1)
        self.tab_graphs.grid_columnconfigure(1, weight=1)
        self.tab_graphs.grid_columnconfigure(2, weight=1)
        self.tab_graphs.grid_rowconfigure(0, weight=1)

        self.graph_frame_1 = ctk.CTkFrame(self.tab_graphs, fg_color="#202025")
        self.graph_frame_1.grid(row=0, column=0, sticky="nsew", padx=3, pady=5)
        self.graph_lbl_1 = ctk.CTkLabel(self.graph_frame_1, text="Lookup Benchmark Chart", font=ctk.CTkFont(size=10, slant="italic"))
        self.graph_lbl_1.pack(expand=True)

        self.graph_frame_2 = ctk.CTkFrame(self.tab_graphs, fg_color="#202025")
        self.graph_frame_2.grid(row=0, column=1, sticky="nsew", padx=3, pady=5)
        self.graph_lbl_2 = ctk.CTkLabel(self.graph_frame_2, text="Collision Rate Chart", font=ctk.CTkFont(size=10, slant="italic"))
        self.graph_lbl_2.pack(expand=True)

        self.graph_frame_3 = ctk.CTkFrame(self.tab_graphs, fg_color="#202025")
        self.graph_frame_3.grid(row=0, column=2, sticky="nsew", padx=3, pady=5)
        self.graph_lbl_3 = ctk.CTkLabel(self.graph_frame_3, text="Load Factor Chart", font=ctk.CTkFont(size=10, slant="italic"))
        self.graph_lbl_3.pack(expand=True)

        # --- MIDDLE PANEL: ANIMATION STEP TRACE & REAL-TIME STATS ---
        self.middle_frame = ctk.CTkFrame(self.right_frame, height=130, fg_color="#18181C")
        self.middle_frame.grid(row=1, column=0, sticky="nsew", padx=15, pady=5)
        self.middle_frame.grid_propagate(False)
        self.middle_frame.grid_columnconfigure(0, weight=2)
        self.middle_frame.grid_columnconfigure(1, weight=1)
        self.middle_frame.grid_rowconfigure(0, weight=1)

        # Step Trace Panel
        self.anim_panel = ctk.CTkFrame(self.middle_frame, fg_color="#202025")
        self.anim_panel.grid(row=0, column=0, sticky="nsew", padx=(5, 2), pady=5)
        self.anim_title = ctk.CTkLabel(self.anim_panel, text="Operation Step Trace", font=ctk.CTkFont(size=12, weight="bold"), text_color="#00E5FF")
        self.anim_title.pack(anchor="w", padx=10, pady=(4, 1))

        # Animated Progress Bar
        self.anim_progress = ctk.CTkProgressBar(self.anim_panel, height=4, corner_radius=2, progress_color="#00E5FF")
        self.anim_progress.set(0.0)
        self.anim_progress.pack(fill="x", padx=10, pady=(2, 3))

        self.anim_text = ctk.CTkLabel(self.anim_panel, text="No ongoing operation.\nUse search or insert to trace execution steps.", justify="left", font=ctk.CTkFont(size=11))
        self.anim_text.pack(fill="both", expand=True, padx=10, pady=(0, 5))

        # Real-time Stats Panel
        self.stats_panel = ctk.CTkFrame(self.middle_frame, fg_color="#202025")
        self.stats_panel.grid(row=0, column=1, sticky="nsew", padx=(2, 5), pady=5)
        self.stats_title = ctk.CTkLabel(self.stats_panel, text="Hash Table Statistics", font=ctk.CTkFont(size=12, weight="bold"), text_color="#00E5FF")
        self.stats_title.grid(row=0, column=0, columnspan=2, sticky="w", padx=10, pady=2)

        self.stats_labels = {}
        fields = [
            ("Users:", "0"),
            ("Table Size:", "0"),
            ("Collisions:", "0"),
            ("Load Factor:", "0.0"),
            ("Avg Bucket Len:", "0.0"),
            ("Max Bucket Len:", "0")
        ]
        for idx, (label_txt, val) in enumerate(fields):
            r = (idx // 2) + 1
            c = (idx % 2) * 2
            lbl = ctk.CTkLabel(self.stats_panel, text=label_txt, font=ctk.CTkFont(size=10, weight="bold"))
            lbl.grid(row=r, column=c, sticky="w", padx=(10, 2), pady=1)
            val_lbl = ctk.CTkLabel(self.stats_panel, text=val, font=ctk.CTkFont(size=10))
            val_lbl.grid(row=r, column=c+1, sticky="w", padx=2, pady=1)
            self.stats_labels[label_txt] = val_lbl

        # --- BOTTOM PANEL: OUTPUT CONSOLE LOG ---
        self.console_log_history = []
        self.console_font_size = 10
        self.console_wrap_mode = True
        self.console_heights = {"S": 140, "M": 210, "L": 350}

        self.console_frame = ctk.CTkFrame(
            self.right_frame, height=210, fg_color="#18181C",
            corner_radius=8, border_width=1, border_color="#282932"
        )
        self.console_frame.grid(row=2, column=0, sticky="nsew", padx=15, pady=(5, 15))
        self.console_frame.grid_propagate(False)

        # Console Header Toolbar
        self.console_header = ctk.CTkFrame(self.console_frame, height=32, fg_color="transparent")
        self.console_header.pack(fill="x", padx=10, pady=(6, 3))

        # Left Header: prompt icon, title, status indicator, line counter
        self.console_left_header = ctk.CTkFrame(self.console_header, fg_color="transparent")
        self.console_left_header.pack(side="left", fill="y")

        self.console_prompt_lbl = ctk.CTkLabel(
            self.console_left_header, text=">_",
            font=ctk.CTkFont(family="Consolas", size=12, weight="bold"),
            text_color="#00E5FF"
        )
        self.console_prompt_lbl.pack(side="left", padx=(0, 6))

        self.console_title = ctk.CTkLabel(
            self.console_left_header, text="OUTPUT CONSOLE & EXECUTION LOG",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#E0E0E0"
        )
        self.console_title.pack(side="left", padx=(0, 8))

        self.console_live_dot = ctk.CTkLabel(
            self.console_left_header, text="●",
            font=ctk.CTkFont(size=10), text_color="#00E676"
        )
        self.console_live_dot.pack(side="left", padx=(0, 2))

        self.console_status_lbl = ctk.CTkLabel(
            self.console_left_header, text="LIVE",
            font=ctk.CTkFont(size=9, weight="bold"),
            text_color="#00E676"
        )
        self.console_status_lbl.pack(side="left", padx=(0, 10))

        self.console_count_lbl = ctk.CTkLabel(
            self.console_left_header, text="0 lines",
            font=ctk.CTkFont(size=9),
            text_color="#888888"
        )
        self.console_count_lbl.pack(side="left")

        # Right Header: Controls (Filter, Height, Zoom, Wrap, Copy, Clear)
        self.console_right_header = ctk.CTkFrame(self.console_header, fg_color="transparent")
        self.console_right_header.pack(side="right", fill="y")

        # Filter Entry
        self.console_filter = ctk.CTkEntry(
            self.console_right_header, placeholder_text="🔍 Filter log...",
            width=105, height=22, font=ctk.CTkFont(size=10)
        )
        self.console_filter.pack(side="left", padx=(0, 6))
        self.console_filter.bind("<KeyRelease>", self.filter_console_logs)



        # Font Zoom A- / A+
        self.btn_font_minus = ctk.CTkButton(
            self.console_right_header, text="A-", width=28, height=22,
            font=ctk.CTkFont(size=9, weight="bold"),
            fg_color="#252733", hover_color="#353846",
            command=self.zoom_font_minus
        )
        self.btn_font_minus.pack(side="left", padx=1)

        self.btn_font_plus = ctk.CTkButton(
            self.console_right_header, text="A+", width=28, height=22,
            font=ctk.CTkFont(size=9, weight="bold"),
            fg_color="#252733", hover_color="#353846",
            command=self.zoom_font_plus
        )
        self.btn_font_plus.pack(side="left", padx=(1, 6))

        # Word Wrap Toggle
        self.btn_wrap = ctk.CTkButton(
            self.console_right_header, text="↩ Wrap", width=55, height=22,
            font=ctk.CTkFont(size=9, weight="bold"),
            fg_color="#252733", hover_color="#353846",
            command=self.toggle_console_wrap
        )
        self.btn_wrap.pack(side="left", padx=(0, 6))

        # Copy & Clear
        self.btn_copy_console = ctk.CTkButton(
            self.console_right_header, text="📋 Copy", width=55, height=22,
            font=ctk.CTkFont(size=10, weight="bold"),
            fg_color="#252733", hover_color="#353846",
            command=self.copy_console
        )
        self.btn_copy_console.pack(side="left", padx=(0, 4))

        self.btn_clear_console = ctk.CTkButton(
            self.console_right_header, text="🗑 Clear", width=55, height=22,
            font=ctk.CTkFont(size=10, weight="bold"),
            fg_color="#252733", hover_color="#353846",
            command=self.clear_console
        )
        self.btn_clear_console.pack(side="left")

        # Textbox with generous spacing and monospace syntax font
        self.console_textbox = ctk.CTkTextbox(
            self.console_frame, state="disabled",
            font=ctk.CTkFont(family="Consolas", size=10),
            fg_color="#0D0E12", corner_radius=6,
            wrap="word"
        )
        self.console_textbox.pack(fill="both", expand=True, padx=8, pady=(2, 8))
        try:
            self.console_textbox._textbox.configure(spacing1=2, spacing3=2)
        except Exception:
            pass
        self.configure_console_tags()
        self.start_pulsing_dot()

    # --- UI CALLBACKS & ACTIONS ---
    def configure_console_tags(self):
        t = self.current_theme
        try:
            tb = self.console_textbox._textbox
            tb.tag_config("normal", foreground=t.get("tag_text", "#D6DEEB"))
            tb.tag_config("time", foreground=t.get("tag_time", "#637777"))
            tb.tag_config("header", foreground=t.get("tag_header", "#00E5FF"), font=("Consolas", self.console_font_size, "bold"))
            tb.tag_config("success", foreground=t.get("tag_success", "#00E676"), font=("Consolas", self.console_font_size, "bold"))
            tb.tag_config("error", foreground=t.get("tag_error", "#FF5252"), font=("Consolas", self.console_font_size, "bold"))
            tb.tag_config("metric", foreground=t.get("tag_metric", "#FFD54F"))
            tb.tag_config("step", foreground=t.get("tag_step", "#B388FF"), font=("Consolas", self.console_font_size, "bold"))
        except Exception:
            pass

    def toggle_console_wrap(self):
        self.console_wrap_mode = not self.console_wrap_mode
        wrap_val = "word" if self.console_wrap_mode else "none"
        self.console_textbox._textbox.configure(wrap=wrap_val)
        self.btn_wrap.configure(text="↩ Wrap" if self.console_wrap_mode else "⇥ NoWrap")

    def zoom_font_minus(self):
        if self.console_font_size > 8:
            self.console_font_size -= 1
            self.apply_console_font()

    def zoom_font_plus(self):
        if self.console_font_size < 16:
            self.console_font_size += 1
            self.apply_console_font()

    def apply_console_font(self):
        self.console_textbox.configure(font=ctk.CTkFont(family="Consolas", size=self.console_font_size))
        try:
            tb = self.console_textbox._textbox
            tb.tag_config("header", font=("Consolas", self.console_font_size, "bold"))
            tb.tag_config("success", font=("Consolas", self.console_font_size, "bold"))
            tb.tag_config("error", font=("Consolas", self.console_font_size, "bold"))
            tb.tag_config("step", font=("Consolas", self.console_font_size, "bold"))
        except Exception:
            pass



    def filter_console_logs(self, event=None):
        query = self.console_filter.get().strip().lower()
        self.render_console_buffer(query)

    def render_console_buffer(self, filter_query: str = ""):
        self.console_textbox.configure(state="normal")
        self.console_textbox.delete("1.0", "end")
        tb = self.console_textbox._textbox
        displayed_count = 0

        for now_str, line, tag in self.console_log_history:
            if filter_query and filter_query not in line.lower() and filter_query not in tag.lower():
                continue
            displayed_count += 1
            tb.insert("end", f"[{now_str}] ", "time")
            tb.insert("end", line + "\n", tag)

        self.console_textbox.see("end")
        self.console_textbox.configure(state="disabled")
        if hasattr(self, "console_count_lbl"):
            total = len(self.console_log_history)
            if filter_query:
                self.console_count_lbl.configure(text=f"{displayed_count}/{total} lines")
            else:
                self.console_count_lbl.configure(text=f"{total} lines")

    def copy_console(self):
        try:
            content = self.console_textbox.get("1.0", "end-1c")
            if content.strip():
                self.clipboard_clear()
                self.clipboard_append(content)
                self.btn_copy_console.configure(text="✓ Copied")
                self.after(1200, lambda: self.btn_copy_console.configure(text="📋 Copy"))
        except Exception:
            pass

    def clear_console(self):
        self.console_log_history.clear()
        self.render_console_buffer()
        self.write_console("Console log cleared. Ready for next operation.")

    def write_console(self, text: str, clear: bool = False):
        if clear:
            self.console_log_history.clear()

        now_str = datetime.datetime.now().strftime("%H:%M:%S")

        for line in text.split("\n"):
            if not line:
                continue

            l_strip = line.strip()
            if l_strip.startswith("===") or l_strip.startswith("---") or l_strip.startswith("━━"):
                tag = "header"
            elif any(w in line for w in ["Success", "FOUND", "successfully", "Speedup", "OK"]):
                tag = "success"
            elif any(w in line for w in ["Failure", "not present", "Error", "Warning", "failed"]):
                tag = "error"
            elif any(w in line for w in ["Users", "Table Size", "Collisions", "Load Factor", "Avg Bucket"]):
                tag = "metric"
            elif any(w in line for w in ["Step", "Running", "Generated", "Target User", "Algorithm:"]):
                tag = "step"
            else:
                tag = "normal"

            self.console_log_history.append((now_str, line, tag))

        filter_query = self.console_filter.get().strip().lower() if hasattr(self, "console_filter") else ""
        self.render_console_buffer(filter_query)

    def start_pulsing_dot(self):
        colors = ["#00E676", "#00B0FF", "#00E5FF", "#69F0AE", "#76FF03", "#00E5FF"]
        sizes  = [10, 11, 10, 9, 10, 11]
        def pulse(phase=0):
            try:
                if hasattr(self, "console_live_dot") and self.console_live_dot.winfo_exists():
                    self.console_live_dot.configure(
                        text_color=colors[phase % len(colors)],
                        font=ctk.CTkFont(size=sizes[phase % len(sizes)])
                    )
                self.after(600, lambda: pulse(phase + 1))
            except Exception:
                pass
        pulse(0)

    # ── UI Animation Helpers ──────────────────────────────────────────────────

    def _btn_flash(self, btn, flash_color: str = "#00E5FF", ms: int = 180):
        """Briefly flash a button's background to give click feedback."""
        try:
            orig = btn.cget("fg_color")
            btn.configure(fg_color=flash_color)
            self.after(ms, lambda: btn.configure(fg_color=orig))
        except Exception:
            pass

    def _set_busy(self, is_busy: bool, label: str = ""):
        """Disable/re-enable all sidebar action buttons and animate progress bar."""
        _sidebar_btns = [
            "btn_gen_ds", "btn_insert", "btn_visualize", "btn_rec_lab",
            "btn_lookup", "btn_collision", "btn_numpy", "btn_graphs",
            "btn_how_it_works", "btn_reset",
        ]
        state = "disabled" if is_busy else "normal"
        for name in _sidebar_btns:
            try:
                getattr(self, name).configure(state=state)
            except Exception:
                pass
        if is_busy:
            if label:
                try:
                    self.anim_title.configure(text=label)
                except Exception:
                    pass
            self._start_indeterminate_bar()
        else:
            self._stop_indeterminate_bar()

    def _start_indeterminate_bar(self):
        """Bounce the progress bar back and forth (indeterminate spinner)."""
        self._indet_active = True
        self._indet_val = 0.0
        self._indet_dir = 1
        accent = self.current_theme.get("accent", "#00E5FF")
        try:
            self.anim_progress.configure(progress_color=accent)
        except Exception:
            pass

        def _tick():
            if not getattr(self, "_indet_active", False):
                return
            self._indet_val = round(self._indet_val + 0.06 * self._indet_dir, 3)
            if self._indet_val >= 1.0:
                self._indet_val = 1.0
                self._indet_dir = -1
            elif self._indet_val <= 0.0:
                self._indet_val = 0.0
                self._indet_dir = 1
            try:
                self.anim_progress.set(self._indet_val)
            except Exception:
                pass
            self.after(28, _tick)

        _tick()

    def _stop_indeterminate_bar(self):
        """Stop the indeterminate animation and reset bar to zero."""
        self._indet_active = False
        try:
            self.anim_progress.set(0.0)
            self.anim_progress.configure(progress_color=self.current_theme.get("accent", "#00E5FF"))
            self.anim_title.configure(text="Operation Step Trace")
        except Exception:
            pass


    def _animate_stat_flash(self, lbl, final_text, flash_color="#00E5FF", normal_color=None):
        """Flash a stat label with a highlight color then restore."""
        if normal_color is None:
            normal_color = self.current_theme.get("text_primary", "#E0E0E0")
        lbl.configure(text=final_text, text_color=flash_color)
        self.after(400, lambda: lbl.configure(text_color=normal_color))

    def update_stats_ui(self):
        if not self.hash_table:
            return
        stats = self.hash_table.get_collision_statistics()
        t = self.current_theme
        nc = t.get("text_primary", "#E0E0E0")
        self._animate_stat_flash(self.stats_labels["Users:"],      str(stats.get("users", 0)),      "#00E5FF", nc)
        self._animate_stat_flash(self.stats_labels["Table Size:"], str(stats.get("table_size", 0)), "#00E5FF", nc)
        coll = stats.get("collisions", 0)
        coll_color = "#FF5252" if coll > 10 else "#00E676"
        self._animate_stat_flash(self.stats_labels["Collisions:"], str(coll),                       coll_color,  nc)
        lf = stats.get("load_factor", 0.0)
        lf_color = "#FF5252" if lf > 0.80 else ("#FFD54F" if lf > 0.60 else "#00E676")
        self._animate_stat_flash(self.stats_labels["Load Factor:"], f"{lf:.2f}",                    lf_color,    nc)
        if hasattr(self.hash_table, "global_depth"):
            self._animate_stat_flash(self.stats_labels["Avg Bucket Len:"], f"G-Depth:{stats.get('global_depth')}",  "#B388FF", nc)
            self._animate_stat_flash(self.stats_labels["Max Bucket Len:"], f"Splits:{stats.get('splits_count')}",    "#B388FF", nc)
        elif hasattr(self.hash_table, "TOMBSTONE"):
            self.stats_labels["Avg Bucket Len:"].configure(text="N/A")
            self.stats_labels["Max Bucket Len:"].configure(text="N/A")
        else:
            self._animate_stat_flash(self.stats_labels["Avg Bucket Len:"], f"{stats.get('avg_bucket_length', 0.0):.2f}", "#B388FF", nc)
            self._animate_stat_flash(self.stats_labels["Max Bucket Len:"], str(stats.get("max_bucket_length", 0)),        "#B388FF", nc)

    def on_strategy_change(self, choice):
        self.selected_strategy = choice
        self.write_console(f"Data structure strategy updated to {choice}.")
        if self.dataset:
            self.click_insert_table()

    def on_dataset_size_change(self, choice):
        self.dataset_size = int(choice)
        self.write_console(f"Dataset Size configured to {self.dataset_size}.")

    def on_table_size_mode_change(self, mode):
        self.table_size_mode = mode
        self.write_console(f"Table Sizing mode set to {mode}.")
        if mode == "Custom":
            self.custom_ts_entry.pack(padx=20, pady=(0, 5), fill="x")
        else:
            self.custom_ts_entry.pack_forget()

    def on_load_factor_change(self, choice):
        self.selected_load_factor = float(choice)
        self.write_console(f"Target Load Factor set to {self.selected_load_factor}.")

    def click_generate_dataset(self):
        self._btn_flash(self.btn_gen_ds)
        self._set_busy(True, "Generating structured dataset...")
        self.update_idletasks()
        try:
            self.dataset = generate_structured_data(num_users=self.dataset_size, num_movies=100, seed=42)
            formatted_txt = format_first_n_users(self.dataset, 10)
            self.write_console("\n" + "="*50 + "\nStructured Dataset Generated (70-80% Genre Preference Skew)!\n" + "="*50, clear=True)
            self.write_console(formatted_txt)
        finally:
            self._set_busy(False)
            self.anim_text.configure(
                text=f"✔ Dataset ready — {len(self.dataset)} user profiles generated.\nNext: click '2. Populate Hash Table'.",
                text_color="#00E676"
            )

    def click_insert_table(self):
        if not self.dataset:
            messagebox.showwarning("No Dataset", "Please generate a dataset first.")
            return

        if self.table_size_mode == "Auto":
            size = int(len(self.dataset) / self.selected_load_factor)
            if size % 2 == 0:
                size += 1
        else:
            try:
                size = int(self.custom_ts_entry.get())
                if size <= 0: raise ValueError
            except ValueError:
                messagebox.showerror("Invalid Size", "Please enter a valid positive integer.")
                return

        if self.selected_strategy == "Linear Probing":
            strat_class = HashTableLinearProbing
        elif self.selected_strategy == "Extendible Hashing":
            strat_class = ExtendibleHashTable
        else:
            strat_class = HashTableChaining

        if self.selected_strategy == "Extendible Hashing":
            self.hash_table = ExtendibleHashTable(table_size=size, bucket_capacity=4)
        else:
            self.hash_table = strat_class(size)

        self.recommender = RecommenderSystem(size, strategy_class=HashTableChaining if self.selected_strategy != "Linear Probing" else HashTableLinearProbing)
        self.recommender.fit(self.dataset, num_movies=100)

        self.sparse_recommender = SparseRecommender(num_movies=100)
        self.sparse_recommender.fit(self.dataset)

        self.write_console(f"Populating Hash Table using {self.selected_strategy}...")

        # Perform trace animation for first user
        first_user = self.dataset[0]
        trace = self.hash_table.insert(first_user[0], first_user[1], record_trace=True)

        # Batch insert rest
        for u_id, movies in self.dataset[1:]:
            self.hash_table.insert(u_id, movies, record_trace=False)

        self.write_console(f"Successfully inserted all {len(self.dataset)} user profiles!")
        self.update_stats_ui()
        self.click_show_hash_table()

        if self.dataset:
            self.rec_user_entry.delete(0, "end")
            self.rec_user_entry.insert(0, str(self.dataset[0][0]))

        # Smooth animated progress fill with color pulse on completion
        if hasattr(self, "anim_progress"):
            self.anim_progress.set(0.0)
            accent = self.current_theme.get("accent", "#00E5FF")
            self.anim_progress.configure(progress_color=accent)
            self.anim_title.configure(text=f"Populating Hash Table ({self.selected_strategy})")
            total = len(self.dataset)
            steps = 12
            def anim_fill(step=0):
                pct = min(1.0, step / steps)
                if step <= steps:
                    self.anim_progress.set(pct)
                    # Color shifts cyan → green as it fills
                    r = int(0x00 + (0x00 - 0x00) * pct)
                    g = int(0xE5 + (0xE6 - 0xE5) * pct)
                    b = int(0xFF + (0x76 - 0xFF) * pct)
                    bar_color = f"#{r:02X}{max(0,min(255,g)):02X}{max(0,min(255,b)):02X}"
                    try:
                        self.anim_progress.configure(progress_color=bar_color)
                    except Exception:
                        pass
                    self.anim_text.configure(
                        text=f"⚡ Batch indexing user profiles... {int(pct*100)}%\nLoaded {total} records into {self.selected_strategy}.",
                        text_color="#00E5FF"
                    )
                    self.after(30, lambda: anim_fill(step + 1))
                else:
                    self.anim_progress.set(1.0)
                    try:
                        self.anim_progress.configure(progress_color="#00E676")
                    except Exception:
                        pass
                    self.anim_text.configure(
                        text=f"✔ Complete: {total} users indexed into {self.selected_strategy}.\nReady — open Recommender Lab or run Benchmarks.",
                        text_color="#00E676"
                    )
                    # Fade bar back after 1.6s
                    self.after(1600, lambda: [
                        self.anim_progress.set(0.0),
                        self.anim_progress.configure(progress_color=accent) if hasattr(self, "anim_progress") else None
                    ])
            anim_fill(0)

    def click_show_hash_table(self):
        if not self.hash_table:
            messagebox.showwarning("Empty Table", "Please populate the Hash Table first.")
            return

        self.content_tabview.set("Hash Table Visualizer")
        for widget in self.visualizer_scroll.winfo_children():
            widget.destroy()

        self.visualizer_rows = {}
        self.visualizer_pills = {}

        if isinstance(self.hash_table, ExtendibleHashTable):
            ascii_art = self.hash_table.print_directory()
            lbl = ctk.CTkLabel(self.visualizer_scroll, text=ascii_art, font=ctk.CTkFont(family="Courier", size=11), justify="left")
            lbl.pack(anchor="w", padx=10, pady=10)
            return

        max_display = 25
        limit = min(self.hash_table.size, max_display)

        for idx in range(limit):
            bucket = self.hash_table.buckets[idx]
            is_collision = isinstance(bucket, list) and len(bucket) > 1
            border_color = graphs.SECONDARY_ACCENT if is_collision else "#333333"

            row_frame = ctk.CTkFrame(self.visualizer_scroll, fg_color="#18181C", border_color=border_color, border_width=1.5, height=55)
            row_frame.pack(fill="x", padx=5, pady=3)
            row_frame.pack_propagate(False)
            self.visualizer_rows[idx] = (row_frame, border_color, "#18181C")

            idx_lbl = ctk.CTkLabel(row_frame, text=f"Index {idx:02d}", width=80, font=ctk.CTkFont(weight="bold"), text_color="#00E5FF")
            idx_lbl.pack(side="left", padx=10)

            arrow_lbl = ctk.CTkLabel(row_frame, text="➔", font=ctk.CTkFont(size=14))
            arrow_lbl.pack(side="left", padx=5)

            content_container = ctk.CTkFrame(row_frame, fg_color="transparent")
            content_container.pack(side="left", fill="both", expand=True, padx=5)

            if not bucket:
                empty_pill = ctk.CTkFrame(content_container, fg_color="#202025", border_color="#333333", border_width=1, corner_radius=6, height=35)
                empty_pill.pack(side="left", pady=10)
                empty_lbl = ctk.CTkLabel(empty_pill, text="[ Empty ]", font=ctk.CTkFont(size=10, slant="italic"), text_color="#666666")
                empty_lbl.pack(padx=10, pady=5)
            else:
                b_list = bucket if isinstance(bucket, list) else [bucket]
                for i, item in enumerate(b_list):
                    if item is None or item == "<TOMBSTONE>":
                        continue
                    k, v = item
                    pill = ctk.CTkFrame(content_container, fg_color="#2A2A32", border_color="#00E5FF", border_width=1, corner_radius=6, height=38)
                    pill.pack(side="left", pady=8)
                    pill.pack_propagate(False)
                    self.visualizer_pills[k] = pill

                    txt_lbl = ctk.CTkLabel(pill, text=f"User {k}", font=ctk.CTkFont(weight="bold", size=10), text_color="#E0E0E0")
                    txt_lbl.pack(padx=8, pady=(2, 0), anchor="w")

                    sub_lbl = ctk.CTkLabel(pill, text=f"{len(v)} items", font=ctk.CTkFont(size=8), text_color="#A0A0A0")
                    sub_lbl.pack(padx=8, pady=(0, 2), anchor="w")

                    if i < len(b_list) - 1:
                        conn_lbl = ctk.CTkLabel(content_container, text=" ➔ ", font=ctk.CTkFont(size=12), text_color="#1F6AA5")
                        conn_lbl.pack(side="left", padx=3)

    def reset_visualizer_highlights(self):
        for idx, (frame, orig_border, orig_fg) in getattr(self, "visualizer_rows", {}).items():
            try:
                frame.configure(border_color=orig_border, border_width=1.5, fg_color=orig_fg)
            except Exception:
                pass
        for k, pill in getattr(self, "visualizer_pills", {}).items():
            try:
                pill.configure(border_color="#00E5FF", fg_color="#2A2A32")
            except Exception:
                pass

    def highlight_visualizer_bucket(self, idx: int, color: str = "#00E5FF"):
        rows = getattr(self, "visualizer_rows", {})
        if idx in rows:
            frame, orig_border, orig_fg = rows[idx]
            try:
                frame.configure(border_color=color, border_width=2.5, fg_color="#222330")
                if hasattr(self.visualizer_scroll, "_parent_canvas"):
                    limit = min(self.hash_table.size if self.hash_table else 25, 25)
                    self.visualizer_scroll._parent_canvas.yview_moveto(max(0.0, (idx - 1) / max(1, limit)))
            except Exception:
                pass

    def click_search_animate(self):
        if not self.hash_table:
            messagebox.showwarning("Empty Table", "Please populate the Hash Table first.")
            return

        search_key_str = self.search_entry.get().strip()
        if not search_key_str: return
        try: search_key = int(search_key_str)
        except ValueError: return

        speed_choice = self.anim_speed_menu.get() if hasattr(self, "anim_speed_menu") else "Normal"
        speed_delays = {"Fast": 300, "Normal": 650, "Slow": 1200}
        step_delay = speed_delays.get(speed_choice, 650)

        val, trace = self.hash_table.search(search_key, record_trace=True)
        self.anim_title.configure(text=f"Search Trace — Key {search_key}")
        self.content_tabview.set("Hash Table Visualizer")

        self.reset_visualizer_highlights()
        total_steps = len(trace)

        def run_search_trace(step_idx=0):
            if step_idx < total_steps:
                step = trace[step_idx]
                pct = (step_idx + 1) / max(1, total_steps)
                if hasattr(self, "anim_progress"):
                    self.anim_progress.set(pct)

                if step["step"] == "input":
                    self.anim_text.configure(
                        text=f"▶ Step 1/3: Reading Input Key\nSearching for User ID: {step['val']}",
                        text_color=self.current_theme["text_primary"]
                    )
                elif step["step"] == "hash":
                    idx = step.get("index", 0)
                    formula = step.get("formula", "")
                    self.anim_text.configure(
                        text=f"▶ Step 2/3: Hash Mapping\nFormula: {formula} ➔ Target Bucket [{idx:02d}]",
                        text_color="#FFD54F"
                    )
                    self.highlight_visualizer_bucket(idx, color="#00E5FF")
                elif step["step"] == "done":
                    if step.get("found"):
                        self.anim_text.configure(
                            text=f"✔ Step 3/3: MATCH FOUND!\nUser {search_key} prefers movies: {step['value']}",
                            text_color="#00E676"
                        )
                        self.write_console(f"Search Success: User {search_key} prefers movies {step['value']}")
                        if search_key in getattr(self, "visualizer_pills", {}):
                            try:
                                self.visualizer_pills[search_key].configure(border_color="#00E676", fg_color="#143A24")
                            except Exception:
                                pass
                    else:
                        self.anim_text.configure(
                            text=f"✖ Step 3/3: Key Not Present\nUser {search_key} was not found in Hash Table.",
                            text_color="#FF5252"
                        )
                        self.write_console(f"Search Failure: User {search_key} not found.")

                self.after(step_delay, lambda: run_search_trace(step_idx + 1))
            else:
                self.after(1800, lambda: self.anim_progress.set(0.0) if hasattr(self, "anim_progress") else None)

        run_search_trace(0)

    def open_recommender_lab(self):
        self.content_tabview.set("Recommender Lab")

    def click_generate_recommendations(self):
        if not self.recommender:
            messagebox.showwarning("No Engine", "Please populate the Hash Table first.")
            return

        uid_str = self.rec_user_entry.get().strip()
        if not uid_str: return
        try: uid = int(uid_str)
        except ValueError: return

        top_n = int(self.rec_top_n_menu.get())
        algo = self.rec_algo_menu.get()
        all_movies = list(range(1, 101))

        for widget in self.rec_output_frame.winfo_children():
            widget.destroy()

        t0 = time.perf_counter()
        if algo == "Baseline (Random)":
            recs, _ = self.recommender.recommend_movies(uid, all_movies, top_n=top_n)
            sources = ["random"] * len(recs)
        elif algo == "Cosine Similarity":
            recs, _ = self.recommender.recommend_movies_cosine(uid, all_movies, top_n=top_n)
            sources = ["cosine"] * len(recs)
        elif algo == "IDF-Weighted Cosine":
            recs, _ = self.recommender.recommend_movies_weighted_cosine(uid, all_movies, top_n=top_n, weighting="idf", popularity_alpha=0.5)
            sources = ["idf-cosine"] * len(recs)
        elif algo == "Hybrid Cold-Start":
            recs, sources, _ = self.recommender.recommend_movies_hybrid(uid, all_movies, top_n=top_n)
        else:
            recs, _ = self.sparse_recommender.recommend_movies(uid, all_movies, top_n=top_n)
            sources = ["sparse-csr"] * len(recs)

        elapsed = time.perf_counter() - t0

        header_lbl = ctk.CTkLabel(
            self.rec_output_frame,
            text=f"Top-{top_n} Recommendations for User {uid} (Algorithm: {algo} | Latency: {elapsed*1000:.2f} ms)",
            font=ctk.CTkFont(size=12, weight="bold"), text_color="#00E5FF"
        )
        header_lbl.pack(anchor="w", padx=10, pady=10)

        # Staggered smooth cascading reveal with flash highlight
        t = self.current_theme
        rank_colors = ["#00E5FF", "#FFD54F", "#B388FF", "#00E676", "#FF80AB",
                       "#80D8FF", "#CCFF90", "#FFD180", "#EA80FC", "#A7FFEB"]
        cards_to_show = []
        for rank, (m_id, src) in enumerate(zip(recs, sources), start=1):
            rc = rank_colors[(rank - 1) % len(rank_colors)]
            card = ctk.CTkFrame(
                self.rec_output_frame,
                fg_color=t["card_inner"],
                border_color=rc,
                border_width=1,
                corner_radius=8
            )
            rank_lbl = ctk.CTkLabel(
                card,
                text=f"  #{rank}",
                font=ctk.CTkFont(size=13, weight="bold"),
                text_color=rc,
                width=36
            )
            rank_lbl.pack(side="left", padx=(8, 4), pady=8)
            detail_lbl = ctk.CTkLabel(
                card,
                text=f"Movie ID {m_id:>4}   ·   via {src}",
                font=ctk.CTkFont(size=11),
                text_color=t["text_primary"]
            )
            detail_lbl.pack(side="left", padx=4, pady=8)
            cards_to_show.append((card, rc, t["card_inner"]))

        def reveal_card(c_idx=0):
            if c_idx < len(cards_to_show):
                card, flash_c, base_c = cards_to_show[c_idx]
                card.pack(fill="x", padx=10, pady=3)
                # Brief border flash then settle
                card.configure(border_color="#FFFFFF")
                self.after(120, lambda c=card, f=flash_c: c.configure(border_color=f))
                self.after(80, lambda: reveal_card(c_idx + 1))

        reveal_card(0)

        self.write_console(f"Generated Top-{top_n} recommendations for User {uid} using {algo}: {recs}")

    def click_run_lookup_benchmark(self):
        self._btn_flash(self.btn_lookup)
        self._set_busy(True, "Running Lookup Benchmark...")
        self.update_idletasks()
        try:
            self.write_console("\nRunning Lookup Benchmark...")
            res = PerformanceSimulator.run_lookup_benchmark([10, 50, 100, 500, 1000], strategy="Both")
            for strat, data in res.items():
                self.write_console(f"\n--- {strat} ---")
                for size, avg_t, coll, mem in data:
                    self.write_console(f"Users: {size:<5} | Avg Lookup: {avg_t*1e6:<8.2f} us | Collisions: {coll:<4} | Mem: {mem} bytes")
        finally:
            self._set_busy(False)

    def click_run_collision_test(self):
        self._btn_flash(self.btn_collision)
        self._set_busy(True, "Running Collision Test...")
        self.update_idletasks()
        try:
            self.write_console("\nRunning Collision Test...")
            res = PerformanceSimulator.run_collision_experiment([0.25, 0.50, 0.75, 0.90], strategy="Both")
            for strat, data in res.items():
                self.write_console(f"\n--- {strat} ---")
                for lf, rate, avg_t, mem, extra in data:
                    self.write_console(f"Load Factor: {lf:<4.2f} | Collision Rate: {rate*100:<5.1f}% | Avg Lookup: {avg_t*1e6:.2f} us")
        finally:
            self._set_busy(False)

    def click_compare_numpy(self):
        self._btn_flash(self.btn_numpy)
        self._set_busy(True, "Comparing NumPy vs Python list...")
        self.update_idletasks()
        try:
            self.write_console("\nRunning NumPy bincount Comparison...")
            res = PerformanceSimulator.compare_numpy_performance(5000)
            self.write_console(f"Python List Time: {res['python_list_time']:.6f} sec")
            self.write_console(f"NumPy bincount Time: {res['numpy_time']:.6f} sec")
            self.write_console(f"NumPy Speedup: {res['speedup_numpy']:.1f}x")
        finally:
            self._set_busy(False)

    def click_generate_graphs(self):
        """Generate performance charts entirely from the live loaded dataset.
        - No hardcoded sizes or load factors: all derived from current UI state.
        - Deterministic: same dataset → identical charts every run.
        """
        if not self.dataset:
            messagebox.showwarning("No Dataset", "Please generate and populate a dataset first.")
            return

        self._btn_flash(self.btn_graphs)
        self._set_busy(True, "Generating performance charts...")
        self.update_idletasks()

        self.content_tabview.set("Performance Charts")
        for f in [self.graph_frame_1, self.graph_frame_2, self.graph_frame_3]:
            for w in f.winfo_children():
                w.destroy()

        import time as _time
        import random as _rng
        import math as _math

        dataset = self.dataset
        n = len(dataset)
        all_keys = [uid for uid, _ in dataset]

        # ── Deterministic seed derived from actual dataset content ──
        # Same dataset  → same charts every time.
        # Different dataset → different charts.
        data_seed = (sum(all_keys) ^ (n * 31337)) % (2 ** 31)

        # ── Lookup benchmark sizes: log-spaced from min(10, n) to n, capped at 6 points ──
        min_sz = min(10, n)
        if n <= min_sz:
            sub_sizes = [n]
        else:
            num_points = min(6, n)
            raw = set()
            for i in range(num_points):
                frac = i / max(1, num_points - 1)
                v = int(min_sz * (_math.exp(_math.log(n / min_sz) * frac)))
                raw.add(max(min_sz, min(n, v)))
            raw.add(n)
            sub_sizes = sorted(raw)

        # ── Load factors: read from UI widget — no hardcoding ──
        load_factors = list(self.load_factor_options)

        # ── Number of lookups: scales with dataset size (200 – 1000) ──
        num_lookups = min(1000, max(200, n * 3))

        # ── Strategy map: always benchmark both for comparison ──
        strategy_pairs = [
            ("Separate Chaining", HashTableChaining),
            ("Linear Probing",   HashTableLinearProbing),
        ]

        # ───────────────────────────────────────────────────────────
        # Chart 1 — Lookup Time vs. Dataset Size
        # ───────────────────────────────────────────────────────────
        lookup_data: dict = {name: [] for name, _ in strategy_pairs}
        for strat_name, klass in strategy_pairs:
            for sz in sub_sizes:
                sub = dataset[:sz]
                # Table size driven by current UI load factor
                tsize = max(13, int(sz / self.selected_load_factor))
                tsize += (tsize % 2 == 0)  # ensure odd
                ht = klass(tsize)
                for uid, movies in sub:
                    ht.insert(uid, movies, record_trace=False)
                sub_keys = [uid for uid, _ in sub]
                # Deterministic lookup sequence for this (strategy, size) pair
                _rng.seed(data_seed ^ hash(strat_name) ^ sz)
                lk = [_rng.choice(sub_keys) for _ in range(num_lookups)]
                t0 = _time.perf_counter()
                for k in lk:
                    ht.search(k, record_trace=False)
                avg_t = (_time.perf_counter() - t0) / num_lookups
                st = ht.get_collision_statistics()
                lookup_data[strat_name].append((sz, avg_t, st["collisions"], ht.estimate_memory_bytes()))

        fig1 = graphs.plot_lookup_benchmark(lookup_data)
        c1 = FigureCanvasTkAgg(fig1, master=self.graph_frame_1)
        c1.draw()
        c1.get_tk_widget().pack(fill="both", expand=True)
        plt.close(fig1)

        # ───────────────────────────────────────────────────────────
        # Charts 2 & 3 — Collision Rate / Lookup Time vs. Load Factor
        # ───────────────────────────────────────────────────────────
        collision_data: dict = {name: [] for name, _ in strategy_pairs}
        for strat_name, klass in strategy_pairs:
            for lf in load_factors:
                tsize = max(13, int(n / lf))
                tsize += (tsize % 2 == 0)
                ht = klass(tsize)
                for uid, movies in dataset:
                    ht.insert(uid, movies, record_trace=False)
                # Deterministic lookup sequence for this (strategy, lf) pair
                _rng.seed(data_seed ^ hash(strat_name) ^ int(lf * 1000))
                lk = [_rng.choice(all_keys) for _ in range(num_lookups)]
                t0 = _time.perf_counter()
                for k in lk:
                    ht.search(k, record_trace=False)
                avg_t = (_time.perf_counter() - t0) / num_lookups
                st = ht.get_collision_statistics()
                collision_data[strat_name].append(
                    (lf, st["collision_rate"], avg_t, ht.estimate_memory_bytes(), {})
                )

        fig2 = graphs.plot_collision_experiment(collision_data)
        c2 = FigureCanvasTkAgg(fig2, master=self.graph_frame_2)
        c2.draw()
        c2.get_tk_widget().pack(fill="both", expand=True)
        plt.close(fig2)

        fig3 = graphs.plot_load_factor_vs_lookup_time(collision_data)
        c3 = FigureCanvasTkAgg(fig3, master=self.graph_frame_3)
        c3.draw()
        c3.get_tk_widget().pack(fill="both", expand=True)
        plt.close(fig3)

        self._set_busy(False)
        self.write_console(
            f"=== Performance Charts — Live Dataset ==="  
            f"\nDataset  : {n} users  |  Strategy : {self.selected_strategy}"
            f"\nLF target: {self.selected_load_factor}  |  Lookups/test: {num_lookups}"
            f"\nBenchmark sizes : {sub_sizes}"
            f"\nLoad factors    : {load_factors}"
        )


    def click_how_it_works(self):
        from app import EduPopup
        popup = EduPopup(self)
        popup.grab_set()

    def click_reset(self):
        self.dataset = []
        self.hash_table = None
        self.recommender = None
        self.sparse_recommender = None
        self.write_console("\nSimulator state reset.", clear=True)
        self.update_stats_ui()
        for w in self.visualizer_scroll.winfo_children(): w.destroy()

    # ── THEME TOGGLE ──
    def toggle_theme(self):
        """Switch between light and dark mode, re-applying colors to all panels."""
        mode = self.theme_switch_var.get()
        if mode == "dark":
            self.current_theme = THEME_DARK
            ctk.set_appearance_mode("dark")
        else:
            self.current_theme = THEME_LIGHT
            ctk.set_appearance_mode("light")
        t = self.current_theme

        # Sidebar
        self.sidebar_frame.configure(fg_color=t["sidebar_bg"])
        self.app_title.configure(text_color=t["accent"])

        # Separator
        self.sep.configure(fg_color=t["separator"])

        # Right panels
        self.right_frame.configure(fg_color=t["main_bg"])
        self.content_tabview.configure(fg_color=t["card_bg"])

        # Middle + console frames
        self.middle_frame.configure(fg_color=t["card_alt"])
        self.anim_panel.configure(fg_color=t["card_inner"])
        self.anim_title.configure(text_color=t["accent"])
        self.stats_panel.configure(fg_color=t["card_inner"])
        self.console_frame.configure(fg_color=t["card_alt"], border_color=t["console_border"])
        self.console_textbox.configure(fg_color=t["console_bg"])
        self.console_prompt_lbl.configure(text_color=t["accent"])
        self.console_title.configure(text_color=t["text_primary"])
        self.console_count_lbl.configure(text_color=t["text_muted"])
        self.btn_copy_console.configure(fg_color=t["console_btn_bg"], hover_color=t["console_btn_hover"])
        self.btn_clear_console.configure(fg_color=t["console_btn_bg"], hover_color=t["console_btn_hover"])
        if hasattr(self, "btn_wrap"):
            self.btn_wrap.configure(fg_color=t["console_btn_bg"], hover_color=t["console_btn_hover"])
            self.btn_font_minus.configure(fg_color=t["console_btn_bg"], hover_color=t["console_btn_hover"])
            self.btn_font_plus.configure(fg_color=t["console_btn_bg"], hover_color=t["console_btn_hover"])
            self.console_filter.configure(fg_color=t["console_bg"], border_color=t["console_border"], text_color=t["text_primary"])

        if hasattr(self, "anim_progress"):
            self.anim_progress.configure(progress_color=t["accent"])
        if hasattr(self, "anim_speed_menu"):
            self.anim_speed_menu.configure(selected_color=t["btn_primary"])
        self.configure_console_tags()

        # Search bar
        self.search_control_frame.configure(fg_color=t["card_alt"])

        # Visualizer + recommender backgrounds
        self.visualizer_scroll.configure(fg_color=t["card_inner"])
        self.rec_control_frame.configure(fg_color=t["card_inner"])
        self.rec_output_frame.configure(fg_color=t["card_alt"])

        # Graph placeholders
        for gf in [self.graph_frame_1, self.graph_frame_2, self.graph_frame_3]:
            gf.configure(fg_color=t["card_inner"])

        # Re-render hash table grid if populated
        if self.hash_table:
            self.click_show_hash_table()

        self.write_console(f"Theme switched to {mode} mode.")


class EduPopup(ctk.CTkToplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("How It Works — Complete Simulator Guide")
        self.geometry("860x700")
        self.resizable(True, True)

        t = parent.current_theme
        self.configure(fg_color=t["main_bg"])

        scroll = ctk.CTkScrollableFrame(self, fg_color=t["sidebar_bg"])
        scroll.pack(fill="both", expand=True, padx=10, pady=10)

        # --- TITLE ---
        ctk.CTkLabel(
            scroll, text="DOE Hash Table & Recommender Simulator — Full Guide",
            font=ctk.CTkFont(size=16, weight="bold"), text_color=t["accent"]
        ).pack(anchor="w", padx=10, pady=(10, 5))

        ctk.CTkLabel(
            scroll,
            text="Follow the numbered buttons (1 → 2 → 3 → 4) in order for the best experience.",
            font=ctk.CTkFont(size=11, slant="italic"), text_color=t["text_secondary"]
        ).pack(anchor="w", padx=10, pady=(0, 10))

        sections = [
            # ── QUICK-START WORKFLOW ──
            ("━━  QUICK-START WORKFLOW  ━━",
             "Step 1 ▸ Choose a Dataset Size and Backend Data Structure from the sidebar.\n"
             "Step 2 ▸ Click \"1. Generate Structured Data\".\n"
             "Step 3 ▸ Click \"2. Populate Hash Table\".\n"
             "Step 4 ▸ Explore the Hash Table Grid, Recommender Lab, and Benchmarks.\n"
             "Step 5 ▸ Click \"Generate Charts\" to visualize performance data.",
             "#FFD54F"),

            # ── SIDEBAR CONFIGURATION ──
            ("━━  SIDEBAR CONFIGURATION CONTROLS  ━━",
             "Dataset Size (Users)\n"
             "   Select how many synthetic user profiles to generate (10–1000).\n"
             "   Larger sizes show scaling behavior but take longer to process.\n\n"
             "Backend Data Structure\n"
             "   • Separate Chaining – Colliding keys are stored in linked chains at each bucket.\n"
             "   • Linear Probing – Colliding keys probe forward to the next empty slot.\n"
             "   • Extendible Hashing – Buckets split dynamically using a bitwise directory.\n"
             "   Changing this re-populates the table automatically if data exists.\n\n"
             "Table Sizing (Auto / Custom)\n"
             "   Auto: Table size = ceil(Users / Target Load Factor), always odd.\n"
             "   Custom: You type in an exact bucket count to see how capacity affects collisions.\n\n"
             "Target Load Factor (0.25 – 0.90)\n"
             "   Controls how full the table is. Higher values = more collisions.\n"
             "   0.75 is a common real-world default.",
             "#00E5FF"),

            # ── BUTTON-BY-BUTTON GUIDE ──
            ("━━  BUTTON-BY-BUTTON GUIDE  ━━", "", "#00E5FF"),

            ("1. Generate Structured Data",
             "Creates synthetic user-movie preference profiles with 70–80% genre skew.\n"
             "Each user gets a random set of movie IDs biased toward their preferred genre.\n\n"
             "  ✦ What to look for in the console:\n"
             "     A table showing User IDs and their movie lists.\n"
             "     The first 10 users are printed so you can see data variety.",
             "#1F6AA5"),

            ("2. Populate Hash Table",
             "Inserts every generated user into the selected hash table structure.\n"
             "Also initializes the Recommender engine and Sparse CSR matrix.\n\n"
             "  ✦ What happens:\n"
             "     • The first user is inserted with a recorded trace (visible in Step Trace).\n"
             "     • Remaining users are batch-inserted silently for speed.\n"
             "     • The Hash Table Visualizer auto-opens showing bucket layout.\n"
             "     • The Stats panel updates with collision count, load factor, etc.\n\n"
             "  ✦ How to read the Stats panel:\n"
             "     Users        – number of records stored\n"
             "     Table Size   – total bucket count\n"
             "     Collisions   – keys that hashed to an already-occupied bucket\n"
             "     Load Factor  – Users / Table Size (higher = more crowded)\n"
             "     Avg Bucket   – average chain length (chaining) or N/A (probing)\n"
             "     Max Bucket   – longest chain (worst-case lookup cost)",
             "#1F6AA5"),

            ("3. View Hash Table Grid",
             "Opens the Visualizer tab showing the internal bucket array.\n\n"
             "  ✦ How to read the grid:\n"
             "     • Each row = one bucket index (Index 00, 01, …).\n"
             "     • Cyan-bordered pills = stored user records (\"User 1005 — 8 items\").\n"
             "     • Arrows (➔) between pills = chained collisions at the same index.\n"
             "     • Orange-bordered rows = buckets with ≥2 collisions.\n"
             "     • \"[ Empty ]\" = unused bucket.\n\n"
             "  ✦ For Extendible Hashing:\n"
             "     An ASCII directory tree is shown instead, displaying global depth,\n"
             "     local depths, and bucket contents.",
             "#1F6AA5"),

            ("Search & Animate (in Visualizer tab)",
             "Type a User ID in the search bar and click \"Search & Animate\".\n\n"
             "  ✦ What happens:\n"
             "     • The Step Trace panel shows a live animation:\n"
             "       – \"Searching for User ID: X\"\n"
             "       – \"Hash Index: Y (Formula: X mod TableSize)\"\n"
             "       – \"FOUND\" (green) or \"Not present\" (red).\n"
             "     • The console logs the result with the user's movie list.\n\n"
             "  ✦ Try searching for a user ID that doesn't exist to see the failure path.",
             "#1F6AA5"),

            ("4. Open Recommender Lab",
             "Switches to the Recommender Lab tab where you can generate\n"
             "personalized movie recommendations.\n\n"
             "  ✦ Controls:\n"
             "     Target User ID – the user to get recommendations for.\n"
             "     Top N          – how many recommendations (3, 5, or 10).\n"
             "     Algorithm      – which engine to use (see below).\n\n"
             "  ✦ Algorithms explained:\n"
             "     Baseline (Random)     – picks random unwatched movies (no intelligence).\n"
             "     Cosine Similarity     – finds users with similar taste via vector angle.\n"
             "     IDF-Weighted Cosine   – same as above, but rare movies count more.\n"
             "     Hybrid Cold-Start     – blends cosine + popularity for new users.\n"
             "     Sparse CSR Cosine     – uses SciPy sparse matrix (fast for large data).\n\n"
             "  ✦ How to read the output:\n"
             "     Each recommendation card shows:\n"
             "       Rank #N: Movie ID XX  |  Source Strategy: xxx\n"
             "     The header shows algorithm name and latency in milliseconds.\n"
             "     Compare algorithms by switching and re-generating for the same user.",
             "#2E7D32"),

            ("Run Lookup Benchmark",
             "Measures average lookup time across dataset sizes (10–1000 users).\n\n"
             "  ✦ Console output columns:\n"
             "     Users      – dataset size\n"
             "     Avg Lookup – average time per lookup in microseconds (µs)\n"
             "     Collisions – total collision count at that size\n"
             "     Mem        – approximate memory footprint in bytes\n\n"
             "  ✦ What to look for:\n"
             "     Lookup time should stay roughly constant as size grows → O(1).\n"
             "     Both Chaining and Linear Probing results are shown side-by-side.",
             "#1F6AA5"),

            ("Run Collision Test",
             "Tests collision rates at different load factors (0.25, 0.50, 0.75, 0.90).\n\n"
             "  ✦ Console output columns:\n"
             "     Load Factor    – how full the table is\n"
             "     Collision Rate – percentage of inserts that collided\n"
             "     Avg Lookup     – lookup latency at this load factor\n\n"
             "  ✦ What to look for:\n"
             "     Collision rate rises steeply above 0.75 load factor.\n"
             "     Linear Probing degrades faster than Chaining at high loads.",
             "#1F6AA5"),

            ("Compare NumPy bincount",
             "Benchmarks Python list counting vs. NumPy vectorized bincount.\n\n"
             "  ✦ Console output:\n"
             "     Python List Time  – seconds for a pure-Python frequency count\n"
             "     NumPy bincount    – seconds for NumPy's C-optimized equivalent\n"
             "     Speedup           – how many times faster NumPy is (typically 10–50×)\n\n"
             "  ✦ This demonstrates why libraries like NumPy matter for data processing.",
             "#1F6AA5"),

            ("Generate Charts",
             "Opens the Performance Charts tab and renders 3 embedded Matplotlib plots:\n\n"
             "  ✦ Chart 1 – Lookup Benchmark:\n"
             "     X-axis = dataset size, Y-axis = avg lookup time.\n"
             "     A flat line confirms O(1) performance.\n\n"
             "  ✦ Chart 2 – Collision Rate:\n"
             "     X-axis = load factor, Y-axis = collision percentage.\n"
             "     Shows exponential growth as tables get fuller.\n\n"
             "  ✦ Chart 3 – Load Factor vs Lookup Time:\n"
             "     X-axis = load factor, Y-axis = lookup latency.\n"
             "     Reveals the cost of overfilling a hash table.",
             "#1F6AA5"),

            ("Reset Simulator",
             "Clears all data: dataset, hash table, recommender engines, and the\n"
             "visualizer grid. Resets the stats panel to zeros.\n"
             "Use this to start a fresh experiment with different settings.",
             "#C62828"),

            # ── CORE CONCEPTS ──
            ("━━  CORE CONCEPTS  ━━", "", "#00E5FF"),

            ("Hash Function",
             "Maps a key to a bucket index:  Index = UserID mod TableSize.\n"
             "Good hash functions distribute keys uniformly across all buckets.",
             "#FFD54F"),

            ("Collision",
             "When two different keys produce the same index.\n"
             "Resolution strategies: Chaining (linked lists), Probing (search forward),\n"
             "or Extendible Hashing (split buckets dynamically).",
             "#FFD54F"),

            ("Load Factor",
             "α = N / M  (stored items / table capacity).\n"
             "Higher α → more collisions → slower lookups.\n"
             "Real-world tables typically resize when α > 0.75.",
             "#FFD54F"),

            ("Cosine Similarity",
             "Measures the angle between two user-preference vectors.\n"
             "  cos(θ) = (A · B) / (||A|| × ||B||)\n"
             "A value of 1.0 = identical taste, 0.0 = completely different.",
             "#FFD54F"),

            ("IDF Weighting",
             "Inverse Document Frequency: penalizes popular movies.\n"
             "  IDF(movie) = log(TotalUsers / UsersWhoWatchedIt)\n"
             "Rare movies receive higher weight, surfacing niche recommendations.",
             "#FFD54F"),
        ]

        for title, body, color in sections:
            sec_color = t["accent"] if color == "#00E5FF" else color
            title_lbl = ctk.CTkLabel(
                scroll, text=title,
                font=ctk.CTkFont(size=13, weight="bold"),
                text_color=sec_color
            )
            title_lbl.pack(anchor="w", padx=10, pady=(12, 2))

            if body:
                body_lbl = ctk.CTkLabel(
                    scroll, text=body,
                    font=ctk.CTkFont(family="Courier", size=11),
                    text_color=t["text_primary"],
                    justify="left", wraplength=780
                )
                body_lbl.pack(anchor="w", padx=20, pady=(0, 4))

        # Close button
        btn_close = ctk.CTkButton(self, text="Close Guide", command=self.destroy, fg_color=t["btn_primary"], width=140)
        btn_close.pack(pady=10)

if __name__ == "__main__":
    app = App()
    app.mainloop()
