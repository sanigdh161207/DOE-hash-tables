import tkinter as tk
from tkinter import messagebox
import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time
import random
from typing import Dict, Any, List, Optional, Tuple

# Import custom modules
from data_generator import generate_user_data, format_first_n_users
from recommender import HashTable, RecommenderSystem
from simulator import PerformanceSimulator
import graphs

# Set appearance mode and default color theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # Configure window settings
        self.title("Hash Table Recommender Simulator")
        self.geometry("1200x750")
        self.resizable(True, True)
        
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
        
        # Create UI layout
        self.setup_ui()
        
        # Initial display
        self.write_console("Welcome to Hash Table Recommender Simulator!\nSelect a dataset size and click 'Generate Dataset' to start.")

    def setup_ui(self):
        # Master grid configuration: 2 columns (Sidebar and Content Area)
        self.grid_columnconfigure(0, weight=1)  # Left sidebar (width ~300px)
        self.grid_columnconfigure(1, weight=3)  # Right panel (width ~900px)
        self.grid_rowconfigure(0, weight=1)
        
        # --- LEFT SIDEBAR FRAME ---
        self.sidebar_frame = ctk.CTkFrame(self, width=280, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        self.sidebar_frame.grid_propagate(False)
        
        # Title Header
        self.app_title = ctk.CTkLabel(self.sidebar_frame, text="SIMULATOR CONTROLS", font=ctk.CTkFont(size=16, weight="bold"))
        self.app_title.pack(pady=(20, 10), padx=20, anchor="w")
        
        # Mode Selection
        self.mode_label = ctk.CTkLabel(self.sidebar_frame, text="Experiment Mode:", font=ctk.CTkFont(size=12, weight="bold"))
        self.mode_label.pack(padx=20, pady=(5, 3), anchor="w")
        self.mode_segmented = ctk.CTkSegmentedButton(
            self.sidebar_frame,
            values=["Reproducible Experiment", "Fresh Random Dataset"],
            command=self.on_mode_change
        )
        self.mode_segmented.set("Reproducible Experiment")
        self.mode_segmented.pack(padx=20, pady=(0, 12), fill="x")

        # 1. Dataset Size Selection
        self.ds_label = ctk.CTkLabel(self.sidebar_frame, text="Dataset Size (Number of Users):", font=ctk.CTkFont(size=12, weight="bold"))
        self.ds_label.pack(padx=20, pady=(5, 5), anchor="w")
        self.ds_option = ctk.CTkOptionMenu(self.sidebar_frame, values=["10", "50", "100", "500", "1000"], command=self.on_dataset_size_change)
        self.ds_option.set("100")
        self.ds_option.pack(padx=20, pady=(0, 15), fill="x")
        
        # 2. Table Size Selection Mode
        self.ts_label = ctk.CTkLabel(self.sidebar_frame, text="Table Size Selection:", font=ctk.CTkFont(size=12, weight="bold"))
        self.ts_label.pack(padx=20, pady=(10, 5), anchor="w")
        self.ts_mode = ctk.CTkSegmentedButton(self.sidebar_frame, values=["Auto", "Custom"], command=self.on_table_size_mode_change)
        self.ts_mode.set("Auto")
        self.ts_mode.pack(padx=20, pady=(0, 10), fill="x")
        
        # Custom Table Size Entry
        self.custom_ts_entry = ctk.CTkEntry(self.sidebar_frame, placeholder_text="Enter Table Size")
        self.custom_ts_entry.insert(0, "150")
        
        # Collision Strategy Selection
        self.strat_label = ctk.CTkLabel(self.sidebar_frame, text="Collision Strategy:", font=ctk.CTkFont(size=12, weight="bold"))
        self.strat_label.pack(padx=20, pady=(10, 5), anchor="w")
        self.strat_segmented = ctk.CTkSegmentedButton(self.sidebar_frame, values=["Separate Chaining", "Linear Probing"], command=self.on_strategy_change)
        self.strat_segmented.set("Separate Chaining")
        self.strat_segmented.pack(padx=20, pady=(0, 15), fill="x")

        # 3. Load Factor Target
        self.lf_label = ctk.CTkLabel(self.sidebar_frame, text="Target Load Factor:", font=ctk.CTkFont(size=12, weight="bold"))
        self.lf_label.pack(padx=20, pady=(10, 5), anchor="w")
        self.lf_segmented = ctk.CTkSegmentedButton(self.sidebar_frame, values=["0.25", "0.50", "0.75", "0.90"], command=self.on_load_factor_change)
        self.lf_segmented.set("0.75")
        self.lf_segmented.pack(padx=20, pady=(0, 20), fill="x")
        
        # Separator line
        self.sep = ctk.CTkFrame(self.sidebar_frame, height=2, fg_color="#333333")
        self.sep.pack(padx=20, pady=10, fill="x")
        
        # Action Buttons Scroll Frame
        self.btn_scroll = ctk.CTkScrollableFrame(self.sidebar_frame, fg_color="transparent", height=320)
        self.btn_scroll.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Dynamic Configuration Inputs
        self.sizes_lbl = ctk.CTkLabel(self.btn_scroll, text="Benchmark Sizes (csv):", font=ctk.CTkFont(size=10, weight="bold"))
        self.sizes_lbl.pack(anchor="w", padx=5, pady=(2, 0))
        self.sizes_entry = ctk.CTkEntry(self.btn_scroll, font=ctk.CTkFont(size=10))
        self.sizes_entry.insert(0, "10, 50, 100, 500, 1000")
        self.sizes_entry.pack(fill="x", padx=5, pady=(0, 5))
        
        self.lfs_lbl = ctk.CTkLabel(self.btn_scroll, text="Collision Load Factors (csv):", font=ctk.CTkFont(size=10, weight="bold"))
        self.lfs_lbl.pack(anchor="w", padx=5, pady=(2, 0))
        self.lfs_entry = ctk.CTkEntry(self.btn_scroll, font=ctk.CTkFont(size=10))
        self.lfs_entry.insert(0, "0.25, 0.50, 0.75, 0.90")
        self.lfs_entry.pack(fill="x", padx=5, pady=(0, 5))
        
        self.numpy_size_lbl = ctk.CTkLabel(self.btn_scroll, text="NumPy Benchmark Size:", font=ctk.CTkFont(size=10, weight="bold"))
        self.numpy_size_lbl.pack(anchor="w", padx=5, pady=(2, 0))
        self.numpy_size_entry = ctk.CTkEntry(self.btn_scroll, font=ctk.CTkFont(size=10))
        self.numpy_size_entry.insert(0, "5000")
        self.numpy_size_entry.pack(fill="x", padx=5, pady=(0, 10))

        self.bench_strat_lbl = ctk.CTkLabel(self.btn_scroll, text="Benchmark Mode:", font=ctk.CTkFont(size=10, weight="bold"))
        self.bench_strat_lbl.pack(anchor="w", padx=5, pady=(2, 0))
        self.bench_strat_menu = ctk.CTkOptionMenu(self.btn_scroll, values=["Separate Chaining", "Linear Probing", "Both"], font=ctk.CTkFont(size=10))
        self.bench_strat_menu.set("Both")
        self.bench_strat_menu.pack(fill="x", padx=5, pady=(0, 10))

        # Buttons
        self.btn_gen_ds = ctk.CTkButton(self.btn_scroll, text="Generate Dataset", command=self.click_generate_dataset)
        self.btn_gen_ds.pack(pady=5, fill="x")
        
        self.btn_insert = ctk.CTkButton(self.btn_scroll, text="Insert Into Hash Table", command=self.click_insert_table)
        self.btn_insert.pack(pady=5, fill="x")
        
        self.btn_lookup = ctk.CTkButton(self.btn_scroll, text="Run Lookup Benchmark", command=self.click_run_lookup_benchmark)
        self.btn_lookup.pack(pady=5, fill="x")
        
        self.btn_collision = ctk.CTkButton(self.btn_scroll, text="Run Collision Test", command=self.click_run_collision_test)
        self.btn_collision.pack(pady=5, fill="x")
        
        self.btn_numpy = ctk.CTkButton(self.btn_scroll, text="Compare NumPy", command=self.click_compare_numpy)
        self.btn_numpy.pack(pady=5, fill="x")
        
        self.btn_week6 = ctk.CTkButton(self.btn_scroll, text="Week 6: Cosine Recs", fg_color="#1565C0", hover_color="#0D47A1", command=self.click_run_week6)
        self.btn_week6.pack(pady=5, fill="x")
        
        self.btn_visualize = ctk.CTkButton(self.btn_scroll, text="Show Hash Table", command=self.click_show_hash_table)
        self.btn_visualize.pack(pady=5, fill="x")
        
        self.btn_graphs = ctk.CTkButton(self.btn_scroll, text="Generate Graphs", command=self.click_generate_graphs)
        self.btn_graphs.pack(pady=5, fill="x")
        
        self.btn_how_it_works = ctk.CTkButton(self.btn_scroll, text="How It Works (Edu)", fg_color="#2E7D32", hover_color="#1B5E20", command=self.click_how_it_works)
        self.btn_how_it_works.pack(pady=5, fill="x")
        
        self.btn_reset = ctk.CTkButton(self.btn_scroll, text="Reset Simulator", fg_color="#C62828", hover_color="#B71C1C", command=self.click_reset)
        self.btn_reset.pack(pady=(15, 5), fill="x")
        
        # --- RIGHT MAIN AREA ---
        self.right_frame = ctk.CTkFrame(self, fg_color="#1A1A1A", corner_radius=0)
        self.right_frame.grid(row=0, column=1, sticky="nsew", padx=0, pady=0)
        
        # Configure layout for main panel
        self.right_frame.grid_columnconfigure(0, weight=1)
        self.right_frame.grid_rowconfigure(0, weight=1)  # Visual/Graph viewport
        self.right_frame.grid_rowconfigure(1, weight=1)  # Animation & Stats
        self.right_frame.grid_rowconfigure(2, weight=1)  # Console Output
        
        # Frame 1: Tabview/Content (Visualizer / Graphs / Benchmark results)
        self.content_tabview = ctk.CTkTabview(self.right_frame)
        self.content_tabview.grid(row=0, column=0, sticky="nsew", padx=15, pady=(15, 5))
        
        self.tab_visualizer = self.content_tabview.add("Hash Table Visualizer")
        self.tab_graphs = self.content_tabview.add("Performance Charts")
        
        # Set up Tab Visualizer Sub-layout
        self.tab_visualizer.grid_columnconfigure(0, weight=1)
        self.tab_visualizer.grid_rowconfigure(0, weight=1)
        self.visualizer_scroll = ctk.CTkScrollableFrame(self.tab_visualizer, fg_color="#242424")
        self.visualizer_scroll.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # Add Search/Interactive section inside the visualizer tab
        self.search_control_frame = ctk.CTkFrame(self.tab_visualizer, height=50)
        self.search_control_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=(0, 5))
        
        self.search_lbl = ctk.CTkLabel(self.search_control_frame, text="Lookup/Search User ID:")
        self.search_lbl.pack(side="left", padx=10)
        self.search_entry = ctk.CTkEntry(self.search_control_frame, width=150, placeholder_text="e.g. 1005")
        self.search_entry.pack(side="left", padx=5)
        self.btn_search = ctk.CTkButton(self.search_control_frame, text="Search & Animate", command=self.click_search_animate, width=120)
        self.btn_search.pack(side="left", padx=10)
        
        # Tab Graphs Sub-layout
        self.tab_graphs.grid_columnconfigure(0, weight=1)
        self.tab_graphs.grid_columnconfigure(1, weight=1)
        self.tab_graphs.grid_columnconfigure(2, weight=1)
        self.tab_graphs.grid_rowconfigure(0, weight=1)
        
        self.graph_frame_1 = ctk.CTkFrame(self.tab_graphs, fg_color="#242424")
        self.graph_frame_1.grid(row=0, column=0, sticky="nsew", padx=3, pady=5)
        self.graph_lbl_1 = ctk.CTkLabel(self.graph_frame_1, text="Lookup Performance Chart\n(Select 'Generate Graphs' to render)", font=ctk.CTkFont(size=10, slant="italic"))
        self.graph_lbl_1.pack(expand=True)
        
        self.graph_frame_2 = ctk.CTkFrame(self.tab_graphs, fg_color="#242424")
        self.graph_frame_2.grid(row=0, column=1, sticky="nsew", padx=3, pady=5)
        self.graph_lbl_2 = ctk.CTkLabel(self.graph_frame_2, text="Collision Rate Chart\n(Select 'Generate Graphs' to render)", font=ctk.CTkFont(size=10, slant="italic"))
        self.graph_lbl_2.pack(expand=True)

        self.graph_frame_3 = ctk.CTkFrame(self.tab_graphs, fg_color="#242424")
        self.graph_frame_3.grid(row=0, column=2, sticky="nsew", padx=3, pady=5)
        self.graph_lbl_3 = ctk.CTkLabel(self.graph_frame_3, text="Load Factor vs Lookup\n(Select 'Generate Graphs' to render)", font=ctk.CTkFont(size=10, slant="italic"))
        self.graph_lbl_3.pack(expand=True)
        
        # Frame 2: Step-by-Step Animation Details & Real-Time Stats
        self.middle_frame = ctk.CTkFrame(self.right_frame, height=130)
        self.middle_frame.grid(row=1, column=0, sticky="nsew", padx=15, pady=5)
        self.middle_frame.grid_propagate(False)
        self.middle_frame.grid_columnconfigure(0, weight=2)
        self.middle_frame.grid_columnconfigure(1, weight=1)
        self.middle_frame.grid_rowconfigure(0, weight=1)
        
        # Animation Steps Panel
        self.anim_panel = ctk.CTkFrame(self.middle_frame, fg_color="#242424")
        self.anim_panel.grid(row=0, column=0, sticky="nsew", padx=(5, 2), pady=5)
        
        self.anim_title = ctk.CTkLabel(self.anim_panel, text="Operation Step Trace", font=ctk.CTkFont(size=12, weight="bold"), text_color=graphs.PRIMARY_BLUE)
        self.anim_title.pack(anchor="w", padx=10, pady=2)
        
        self.anim_text = ctk.CTkLabel(self.anim_panel, text="No ongoing operation.\nUse search or insert to trace execution steps.", justify="left", font=ctk.CTkFont(size=11))
        self.anim_text.pack(fill="both", expand=True, padx=10, pady=(0, 5))
        
        # Stats panel
        self.stats_panel = ctk.CTkFrame(self.middle_frame, fg_color="#242424")
        self.stats_panel.grid(row=0, column=1, sticky="nsew", padx=(2, 5), pady=5)
        
        self.stats_title = ctk.CTkLabel(self.stats_panel, text="Hash Table Statistics", font=ctk.CTkFont(size=12, weight="bold"), text_color=graphs.PRIMARY_BLUE)
        self.stats_title.grid(row=0, column=0, columnspan=2, sticky="w", padx=10, pady=2)
        
        # Create labels for statistics grid
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
            
        # Frame 3: Output Console View
        self.console_frame = ctk.CTkFrame(self.right_frame, height=180)
        self.console_frame.grid(row=2, column=0, sticky="nsew", padx=15, pady=(5, 15))
        self.console_frame.grid_propagate(False)
        
        self.console_title = ctk.CTkLabel(self.console_frame, text="Output Console Log", font=ctk.CTkFont(size=12, weight="bold"))
        self.console_title.pack(anchor="w", padx=10, pady=2)
        
        self.console_textbox = ctk.CTkTextbox(self.console_frame, state="disabled", font=ctk.CTkFont(family="Courier", size=11))
        self.console_textbox.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    # --- UI INTERACTION & CALLBACKS ---
    
    def write_console(self, text: str, clear: bool = False):
        self.console_textbox.configure(state="normal")
        if clear:
            self.console_textbox.delete("1.0", "end")
        self.console_textbox.insert("end", text + "\n")
        self.console_textbox.see("end")
        self.console_textbox.configure(state="disabled")
        
    def update_stats_ui(self):
        if not self.hash_table:
            return
        stats = self.hash_table.get_collision_statistics()
        self.stats_labels["Users:"].configure(text=str(stats["users"]))
        self.stats_labels["Table Size:"].configure(text=str(stats["table_size"]))
        self.stats_labels["Collisions:"].configure(text=str(stats["collisions"]))
        self.stats_labels["Load Factor:"].configure(text=f"{stats['load_factor']:.2f}")
        if hasattr(self.hash_table, "TOMBSTONE"):
            self.stats_labels["Avg Bucket Len:"].configure(text="N/A")
            self.stats_labels["Max Bucket Len:"].configure(text="N/A")
        else:
            self.stats_labels["Avg Bucket Len:"].configure(text=f"{stats['avg_bucket_length']:.2f}")
            self.stats_labels["Max Bucket Len:"].configure(text=str(stats["max_bucket_length"]))

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
            # Display entry field for custom size
            self.custom_ts_entry.pack(padx=20, pady=(0, 10), fill="x")
        else:
            self.custom_ts_entry.pack_forget()
            
    def on_load_factor_change(self, choice):
        self.selected_load_factor = float(choice)
        self.write_console(f"Target Load Factor configured to {self.selected_load_factor}.")

    # --- SIMULATOR ACTIONS ---
    
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
            # Select prime/odd size to balance load factor
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
                
        from recommender import HashTableChaining, HashTableLinearProbing
        strat_class = HashTableChaining if self.selected_strategy == "Separate Chaining" else HashTableLinearProbing
        self.hash_table = strat_class(size)
        self.recommender = RecommenderSystem(size, strategy_class=strat_class)
        
        self.write_console(f"Inserting {len(self.dataset)} records into Hash Table...")
        
        # We can animate insertion for a few steps and run the rest instantaneously
        self.anim_title.configure(text="Insertion Animation & Trace")
        
        # Let's perform step-by-step trace simulation for the first element
        first_user = self.dataset[0]
        trace = self.hash_table.insert(first_user[0], first_user[1])
        
        def run_insertion_trace(step_idx=0):
            if step_idx < len(trace):
                step = trace[step_idx]
                if step["step"] == "input":
                    self.anim_text.configure(text=f"Inserting Key (User ID): {step['val']}\nWaiting to process hash function...")
                elif step["step"] == "hash":
                    self.anim_text.configure(text=f"Computing Hash Index:\nFormula: {step['formula']}\nIndex = {step['index']}")
                    self.highlight_visualizer_bucket(step["index"])
                elif step["step"] == "traverse":
                    self.anim_text.configure(text=f"Traversing Chaining Index {step['index']}.\nExisting elements: {step['bucket']}\nOperation: Adding user to chaining list.")
                elif step["step"] == "probe":
                    action_str = "Inserting here!" if step["action"] == "inserted" else "Probing next index..."
                    self.anim_text.configure(text=f"Probing Linear Index {step['index']}: slot is {step['state']}.\nProbe count: {step['probes']}\nAction: {action_str}")
                    self.highlight_visualizer_bucket(step["index"])
                elif step["step"] == "done":
                    self.anim_text.configure(text=f"Successfully inserted User into index {step['index']}!")
                    self.highlight_visualizer_bucket(step["index"])
                self.after(800, lambda: run_insertion_trace(step_idx + 1))
            else:
                # Insert the remaining dataset
                for user_id, movies in self.dataset[1:]:
                    self.hash_table.insert(user_id, movies)
                self.recommender.fit(self.dataset, movie_vocab=list(range(1, 101)))
                
                self.write_console(f"Successfully inserted all {len(self.dataset)} users into Hash Table.")
                self.update_stats_ui()
                self.click_show_hash_table()  # Refresh visual representation
                
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
            "Higher Load Factor\n"
            "  ↓\n"
            "Less Empty Space\n"
            "  ↓\n"
            "More Collisions\n"
            "  ↓\n"
            "Slightly Slower Search due to longer chain traversal or probing steps."
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
        is_reproducible = (self.experiment_mode == "Reproducible Experiment")
        mode_title = "MODE A — REPRODUCIBLE WEEK 6 EXPERIMENT" if is_reproducible else "MODE B — FRESH DATA DEMO"
        
        if not self.dataset or not self.recommender:
            seed = 42 if is_reproducible else None
            self.write_console("\n" + "="*70)
            self.write_console(f"Running {mode_title} (Standalone N=500, M=100)...")
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
        self.write_console(f"{mode_title} (CURRENT HASH TABLE - {n_users} Users)")
        self.write_console("="*70)
        self.update()

        # Benchmark currently loaded users
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

        # Pick 3 representative users from the CURRENT dataset
        sample_indices = [0, n_users // 2, n_users - 1] if n_users >= 3 else list(range(n_users))
        sample_indices = list(dict.fromkeys(sample_indices))

        label_context = "Same 3 Benchmark Users" if is_reproducible else "3 Newly Sampled Users"
        self.write_console(f"\n--- Evaluation of {label_context} ---")
        self.write_console("-" * 70)
        for idx in sample_indices:
            target_uid, original_prefs = self.dataset[idx]
            b_recs, b_timing, _ = self.recommender.recommend_movies_baseline(target_uid, top_n=5)
            c_recs, c_timing, c_neighbors, _ = self.recommender.recommend_movies_cosine(target_uid, top_n=5, top_k_users=10)
            
            self.write_console(f"User ID: {target_uid} (Visible in Hash Table)")
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
        
        # Clear visualizer scroll frame
        for widget in self.visualizer_scroll.winfo_children():
            widget.destroy()
            
        # Draw the buckets
        max_display = 25
        limit = min(self.hash_table.size, max_display)
        
        for idx in range(limit):
            bucket = self.hash_table.buckets[idx]
            is_collision = len(bucket) > 1
            border_color = graphs.SECONDARY_ACCENT if is_collision else "#333333"
            
            # Row container for this index
            row_frame = ctk.CTkFrame(self.visualizer_scroll, fg_color="#1E1E1E", border_color=border_color, border_width=1.5, height=55)
            row_frame.pack(fill="x", padx=5, pady=3)
            row_frame.pack_propagate(False)
            
            # Index header block
            idx_lbl = ctk.CTkLabel(row_frame, text=f"Index {idx:02d}", width=80, font=ctk.CTkFont(weight="bold"), text_color=graphs.PRIMARY_BLUE)
            idx_lbl.pack(side="left", padx=10)
            
            # Initial link arrow
            arrow_lbl = ctk.CTkLabel(row_frame, text="➔", font=ctk.CTkFont(size=14))
            arrow_lbl.pack(side="left", padx=5)
            
            # Content container within the row
            content_container = ctk.CTkFrame(row_frame, fg_color="transparent")
            content_container.pack(side="left", fill="both", expand=True, padx=5)
            
            if not bucket:
                # Dimmed card for empty index
                empty_pill = ctk.CTkFrame(content_container, fg_color="#242424", border_color="#333333", border_width=1, corner_radius=6, height=35)
                empty_pill.pack(side="left", pady=10)
                empty_lbl = ctk.CTkLabel(empty_pill, text="[ Empty ]", font=ctk.CTkFont(size=10, slant="italic"), text_color="#666666")
                empty_lbl.pack(padx=10, pady=5)
            else:
                for i, (k, v) in enumerate(bucket):
                    # Pill frame for user node
                    node_color = "#3A3A3A" if not is_collision else "#3E2A1C"
                    node_border = "#555555" if not is_collision else graphs.SECONDARY_ACCENT
                    
                    pill = ctk.CTkFrame(content_container, fg_color=node_color, border_color=node_border, border_width=1, corner_radius=6, height=38)
                    pill.pack(side="left", pady=8)
                    pill.pack_propagate(False)
                    
                    # Inside pill: User details
                    txt_lbl = ctk.CTkLabel(pill, text=f"User {k}", font=ctk.CTkFont(weight="bold", size=10), text_color=graphs.TEXT_COLOR)
                    txt_lbl.pack(padx=8, pady=(2, 0), anchor="w")
                    
                    sub_lbl = ctk.CTkLabel(pill, text=f"{len(v)} items", font=ctk.CTkFont(size=8), text_color="#A0A0A0")
                    sub_lbl.pack(padx=8, pady=(0, 2), anchor="w")
                    
                    # If not the last element in the bucket chain, show a connector arrow
                    if i < len(bucket) - 1:
                        conn_lbl = ctk.CTkLabel(content_container, text=" ➔ ", font=ctk.CTkFont(size=12), text_color=graphs.PRIMARY_BLUE)
                        conn_lbl.pack(side="left", padx=3)
            
        if self.hash_table.size > max_display:
            more_lbl = ctk.CTkLabel(self.visualizer_scroll, text=f"... and {self.hash_table.size - max_display} more indices ...", font=ctk.CTkFont(slant="italic"))
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
            
        # Get trace and perform lookup animation
        val, trace = self.hash_table.search(search_key)
        self.anim_title.configure(text="Search/Lookup Animation Trace")
        
        self.content_tabview.set("Hash Table Visualizer")
        
        def run_search_trace(step_idx=0):
            if step_idx < len(trace):
                step = trace[step_idx]
                if step["step"] == "input":
                    self.anim_text.configure(text=f"Searching for User ID: {step['val']}\nInitializing hash lookup...", text_color=graphs.TEXT_COLOR)
                elif step["step"] == "hash":
                    self.anim_text.configure(text=f"Hashing Key to Index:\nFormula: {step['formula']}\nTarget Index = {step['index']}", text_color=graphs.PRIMARY_BLUE)
                    self.highlight_visualizer_bucket(step["index"])
                elif step["step"] == "compare":
                    matched_str = "MATCHED!" if step["matched"] else "no match"
                    self.anim_text.configure(text=f"Searching chaining list in Index {step['index']}:\nElement position {step['bucket_idx']}: Key {step['current_key']} vs Target {step['target_key']} -> {matched_str}", text_color=graphs.SECONDARY_ACCENT if step["matched"] else graphs.TEXT_COLOR)
                elif step["step"] == "probe_search":
                    matched_str = "MATCHED!" if step["matched"] else "no match"
                    self.anim_text.configure(text=f"Probing Linear Index {step['index']}: slot is {step['state']}.\nProbe count: {step['probes']}\nMatch: {matched_str}", text_color=graphs.SECONDARY_ACCENT if step["matched"] else graphs.TEXT_COLOR)
                    self.highlight_visualizer_bucket(step["index"])
                elif step["step"] == "done":
                    if step["found"]:
                        self.anim_text.configure(text=f"SUCCESS: Found User ID {search_key}!\nMovies liked: {step['value']}", text_color="#2E7D32")
                        self.write_console(f"Search Success: User {search_key} prefers movies: {step['value']}")
                        if self.recommender:
                            b_recs, b_t, _ = self.recommender.recommend_movies_baseline(search_key, top_n=3)
                            c_recs, c_t, c_sims, _ = self.recommender.recommend_movies_cosine(search_key, top_n=3)
                            top_sim_info = f", Top Neighbor: {c_sims[0][0]} ({c_sims[0][1]:.2f})" if c_sims else ""
                            self.write_console(f"  * Baseline Recs: {b_recs} (Total: {b_t['total_time']*1000:.3f} ms)")
                            self.write_console(f"  * Cosine Recs:   {c_recs} (Total: {c_t['total_time']*1000:.3f} ms{top_sim_info})")
                    else:
                        self.anim_text.configure(text=f"FAILURE: User ID {search_key} not present in hash table.", text_color="#C62828")
                        self.write_console(f"Search Failure: User {search_key} not found.")
                self.after(900, lambda: run_search_trace(step_idx + 1))
                
        run_search_trace(0)

    def highlight_visualizer_bucket(self, index: int):
        # Visually highlight the row in the scrollable view
        # We can scan the children of visualizer_scroll
        children = self.visualizer_scroll.winfo_children()
        if index < len(children) and index >= 0:
            row_frame = children[index]
            if isinstance(row_frame, ctk.CTkFrame):
                # Flash border color
                original_color = row_frame.cget("border_color")
                row_frame.configure(border_color="#00E5FF")
                # Reset after 1200ms
                self.after(1200, lambda: row_frame.configure(border_color=original_color))

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
        
        # Clear existing content from graph frames
        for f in [self.graph_frame_1, self.graph_frame_2, self.graph_frame_3]:
            for w in f.winfo_children():
                w.destroy()
                
        # 1. Lookup Time vs Size
        lookup_data = PerformanceSimulator.run_lookup_benchmark(sizes, strategy=mode)
        # If lookup_data only has one strategy, extract the list for backwards compatibility inside graph functions
        lookup_plot_data = lookup_data if mode == "Both" else lookup_data[mode]
        fig1 = graphs.plot_lookup_benchmark(lookup_plot_data)
        canvas1 = FigureCanvasTkAgg(fig1, master=self.graph_frame_1)
        canvas1.draw()
        canvas1.get_tk_widget().pack(fill="both", expand=True)
        
        # 2. Collision rate vs Load Factor
        collision_data = PerformanceSimulator.run_collision_experiment(lfs, strategy=mode)
        collision_plot_data = collision_data if mode == "Both" else collision_data[mode]
        fig2 = graphs.plot_collision_experiment(collision_plot_data)
        canvas2 = FigureCanvasTkAgg(fig2, master=self.graph_frame_2)
        canvas2.draw()
        canvas2.get_tk_widget().pack(fill="both", expand=True)
        
        # 3. Load Factor vs Lookup Time (Comparison plot)
        if mode == "Both":
            fig3 = graphs.plot_load_factor_vs_lookup_time(collision_data)
            canvas3 = FigureCanvasTkAgg(fig3, master=self.graph_frame_3)
            canvas3.draw()
            canvas3.get_tk_widget().pack(fill="both", expand=True)
        else:
            # Display a helper message in frame 3 if not in comparison mode
            lbl = ctk.CTkLabel(self.graph_frame_3, text="Load Factor vs Lookup Chart\n(Select 'Both' mode to render comparison)", font=ctk.CTkFont(size=10, slant="italic"))
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
        self.anim_text.configure(text="No ongoing operation.\nUse search or insert to trace execution steps.", text_color=graphs.TEXT_COLOR)
        self.anim_title.configure(text="Operation Step Trace")
        
        # Reset stats labels
        for k in self.stats_labels:
            self.stats_labels[k].configure(text="0" if k != "Load Factor:" and k != "Avg Bucket Len:" else "0.0")
            
        # Clear visualizer content
        for widget in self.visualizer_scroll.winfo_children():
            widget.destroy()
            
        # Reset graphs frames
        for f, lbl in [(self.graph_frame_1, "Lookup Performance Chart\n(Select 'Generate Graphs' to render)"), 
                        (self.graph_frame_2, "Collision Rate Chart\n(Select 'Generate Graphs' to render)"),
                        (self.graph_frame_3, "Load Factor vs Lookup Chart\n(Select 'Generate Graphs' to render)")]:
            for w in f.winfo_children():
                w.destroy()
            ctk.CTkLabel(f, text=lbl, font=ctk.CTkFont(size=10, slant="italic")).pack(expand=True)


class EduPopup(ctk.CTkToplevel):
    """
    Educational popup window explaining Hash Table concepts visually.
    """
    def __init__(self, parent):
        super().__init__(parent)
        self.title("How it Works: Hash Table Visual Guide")
        self.geometry("700x550")
        self.resizable(False, False)
        
        # Grid layout
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # Scrollable frame for all concepts
        scroll = ctk.CTkScrollableFrame(self, fg_color="#1A1A1A")
        scroll.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        # 1. HASH FUNCTION
        f_hash = ctk.CTkFrame(scroll, fg_color="#242424", border_color="#333333", border_width=1)
        f_hash.pack(fill="x", padx=10, pady=10)
        lbl_h = ctk.CTkLabel(f_hash, text="1. The Hash Function", font=ctk.CTkFont(size=14, weight="bold"), text_color=graphs.PRIMARY_BLUE)
        lbl_h.pack(anchor="w", padx=15, pady=(10, 5))
        
        diag_h = (
            "A Hash Function converts an arbitrary key (like a User ID) into an index\n"
            "within the memory bounds of the array.\n\n"
            "  User ID               Hash Formula               Memory Index\n"
            " ┌───────┐          ┌───────────────────┐          ┌──────────────┐\n"
            " │  105  │  ──────➔  │  105 % TableSize  │  ──────➔  │      5       │\n"
            " └───────┘          └───────────────────┘          └──────────────┘"
        )
        lbl_diag_h = ctk.CTkLabel(f_hash, text=diag_h, font=ctk.CTkFont(family="Courier", size=11), justify="left")
        lbl_diag_h.pack(anchor="w", padx=25, pady=(0, 15))

        # 2. COLLISIONS & CHAINING
        f_coll = ctk.CTkFrame(scroll, fg_color="#242424", border_color="#333333", border_width=1)
        f_coll.pack(fill="x", padx=10, pady=10)
        lbl_c = ctk.CTkLabel(f_coll, text="2. Hashing Collisions & Chaining", font=ctk.CTkFont(size=14, weight="bold"), text_color=graphs.PRIMARY_BLUE)
        lbl_c.pack(anchor="w", padx=15, pady=(10, 5))
        
        diag_c = (
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
        )
        lbl_diag_c = ctk.CTkLabel(f_coll, text=diag_c, font=ctk.CTkFont(family="Courier", size=11), justify="left")
        lbl_diag_c.pack(anchor="w", padx=25, pady=(0, 15))

        # 3. LOAD FACTOR
        f_lf = ctk.CTkFrame(scroll, fg_color="#242424", border_color="#333333", border_width=1)
        f_lf.pack(fill="x", padx=10, pady=10)
        lbl_lf = ctk.CTkLabel(f_lf, text="3. Load Factor", font=ctk.CTkFont(size=14, weight="bold"), text_color=graphs.PRIMARY_BLUE)
        lbl_lf.pack(anchor="w", padx=15, pady=(10, 5))
        
        diag_lf = (
            "Load Factor represents how full the hash table is:\n\n"
            "               Total Inserted Keys (N)\n"
            " Load Factor = ───────────────────────\n"
            "                  Table Size (M)\n\n"
            "  - Higher Load Factor   = More full table, higher probability of collisions.\n"
            "  - Optimal Load Factor  = Typically between 0.70 and 0.75."
        )
        lbl_diag_lf = ctk.CTkLabel(f_lf, text=diag_lf, font=ctk.CTkFont(family="Courier", size=11), justify="left")
        lbl_diag_lf.pack(anchor="w", padx=25, pady=(0, 15))

        # 4. REHASHING
        f_re = ctk.CTkFrame(scroll, fg_color="#242424", border_color="#333333", border_width=1)
        f_re.pack(fill="x", padx=10, pady=10)
        lbl_re = ctk.CTkLabel(f_re, text="4. Rehashing", font=ctk.CTkFont(size=14, weight="bold"), text_color=graphs.PRIMARY_BLUE)
        lbl_re.pack(anchor="w", padx=15, pady=(10, 5))
        
        diag_re = (
            "When the load factor exceeds a threshold, the table undergoes rehashing:\n\n"
            " Double Table Size      Recompute indices for       Collisions and\n"
            " (M ──➔ 2 * M)      ──➔ all existing keys       ──➔ chain lengths\n"
            "                        with new Table Size          decrease!"
        )
        lbl_diag_re = ctk.CTkLabel(f_re, text=diag_re, font=ctk.CTkFont(family="Courier", size=11), justify="left")
        lbl_diag_re.pack(anchor="w", padx=25, pady=(0, 15))
        
        # Close Button
        btn_close = ctk.CTkButton(self, text="Close Guide", command=self.destroy, fg_color=graphs.PRIMARY_BLUE)
        btn_close.grid(row=1, column=0, pady=10)

if __name__ == "__main__":
    app = App()
    app.mainloop()
