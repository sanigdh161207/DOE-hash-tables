import os

def calculate_hot1_memory() -> str:
    num_users = 1_000_000
    num_items = 100_000
    total_cells = num_users * num_items

    dense_bytes = total_cells * 8
    dense_gb = dense_bytes / (1024 ** 3)

    lines = []
    lines.append("============================================================")
    lines.append("HOT1 SPARSE MEMORY CALCULATION REPORT")
    lines.append("============================================================")
    lines.append(f"Dimensions: {num_users:,} Users x {num_items:,} Items")
    lines.append(f"Total Matrix Cells: {total_cells:,} cells")
    lines.append("")
    lines.append("1. DENSE FLOAT64 MATRIX")
    lines.append("------------------------------------------------------------")
    lines.append(f"Formula: Total Cells x 8 bytes")
    lines.append(f"Exact Size: {dense_bytes:,} bytes")
    lines.append(f"Size in GiB: {dense_gb:.2f} GiB")
    lines.append("")
    lines.append("2. SPARSE CSR REPRESENTATION BREAKDOWN")
    lines.append("------------------------------------------------------------")

    densities = [0.0001, 0.00001]
    labels = ["0.01% Density (0.0001)", "0.001% Density (0.00001)"]

    for d, label in zip(densities, labels):
        nnz = int(total_cells * d)
        data_bytes = nnz * 8          # float64
        indices_bytes = nnz * 4       # int32
        indptr_bytes = (num_users + 1) * 4  # int32
        total_csr = data_bytes + indices_bytes + indptr_bytes
        total_csr_mb = total_csr / (1024 ** 2)

        lines.append(f"--- {label} ---")
        lines.append(f"Non-zero elements (NNZ): {nnz:,}")
        lines.append(f"  - data array (float64):   {data_bytes:,} bytes ({data_bytes / (1024**2):.2f} MB)")
        lines.append(f"  - indices array (int32):  {indices_bytes:,} bytes ({indices_bytes / (1024**2):.2f} MB)")
        lines.append(f"  - indptr array (int32):   {indptr_bytes:,} bytes ({indptr_bytes / (1024**2):.2f} MB)")
        lines.append(f"  Total CSR Footprint:      {total_csr:,} bytes ({total_csr_mb:.2f} MB)")
        lines.append("")

    lines.append("3. ANALYSIS OF HANDOUT '8 MB' CLAIM VS EXACT MATH")
    lines.append("------------------------------------------------------------")
    lines.append("Observation:")
    lines.append("The handout's approximate '8 MB' claim corresponds specifically to the raw")
    lines.append("data payload (1,000,000 non-zero entries * 8 bytes = 8,000,000 bytes = 8 MB)")
    lines.append("at 0.001% density, excluding structural indexing arrays (indices + indptr).")
    lines.append("When accounting for complete SciPy CSR indexing overhead:")
    lines.append("  - 0.01% density requires ~124 MB (~118.25 MiB)")
    lines.append("  - 0.001% density requires ~16 MB (~15.26 MiB)")
    lines.append("Conclusion: Sparse CSR representation achieves over 99.98% memory reduction")
    lines.append("compared to the 745 GiB dense matrix.")
    lines.append("============================================================")

    output_text = "\n".join(lines)

    os.makedirs("results", exist_ok=True)
    with open("results/hot1_memory.txt", "w", encoding="utf-8") as f:
        f.write(output_text)

    return output_text

if __name__ == "__main__":
    print(calculate_hot1_memory())
