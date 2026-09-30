# Week 8 — HOT1 & HOT2 Reference Guide

---

## 1. HOT1 — Sparse Memory Derivation & Handout Analysis

**Domain**: $U = 1,000,000$ Users $\times M = 100,000$ Movies ($100,000,000,000$ cells)

### Dense Float64 Matrix
$$\text{Memory}_{\text{dense}} = 100,000,000,000 \times 8 \text{ bytes} = 800,000,000,000 \text{ bytes} \approx 745.06 \text{ GiB}$$

### Sparse CSR Representation Breakdown
CSR consists of three arrays: `data` (float64), `indices` (int32), and `indptr` (int32).

$$\text{Memory}_{\text{CSR}} = NNZ \times 8 + NNZ \times 4 + (U + 1) \times 4 \text{ bytes}$$

- **For 0.01% Density ($D = 0.0001$, $NNZ = 10,000,000$)**:
  - `data`: $10,000,000 \times 8 = 80,000,000$ bytes ($76.29$ MB)
  - `indices`: $10,000,000 \times 4 = 40,000,000$ bytes ($38.15$ MB)
  - `indptr`: $1,000,001 \times 4 = 4,000,004$ bytes ($3.81$ MB)
  - **Total Footprint**: $124,000,004$ bytes $\approx \mathbf{124 \text{ MB}}$ ($\mathbf{118.26 \text{ MiB}}$)

- **For 0.001% Density ($D = 0.00001$, $NNZ = 1,000,000$)**:
  - `data`: $1,000,000 \times 8 = 8,000,000$ bytes ($7.63$ MB)
  - `indices`: $1,000,000 \times 4 = 4,000,000$ bytes ($3.81$ MB)
  - `indptr`: $1,000,001 \times 4 = 4,000,004$ bytes ($3.81$ MB)
  - **Total Footprint**: $16,000,004$ bytes $\approx \mathbf{16 \text{ MB}}$ ($\mathbf{15.26 \text{ MiB}}$)

### Clarification of Handout "8 MB" Claim
The handout's "8 MB" figure represents **only the raw data payload array** ($1,000,000 \times 8 \text{ bytes} = 8\text{ MB}$) at $0.001\%$ density, ignoring the $4\text{ MB}$ `indices` and $4\text{ MB}$ `indptr` overhead. Accounting for full SciPy indexing structure yields exactly $\mathbf{16\text{ MB}}$.

---

## 2. HOT2 — Extendible Hashing Mechanics

Extendible Hashing resolves collisions dynamically without full-table array reallocation:
1. **Global Depth ($G$)**: Controls directory size ($2^G$).
2. **Local Depth ($L$)**: Specifies how many bits of hash value bucket keys share.
3. **Directory Pointer Array**: Contains $2^G$ pointers pointing to buckets.
4. **Bucket Split & Directory Doubling**: When a bucket with $L = G$ overflows, $G$ increments by 1, doubling the directory array. Only the overflowing bucket splits into two buckets with $L + 1$, leaving all other buckets unchanged.

```
Directory (G=2)            Buckets
┌───────┐                ┌───────────────────────────────┐
│  00   │  ───────────➔  │ Bucket A (L=2): [Key 4, Key 8] │
├───────┤                └───────────────────────────────┘
│  01   │  ───────────➔  ┌───────────────────────────────┐
├───────┤                │ Bucket B (L=1): [Key 1, Key 3] │
│  11   │  ───────────┘  └───────────────────────────────┘
└───────┘
```
