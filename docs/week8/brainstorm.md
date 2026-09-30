# Week 8 — Technical Innovation & Domain Extension Brainstorming

---

## 1. Technical Innovations & Extensions

1. **IDF-Weighted Cosine & Popularity Damping**: Incorporates smoothed Inverse Document Frequency weights to penalize ubiquitous blockbusters and boost long-tail catalog items (*Status: Kept & Implemented*).
2. **Sparse CSR Matrix Representation**: Replaces dense 2D floating-point NumPy arrays with SciPy `csr_matrix` row-normalized dot products (*Status: Kept & Implemented*).
3. **Extendible Hashing with Dynamic Depth Splitting**: Implements bitwise global/local depth directory pointers to split overflowing buckets dynamically without full-table rehashing (*Status: Kept & Implemented*).
4. **Cold-Start Hybrid Fallback Engine**: Multi-stage fallback routing unknown users through popularity fallback or transient seed-based cosine matching (*Status: Kept & Implemented*).
5. **Inverted Index Candidate Restricted Filtering**: Maps `movie_id -> set(user_ids)` to restrict candidate neighbor evaluations to users sharing at least one common item (*Status: Kept & Implemented*).
6. **Approximate Nearest Neighbors (ANN) via MinHash LSH**: Locality-Sensitive Hashing to achieve sub-linear $O(1)$ similarity lookup (*Status: Discarded for future scope due to memory constraints*).

---

## 2. Alternative Application Domains

### Domain 1: E-Commerce Product Recommendations
- **Users**: Shoppers
- **Items**: Catalog SKUs
- **What gets hashed**: `user_id -> List[purchased_sku_ids]`
- **Recommendation signal**: Co-purchased item baskets
- **Required design changes**: Incorporate purchase recency and category affinity.

### Domain 2: Music & Audio Streaming
- **Users**: Listeners
- **Items**: Song Tracks / Artists
- **What gets hashed**: `user_id -> List[track_ids]`
- **Recommendation signal**: Playlist co-occurrence and acoustic feature similarity
- **Required design changes**: Sequential transition probability weights.

### Domain 3: Online Learning Course Recommendations
- **Users**: Students
- **Items**: Educational Courses / Modules
- **What gets hashed**: `student_id -> List[enrolled_course_ids]`
- **Recommendation signal**: Skill prerequisite graphs and completion rates
- **Required design changes**: Topological DAG ordering constraints.

### Domain 4: News & Article Personalization
- **Users**: Readers
- **Items**: News Articles
- **What gets hashed**: `reader_id -> List[article_ids]`
- **Recommendation signal**: Recency decay and topic category affinity
- **Required design changes**: Exponential time-decay factor $\mathbf{e}^{-\lambda t}$.

### Domain 5: Job Matching & Career Portal
- **Users**: Job Seekers
- **Items**: Open Job Listings
- **What gets hashed**: `user_id -> List[applied_job_ids]`
- **Recommendation signal**: Required skills matching and geographic distance
- **Required design changes**: Asymmetric matching (candidate skills vs job requirements).
