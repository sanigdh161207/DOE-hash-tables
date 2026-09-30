# Week 8 — Long-Term Feedback-Loop Recommendation Architecture Specification

---

## 1. Problem Statement
Static collaborative filtering recommendations suffer from staleness over time as user preferences evolve. Without explicit feedback loops (likes/dislikes) and implicit temporal decay, past interactions permanently bias future recommendations.

---

## 2. Proposed Data Model & Formula

### User Preference Vector Update Rule
For user $u$ upon receiving feedback on movie $m$ at time $t$:
$$v_{u, m}^{(t)} = v_{u, m}^{(t-1)} \cdot e^{-\lambda (t - t_{last})} + \eta \cdot \text{feedback}(m)$$

where:
- $\lambda = 0.05$: Exponential time decay constant
- $\eta = 1.0$: Feedback learning rate
- $\text{feedback}(m) = +1.0$ (Like), $-1.0$ (Dislike), $+0.2$ (Click/View)

---

## 3. Algorithm Pseudocode

```python
def update_user_feedback(user_id: int, movie_id: int, feedback_type: str, current_time: float):
    user_profile = user_hash_table.search(user_id)
    if not user_profile:
        user_profile = {"vector": {}, "last_updated": current_time}
    
    dt = current_time - user_profile["last_updated"]
    decay_factor = math.exp(-0.05 * dt)
    
    # Apply exponential decay to all existing preference weights
    for m in user_profile["vector"]:
        user_profile["vector"][m] *= decay_factor
        
    feedback_signal = 1.0 if feedback_type == "LIKE" else (-1.0 if feedback_type == "DISLIKE" else 0.2)
    user_profile["vector"][movie_id] = user_profile["vector"].get(movie_id, 0.0) + feedback_signal
    user_profile["last_updated"] = current_time
    
    user_hash_table.insert(user_id, user_profile)
```

---

## 4. Required Data, Tools & Evaluation Metrics

- **Required Data**: Interaction logs containing `(user_id, movie_id, timestamp, feedback_type)`.
- **Tools**: SciPy sparse matrices, Redis/In-Memory Hash Table for transient user vector state.
- **Evaluation Metrics**:
  - *Click-Through Rate (CTR)*
  - *Conversion / Like Ratio*
  - *Temporal Recency Sensitivity*
