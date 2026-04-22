# Feature Engineering Agent — User Manual

## Part 1 — Concepts

This agent is designed to bridge the gap between raw data and ML models by automating the construction of high-quality features while keeping you in control.

### Semantic Types
Beyond raw dtypes (int, float), the agent identifies:
- **Numeric Continuous/Discrete:** Values to be scaled or binned.
- **Categorical Low/High:** Labels to be encoded (OHE vs Frequency/Target).
- **ID Columns:** High-cardinality unique values (automatically dropped).
- **Boolean:** Flags to be standardized to 0/1.
- **DateTime:** Timestamps to be decomposed into cyclical features.
- **Text:** Free-form strings for length and pattern extraction.

### Human-in-the-Loop (HITL) Interactions
To avoid the "combinatorial explosion" of features (where adding 100 columns could create 10,000 interactions), the agent:
1. Ranks features using **Mutual Information**.
2. Presents only the most promising pairs to you via a prompt.
3. Only computes what you approve.

## Part 2 — Interaction & Decision Logging

Every decision the agent makes is recorded in:
- **Audit Report (Markdown):** A human-readable summary of what happened and why.
- **Decision Log (JSON):** A technical trace for debugging or automated review.

## Part 3 — Schema Overrides

If the auto-profiler misidentifies a column (e.g., treating a Zip Code as a number), you can use the `column_overrides` config:
```yaml
column_overrides:
  zip_code:
    semantic_type: categorical_high
  tier:
    semantic_type: ordinal
    ordinal_order: [1, 2, 3, 4]
```

## Part 4 — Predictive Power Metrics
The agent calculates the following for every feature (including interactions) to help you decide what to keep:
- **Mutual Information (MI):** Measures non-linear dependency between feature and target.
- **Information Value (IV):** Quantifies predictive power for binary classification.
- **Pearson Correlation (r):** Measures linear relationship.

*Metrics Interpretation:*
| IV Range | Interpretation |
|---|---|
| < 0.02 | Useless |
| 0.1 – 0.3 | Medium |
| > 0.5 | Suspicious (Potential Leakage) |

## Part 5 — Leakage Guard
The agent proactively flags features with suspiciously high predictive metrics or those matching target naming heuristics.
