# Feature Engineering Agent

A robust, optionally LLM-assisted feature engineering agent for tabular data.

## Features
- **Semantic Profiling:** Automatically detects semantic types (continuous, categorical, datetime, etc.).
- **Interaction Engine:** Generates numeric interactions and polynomial features with Human-in-the-Loop (HITL) approval.
- **Predictive Metrics:** Tracks Mutual Information (MI), ANOVA, and Cramer's V for all features.
- **LLM Reasoning:** Integrates local LLMs (e.g., Ollama) for domain-specific feature engineering suggestions.
- **Pipeline Replay:** Serialize all transformations and pruning decisions to a JSON artifact for consistent application to test data.
- **Integrity Checks:** Automated leakage warnings and quasi-constant column detection.
- **Full Auditability:** Generates detailed Markdown audit reports and machine-readable decision logs.

## Quick Start
### 1. Training (Generate Pipeline)
```bash
python run_agent.py --source train.csv --target y --interactions --polynomials --llm
```

### 2. Testing (Replay Pipeline)
```bash
python run_agent.py --source test.csv --replay ./fe_output/pipeline_XXX.json
```

## Documentation
- **[User Manual](USER_MANUAL.md)**: Conceptual guide and configuration options.
- **[Quickstart](QUICKSTART.md)**: Setup and usage guide.
- **[Changelog](CHANGELOG.md)**: Version history.
