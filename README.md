# Feature Engineering Agent

A robust, optionally LLM-assisted feature engineering agent for tabular data.

## Features
- **Semantic Profiling:** Automatically detects semantic types (continuous, categorical, datetime, etc.).
- **Interaction Engine:** Generates numeric interactions and polynomial features with Human-in-the-Loop (HITL) approval.
- **Predictive Metrics:** Tracks Mutual Information (MI), Information Value (IV), and Pearson Correlation for all features.
- **LLM Reasoning:** Integrates local LLMs (e.g., Ollama) for domain-specific feature engineering suggestions.
- **Integrity Checks:** Automated leakage warnings and quasi-constant column detection.
- **Full Auditability:** Generates detailed Markdown audit reports and machine-readable decision logs.

## Quick Start
```python
from fe_agent import FeatureEngineeringAgent, FEConfig

# 1. Configure the agent
config = FEConfig(target_column="y")

# 2. Initialize and run
agent = FeatureEngineeringAgent(config=config)
result = agent.run(source="data.csv")

# 3. Access results
print(result.transformed_df.head())
```

## Documentation
- **[User Manual](USER_MANUAL.md)**: Conceptual guide and configuration options.
- **[Quickstart](QUICKSTART.md)**: Setup and usage guide.
- **[Changelog](CHANGELOG.md)**: Version history.
