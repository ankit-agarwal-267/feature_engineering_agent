# Feature Engineering Agent — Quickstart

## 1. Install Dependencies
```bash
pip install pandas polars pydantic-settings scikit-learn sqlalchemy rich pyarrow
```

## 2. Prepare Your Data
Ensure you have a CSV or Parquet file with a clear **Target** column.

## 3. Run the Inspector (Dry-Run)
Use the dry-run flag to see what the agent thinks of your data before committing to transformations.
```python
from fe_agent import FeatureEngineeringAgent, FEConfig
config = FEConfig(target_column="churn", dry_run=True)
agent = FeatureEngineeringAgent(config=config)
agent.run("data.csv")
```

## 4. Run with All Features
The agent is rule-based by default. To enable LLM reasoning, pass the `--llm` flag. You can change the model by modifying the `LLMConfig` in your driver script (default is `mistral:7b`):
```bash
python run_agent.py --source data.csv --target y --llm --output_dir ./fe_output
```

## 5. Review Output
Check the `./fe_output` directory for:
- `audit_report_{run_id}.md` - Predictive metrics, LLM advice, and feature decisions.
- `decision_log_{run_id}.json` - Machine-readable logs.
- `pipeline_{run_id}.json` - For replaying on new test data.
