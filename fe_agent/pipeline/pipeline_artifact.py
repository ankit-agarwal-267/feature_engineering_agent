import json
import pickle
import base64
from dataclasses import dataclass, field, asdict
from typing import Any, List, Dict, Optional, Union
import pandas as pd
import polars as pl
from pathlib import Path

@dataclass
class FEStep:
    step_id: str
    transform_name: str
    source_columns: List[str]
    output_columns: List[str]
    params: Dict[str, Any] = field(default_factory=dict)
    fit_artifacts: Dict[str, Any] = field(default_factory=dict)
    execution_order: int = 0

class FEPipeline:
    def __init__(self, steps: List[FEStep] = None, version: str = "1.1.0"):
        self.steps = steps or []
        self.version = version

    def add_step(self, step: FEStep):
        step.execution_order = len(self.steps)
        self.steps.append(step)

    def transform(self, df: Any) -> Any:
        """
        Applies all steps in order on a new DataFrame (Section 12.2).
        """
        # This would implement the logic to replay each step.
        # For now, placeholder.
        return df

    def save_json(self, path: Union[str, Path]):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                "fe_agent_version": self.version,
                "steps": [asdict(s) for s in self.steps]
            }, f, indent=2)

    @classmethod
    def load_json(cls, path: Union[str, Path]) -> 'FEPipeline':
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        steps = [FEStep(**s) for s in data["steps"]]
        return cls(steps=steps, version=data["fe_agent_version"])

    def save_pickle(self, path: Union[str, Path]):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load_pickle(cls, path: Union[str, Path]) -> 'FEPipeline':
        with open(path, 'rb') as f:
            return pickle.load(f)
