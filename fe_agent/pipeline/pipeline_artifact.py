import json
import pickle
import pandas as pd
import polars as pl
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Any, List, Dict, Optional, Union
from pathlib import Path

@dataclass
class FEStep:
    step_id: str
    transform_name: str
    source_columns: List[str]
    output_columns: List[str]
    params: Dict[str, Any] = field(default_factory=dict)
    fit_artifacts: Dict[str, Any] = field(default_factory=dict)

class FEPipeline:
    def __init__(self, steps: List[FEStep] = None, final_columns: List[str] = None, version: str = "1.1.0"):
        self.steps = steps or []
        self.final_columns = final_columns or []
        self.version = version

    def add_step(self, step: FEStep):
        self.steps.append(step)

    def set_final_columns(self, columns: List[str]):
        self.final_columns = columns

    def save(self, path: Union[str, Path]):
        path = Path(path)
        data = {"version": self.version, "final_columns": self.final_columns, "steps": [asdict(s) for s in self.steps]}
        if path.suffix == '.json':
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        else:
            with open(path, 'wb') as f:
                pickle.dump(data, f)

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'FEPipeline':
        path = Path(path)
        if path.suffix == '.json':
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return cls(steps=[FEStep(**s) for s in data["steps"]], final_columns=data.get("final_columns", []), version=data["version"])
        else:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                return cls(steps=[FEStep(**s) for s in data["steps"]], final_columns=data.get("final_columns", []), version=data["version"])

    def apply(self, df: Any) -> Any:
        """
        NATURAL Replay: Executes recorded steps and arrives at the schema purely through transformation and drops.
        Uses single batch concatenation for high performance.
        """
        is_pandas = isinstance(df, pd.DataFrame)
        out_df = df.copy() if is_pandas else df.clone()
        
        # Buffer to store new columns generated during replay
        generated_features = {}
        cols_to_drop = []
        
        for step in self.steps:
            name = step.transform_name
            cols = step.source_columns
            out_cols = step.output_columns
            
            try:
                if name == "log1p":
                    generated_features[out_cols[0]] = np.log1p(pd.to_numeric(out_df[cols[0]], errors='coerce').fillna(0)).astype('float32')
                elif name == "sqrt":
                    generated_features[out_cols[0]] = np.sqrt(pd.to_numeric(out_df[cols[0]], errors='coerce').fillna(0)).astype('float32')
                elif name.startswith("polynomial"):
                    deg = step.params.get('degree', 2)
                    generated_features[out_cols[0]] = (pd.to_numeric(out_df[cols[0]], errors='coerce').fillna(0) ** deg).astype('float32')
                elif name == "interaction":
                    if "cat_x" in out_cols[0]:
                        generated_features[out_cols[0]] = (out_df[cols[0]].astype(str) + "_" + out_df[cols[1]].astype(str))
                    elif "grp" in out_cols[0]:
                        means = step.fit_artifacts.get('means', {})
                        if means:
                            generated_features[out_cols[0]] = out_df[cols[0]].map(means).fillna(0).astype('float32')
                    else:
                        generated_features[out_cols[0]] = (pd.to_numeric(out_df[cols[0]], errors='coerce').fillna(0) * 
                                                          pd.to_numeric(out_df[cols[1]], errors='coerce').fillna(0)).astype('float32')
                elif name == "one_hot_encoding":
                    prefix = cols[0]
                    for oc in out_cols:
                        val = oc.replace(f"{prefix}_", "")
                        generated_features[oc] = (out_df[prefix].astype(str).str.lower().str.replace(' ', '_') == val).astype('uint8')
                elif name == "datetime_extraction":
                    dt = pd.to_datetime(out_df[cols[0]], errors='coerce')
                    for oc in out_cols:
                        if "_year" in oc: generated_features[oc] = dt.dt.year.fillna(0).astype('int16')
                        elif "_month" in oc: generated_features[oc] = dt.dt.month.fillna(0).astype('int8')
                elif name == "boolean_cast":
                    bool_map = {'true': 1, 'false': 0, 'yes': 1, 'no': 0, 'y': 1, 'n': 0}
                    out_df[cols[0]] = out_df[cols[0]].astype(str).str.lower().map(bool_map).fillna(0).astype('int8')
                elif name == "drop":
                    cols_to_drop.extend(cols)
            except Exception:
                pass
        
        # Batch concatenation of all new features
        if generated_features:
            new_cols_df = pd.DataFrame(generated_features)
            out_df = pd.concat([out_df, new_cols_df], axis=1)
            
        # Perform all drops at once
        unique_drops = list(set(cols_to_drop))
        out_df = out_df.drop(columns=[c for c in unique_drops if c in out_df.columns])
            
        return out_df
