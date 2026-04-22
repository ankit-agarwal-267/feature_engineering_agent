import warnings
from typing import List
from fe_agent.profiler.semantic_types import ColumnProfile, SemanticType
from fe_agent.config.config_schema import FEConfig

class OverrideResolver:
    def __init__(self, config: FEConfig):
        self.config = config

    def resolve(self, profiles: List[ColumnProfile]) -> List[ColumnProfile]:
        """
        Applies user-supplied column_overrides to the auto-inferred profiles.
        """
        overrides = self.config.column_overrides
        if not overrides:
            return profiles

        profile_map = {p.name: p for p in profiles}
        
        # Section 6.5.5: Validation rules
        for col_name in overrides:
            if col_name not in profile_map:
                msg = f"Unknown column '{col_name}' in column_overrides."
                if self.config.strict_overrides:
                    raise ValueError(msg)
                else:
                    warnings.warn(msg)
                continue

            profile = profile_map[col_name]
            override = overrides[col_name]

            if override.drop:
                profile.drop = True
                profile.semantic_type = profile.inferred_semantic_type # Keep for audit
                profile.override_applied = True
                profile.override_source = "config"
                continue

            if override.semantic_type:
                # Section 6.5.5: ordinal check
                if override.semantic_type == SemanticType.ORDINAL and not override.ordinal_order:
                    raise ValueError(f"semantic_type: ordinal specified for '{col_name}' without ordinal_order.")
                
                profile.semantic_type = override.semantic_type
                profile.detected_order = override.ordinal_order
                profile.override_applied = True
                profile.override_source = "config"

            if override.skip_transforms:
                profile.skip_transforms = override.skip_transforms
                profile.override_applied = True
                profile.override_source = "config"

            if override.force_transforms:
                profile.force_transforms = override.force_transforms
                profile.override_applied = True
                profile.override_source = "config"

            if override.datetime_format:
                profile.datetime_format = override.datetime_format
                profile.override_applied = True
                profile.override_source = "config"

        return profiles
