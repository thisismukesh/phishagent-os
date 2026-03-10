"""Profile loading, generation, and validation.

Loads victim profiles from YAML, validates them, and generates factorial profile
sets for batch experiments.
"""

import copy
import itertools
from pathlib import Path

import yaml

from phishagent.models import FactorialSpec, VictimProfile
from phishagent.utils import get_logger

logger = get_logger(__name__)


class ProfileManager:
    def load_profile(self, path: str) -> VictimProfile:
        """Load a single profile from a YAML file."""
        filepath = Path(path)
        if not filepath.exists():
            raise FileNotFoundError(f"Profile not found: {path}")

        with open(filepath) as f:
            data = yaml.safe_load(f)

        if not data or not isinstance(data, dict):
            raise ValueError(f"Profile file is empty or not a valid YAML mapping: {path}")

        profile = VictimProfile(**data)
        logger.info(f"Loaded profile '{profile.name}' from {path}")
        return profile

    def load_profiles(self, path: str) -> list[VictimProfile]:
        """Load multiple profiles from a YAML file containing a list."""
        filepath = Path(path)
        if not filepath.exists():
            raise FileNotFoundError(f"Profiles file not found: {path}")

        with open(filepath) as f:
            data = yaml.safe_load(f)

        if not data:
            raise ValueError(f"Profiles file is empty: {path}")

        if isinstance(data, list):
            profiles = [VictimProfile(**item) for item in data]
        elif isinstance(data, dict):
            profiles = [VictimProfile(**data)]
        else:
            raise ValueError(f"Expected list or dict in {path}, got {type(data)}")

        logger.info(f"Loaded {len(profiles)} profiles from {path}")
        return profiles

    def generate_factorial_profiles(self, spec: FactorialSpec) -> list[VictimProfile]:
        """Generate profiles from a factorial design specification.

        Cartesian product of all `vary` fields. For each combination, deep-copy
        the base_profile and override the specified fields.
        """
        if not spec.vary:
            return [spec.base_profile]

        # Separate field names and their value lists
        field_names = list(spec.vary.keys())
        value_lists = list(spec.vary.values())

        # Generate cartesian product
        combinations = list(itertools.product(*value_lists))
        profiles = []

        for combo in combinations:
            # Deep copy base profile as a dict for modification
            profile_data = spec.base_profile.model_dump()

            # Build a name suffix encoding the varied parameters
            name_parts = []

            for field_name, value in zip(field_names, combo):
                # Handle nested fields like "personality.agreeableness"
                self._set_nested_field(profile_data, field_name, value)

                # Build name part: first letter of field + value
                short_field = field_name.split(".")[-1][0].upper()
                name_parts.append(f"{short_field}{value}")

            # Assign synthetic name
            profile_data["name"] = f"Victim_{'_'.join(str(p) for p in name_parts)}"

            profiles.append(VictimProfile(**profile_data))

        logger.info(
            f"Generated {len(profiles)} factorial profiles from {len(field_names)} varied fields"
        )
        return profiles

    def validate_profile(self, profile: VictimProfile) -> list[str]:
        """Return list of warnings (not errors — Pydantic handles errors).

        Flags unusual but valid combinations.
        """
        warnings = []

        if profile.tech_proficiency > 0.7 and profile.security_awareness.value == "high":
            warnings.append(
                "High tech_proficiency + high security_awareness is an unusual combination"
            )

        if profile.impulsivity > 0.7 and profile.personality.conscientiousness > 0.7:
            warnings.append(
                "High impulsivity + high conscientiousness is psychologically unusual"
            )

        if profile.personality.agreeableness < 0.3 and profile.personality.extraversion > 0.8:
            warnings.append(
                "Very low agreeableness + very high extraversion is an unusual personality profile"
            )

        return warnings

    @staticmethod
    def _set_nested_field(data: dict, field_path: str, value) -> None:
        """Set a nested field in a dictionary using dot notation.

        E.g., 'personality.agreeableness' sets data['personality']['agreeableness'].
        """
        parts = field_path.split(".")
        current = data
        for part in parts[:-1]:
            if not isinstance(current, dict) or part not in current:
                raise KeyError(f"Invalid field path '{field_path}': key '{part}' not found")
            current = current[part]
        if not isinstance(current, dict):
            raise KeyError(f"Invalid field path '{field_path}': cannot set on non-dict")
        current[parts[-1]] = value
