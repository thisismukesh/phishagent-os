"""Unit tests for profile loading and factorial generation."""

import tempfile

import pytest
import yaml

from phishagent.models import FactorialSpec, SecurityAwareness, VictimProfile
from phishagent.profile_manager import ProfileManager


@pytest.fixture
def manager():
    return ProfileManager()


class TestLoadProfile:
    def test_load_valid_profile(self, manager):
        profile = manager.load_profile("config/profiles/high_agreeableness.yaml")
        assert profile.name == "Alex Chen"
        assert profile.personality.agreeableness == 0.9

    def test_load_nonexistent_file(self, manager):
        with pytest.raises(FileNotFoundError):
            manager.load_profile("nonexistent.yaml")

    def test_load_all_sample_profiles(self, manager):
        for name in ["high_agreeableness", "low_conscientiousness", "overconfident_techie"]:
            profile = manager.load_profile(f"config/profiles/{name}.yaml")
            assert isinstance(profile, VictimProfile)


class TestLoadProfiles:
    def test_load_single_as_list(self, manager):
        profiles = manager.load_profiles("config/profiles/high_agreeableness.yaml")
        assert len(profiles) == 1

    def test_load_list_file(self, manager):
        data = [
            {
                "name": "V1", "personality": {"openness": 0.5, "conscientiousness": 0.5,
                "extraversion": 0.5, "agreeableness": 0.5, "neuroticism": 0.5},
                "communication_style": "casual", "security_awareness": "low",
                "interests": ["tech"], "occupation": "engineer",
                "tech_proficiency": 0.5, "impulsivity": 0.5,
            },
            {
                "name": "V2", "personality": {"openness": 0.5, "conscientiousness": 0.5,
                "extraversion": 0.5, "agreeableness": 0.8, "neuroticism": 0.5},
                "communication_style": "formal", "security_awareness": "high",
                "interests": ["finance"], "occupation": "accountant",
                "tech_proficiency": 0.3, "impulsivity": 0.2,
            },
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(data, f)
            f.flush()
            profiles = manager.load_profiles(f.name)
        assert len(profiles) == 2
        assert profiles[0].name == "V1"
        assert profiles[1].name == "V2"


class TestFactorialGeneration:
    def test_single_variable(self, manager, sample_profile):
        spec = FactorialSpec(
            base_profile=sample_profile,
            vary={"personality.agreeableness": [0.2, 0.5, 0.8]},
        )
        profiles = manager.generate_factorial_profiles(spec)
        assert len(profiles) == 3
        assert profiles[0].personality.agreeableness == 0.2
        assert profiles[1].personality.agreeableness == 0.5
        assert profiles[2].personality.agreeableness == 0.8

    def test_two_variables(self, manager, sample_profile):
        spec = FactorialSpec(
            base_profile=sample_profile,
            vary={
                "personality.agreeableness": [0.2, 0.8],
                "security_awareness": ["low", "high"],
            },
        )
        profiles = manager.generate_factorial_profiles(spec)
        # 2 × 2 = 4
        assert len(profiles) == 4

    def test_three_by_three(self, manager, sample_profile):
        spec = FactorialSpec(
            base_profile=sample_profile,
            vary={
                "personality.agreeableness": [0.2, 0.5, 0.8],
                "security_awareness": ["low", "medium", "high"],
            },
        )
        profiles = manager.generate_factorial_profiles(spec)
        # 3 × 3 = 9
        assert len(profiles) == 9

    def test_non_varied_fields_unchanged(self, manager, sample_profile):
        spec = FactorialSpec(
            base_profile=sample_profile,
            vary={"personality.agreeableness": [0.2, 0.8]},
        )
        profiles = manager.generate_factorial_profiles(spec)
        for profile in profiles:
            assert profile.personality.openness == sample_profile.personality.openness
            assert profile.occupation == sample_profile.occupation

    def test_empty_vary_returns_base(self, manager, sample_profile):
        spec = FactorialSpec(base_profile=sample_profile, vary={})
        profiles = manager.generate_factorial_profiles(spec)
        assert len(profiles) == 1

    def test_profiles_have_synthetic_names(self, manager, sample_profile):
        spec = FactorialSpec(
            base_profile=sample_profile,
            vary={"personality.agreeableness": [0.2, 0.8]},
        )
        profiles = manager.generate_factorial_profiles(spec)
        for profile in profiles:
            assert profile.name.startswith("Victim_")


class TestValidateProfile:
    def test_no_warnings_for_typical_profile(self, manager, sample_profile):
        warnings = manager.validate_profile(sample_profile)
        assert len(warnings) == 0

    def test_warns_high_tech_high_security(self, manager, sample_profile):
        profile = sample_profile.model_copy(
            update={"tech_proficiency": 0.9, "security_awareness": SecurityAwareness.HIGH}
        )
        warnings = manager.validate_profile(profile)
        assert any("tech_proficiency" in w for w in warnings)

    def test_warns_high_impulsivity_high_conscientiousness(self, manager, sample_personality):
        from phishagent.models import CommunicationStyle, PersonalityTraits

        traits = sample_personality.model_copy(update={"conscientiousness": 0.9})
        profile = VictimProfile(
            name="Test", personality=traits,
            communication_style=CommunicationStyle.CASUAL,
            security_awareness=SecurityAwareness.LOW,
            interests=["tech"], occupation="engineer",
            tech_proficiency=0.5, impulsivity=0.8,
        )
        warnings = manager.validate_profile(profile)
        assert any("impulsivity" in w for w in warnings)
