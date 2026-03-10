"""All Pydantic data models for PhishAgent-OS — the single source of truth.

Every other module imports from here. No data classes defined elsewhere.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ── Victim Profile Models ──────────────────────────────────────────────────────


class PersonalityTraits(BaseModel):
    """Big Five personality traits, each on a 0.0–1.0 scale.

    Grounded in:
    - Heliyon (2025) systematic review: extraversion, agreeableness, neuroticism
      positively associated with phishing vulnerability; conscientiousness protective.
    - Denisenko et al. (2026): overconfidence (high self-assessed competence) is a
      risk amplifier, not protective.
    - Hart et al. (2025): Dark Triad facets increase susceptibility via decreased
      social awareness.
    """

    openness: float = Field(ge=0.0, le=1.0, description="Openness to experience")
    conscientiousness: float = Field(ge=0.0, le=1.0, description="Conscientiousness")
    extraversion: float = Field(ge=0.0, le=1.0, description="Extraversion")
    agreeableness: float = Field(ge=0.0, le=1.0, description="Agreeableness")
    neuroticism: float = Field(ge=0.0, le=1.0, description="Neuroticism")


class SecurityAwareness(str, Enum):
    """Based on Muhanad et al. (2025): personality traits are weak predictors
    when targets are already risk-aware. This is a separate moderating variable."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class CommunicationStyle(str, Enum):
    """How the victim communicates. Affects response length, formality, emoji use."""

    FORMAL = "formal"
    CASUAL = "casual"
    TERSE = "terse"
    VERBOSE = "verbose"


class VictimProfile(BaseModel):
    """Complete synthetic victim profile.

    Profile features are drawn from Denisenko et al. (2026) H1–H9 variables,
    Alshammari et al. (2025) 90-variable taxonomy, and Shin et al. (2023)
    agent-based simulation parameters.
    """

    name: str = Field(description="Synthetic name for conversation context")
    personality: PersonalityTraits
    communication_style: CommunicationStyle
    security_awareness: SecurityAwareness
    interests: list[str] = Field(
        min_length=1,
        max_length=5,
        description="e.g. ['sports', 'finance', 'technology']",
    )
    occupation: str = Field(description="e.g. 'software engineer', 'marketing manager'")
    tech_proficiency: float = Field(
        ge=0.0,
        le=1.0,
        description="Self-declared tech proficiency (Denisenko H7: higher = more vulnerable due to overconfidence)",
    )
    impulsivity: float = Field(
        ge=0.0,
        le=1.0,
        description="Response speed proxy (Denisenko H9: faster = more vulnerable to emotional content)",
    )


# ── Attacker Config Models ─────────────────────────────────────────────────────


class AttackGoal(str, Enum):
    """What the attacker is trying to get the victim to do."""

    CLICK_LINK = "click_link"
    SHARE_PERSONAL_INFO = "share_personal_info"
    DOWNLOAD_FILE = "download_file"
    SHARE_CREDENTIALS = "share_credentials"


class AttackStrategy(str, Enum):
    """Social engineering strategies grounded in Cialdini's principles
    and adapted per SE-VSim (Kumarage et al. 2025) scenario types."""

    AUTHORITY = "authority"
    RECIPROCITY = "reciprocity"
    URGENCY = "urgency"
    SOCIAL_PROOF = "social_proof"
    RAPPORT = "rapport"
    FLATTERY = "flattery"


class AttackerScenario(str, Enum):
    """Scenario types from SE-VSim: attacker's cover story."""

    RECRUITER = "recruiter"
    IT_SUPPORT = "it_support"
    COLLEAGUE = "colleague"
    VENDOR = "vendor"
    EVENT_ORGANIZER = "event_organizer"


class AttackerConfig(BaseModel):
    """Configuration for the attacker agent for a single conversation."""

    goal: AttackGoal
    strategy: AttackStrategy
    scenario: AttackerScenario
    max_turns: int = Field(
        default=10, ge=3, le=10, description="Max conversation turns before forced termination"
    )
    escalation_threshold: int = Field(
        default=5, description="Turn at which attacker escalates tactics"
    )


# ── LLM Response Model ─────────────────────────────────────────────────────────


class LLMResponse(BaseModel):
    """Structured response from the Ollama LLM."""

    content: str
    model: str
    token_count: int = Field(default=0)
    duration_ms: int = Field(default=0)
    done: bool = Field(default=True)


# ── Conversation Data Structures ───────────────────────────────────────────────


class Message(BaseModel):
    """A single message in a conversation."""

    role: str = Field(pattern="^(attacker|victim)$")
    content: str
    turn_number: int = Field(ge=0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    token_count: int = Field(default=0, description="Tokens consumed by this message")


class ConversationOutcome(str, Enum):
    """How the conversation ended."""

    COMPLIANCE = "compliance"
    PARTIAL_COMPLIANCE = "partial"
    REFUSAL = "refusal"
    SUSPICION = "suspicion"
    MAX_TURNS = "max_turns"
    ERROR = "error"


class ConversationScore(BaseModel):
    """Scores for a completed conversation. Each score is 0.0–1.0."""

    persuasion: float = Field(ge=0.0, le=1.0, description="How effectively the attacker persuaded")
    coherence: float = Field(
        ge=0.0, le=1.0, description="Dialogue naturalness and consistency"
    )
    detectability: float = Field(
        ge=0.0,
        le=1.0,
        description="How easily detectable as malicious (0=undetectable, 1=obvious)",
    )
    composite: float = Field(ge=0.0, le=1.0, description="Weighted composite score")
    persuasion_rationale: str = Field(default="", description="LLM judge explanation")
    coherence_rationale: str = Field(default="", description="LLM judge explanation")
    detectability_rationale: str = Field(default="", description="LLM judge explanation")


class ConversationResult(BaseModel):
    """Complete record of a single simulated conversation."""

    conversation_id: str = Field(description="UUID for this conversation")
    victim_profile: VictimProfile
    attacker_config: AttackerConfig
    messages: list[Message]
    outcome: ConversationOutcome
    scores: Optional[ConversationScore] = None
    model_name: str = Field(description="LLM model used, e.g. 'mistral:7b'")
    total_tokens: int = Field(default=0)
    total_duration_seconds: float = Field(default=0.0)
    watermark: str = Field(
        default="SYNTHETIC_RESEARCH_OUTPUT:PhishAgent-OS",
        description="Identifies this as synthetic research data",
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ── Experiment Data Structures ─────────────────────────────────────────────────


class FactorialSpec(BaseModel):
    """Specification for generating a factorial set of victim profiles."""

    base_profile: VictimProfile
    vary: dict[str, list]  # field_name → list of values to try


class ExperimentConfig(BaseModel):
    """Configuration for a batch experiment run."""

    experiment_id: str
    description: str
    model_name: str = Field(default="mistral:7b")
    profiles: list[VictimProfile]
    attacker_configs: list[AttackerConfig]
    repetitions: int = Field(default=3, ge=1, description="Runs per profile x config combination")
    output_dir: str = Field(default="output/experiments")


class ExperimentResult(BaseModel):
    """Summary of a completed batch experiment."""

    experiment_id: str
    total_conversations: int
    completed_conversations: int
    failed_conversations: int
    results: list[ConversationResult]
    started_at: datetime
    completed_at: datetime
