"""Multi-turn conversation orchestration.

Manages the conversation loop between attacker and victim agents. Handles
turn counting, early termination, error recovery, and state management.
"""

import time

from phishagent.attacker_agent import AttackerAgent
from phishagent.config import ConversationConfig
from phishagent.llm_client import LLMClientError
from phishagent.models import (
    AttackerConfig,
    ConversationOutcome,
    ConversationResult,
    Message,
    VictimProfile,
)
from phishagent.utils import generate_conversation_id, get_logger
from phishagent.victim_agent import VictimAgent

logger = get_logger(__name__)


class ConversationEngine:
    def __init__(
        self,
        attacker: AttackerAgent,
        victim: VictimAgent,
        config: ConversationConfig,
    ):
        """Initialize with both agents and conversation parameters."""
        self.attacker = attacker
        self.victim = victim
        self.config = config

    def run(
        self,
        victim_profile: VictimProfile,
        attacker_config: AttackerConfig,
        *,
        turn_callback=None,
    ) -> ConversationResult:
        """Execute the full conversation and return the result.

        Flow:
        1. Attacker generates opening message.
        2. Loop: victim responds → check termination → attacker responds → check.
        3. Attacker assesses final outcome.
        4. Return ConversationResult with all messages and metadata.

        Args:
            victim_profile: The victim's profile.
            attacker_config: The attacker's configuration.
            turn_callback: Optional callable(Message) called after each message
                is appended. Useful for live display in interactive mode.
        """
        conversation_id = generate_conversation_id()
        messages: list[Message] = []
        outcome = None
        start_time = time.monotonic()

        try:
            # Turn 0: Attacker opens
            opening = self.attacker.get_opening_message(victim_profile)
            if not opening or not opening.strip():
                opening = self._retry_empty_response(
                    "attacker_opening", victim_profile, messages, 0
                )
            msg = Message(
                role="attacker",
                content=opening,
                turn_number=0,
                token_count=len(opening.split()),
            )
            messages.append(msg)
            if turn_callback:
                turn_callback(msg)
            logger.info(f"[{conversation_id}] Turn 0: Attacker opened")

            for turn in range(1, self.config.max_turns + 1):
                # Optional delay between turns
                if self.config.turn_delay_seconds > 0:
                    time.sleep(self.config.turn_delay_seconds)

                # Victim responds
                victim_resp = self.victim.get_response(messages, turn)
                if not victim_resp or not victim_resp.strip():
                    victim_resp = self._retry_empty_response(
                        "victim", victim_profile, messages, turn
                    )
                msg = Message(
                    role="victim",
                    content=victim_resp,
                    turn_number=turn,
                    token_count=len(victim_resp.split()),
                )
                messages.append(msg)
                if turn_callback:
                    turn_callback(msg)
                logger.info(f"[{conversation_id}] Turn {turn}: Victim responded")

                # Check for early termination
                if self.config.early_termination:
                    outcome = self.attacker.assess_outcome(messages)
                    if outcome in {
                        ConversationOutcome.COMPLIANCE,
                        ConversationOutcome.REFUSAL,
                        ConversationOutcome.SUSPICION,
                    }:
                        logger.info(
                            f"[{conversation_id}] Early termination: "
                            f"{outcome.value} at turn {turn}"
                        )
                        break

                # Attacker responds (if turns remain)
                if turn < self.config.max_turns:
                    atk_resp = self.attacker.get_response(
                        messages, victim_profile, turn
                    )
                    if not atk_resp or not atk_resp.strip():
                        atk_resp = self._retry_empty_response(
                            "attacker", victim_profile, messages, turn
                        )
                    msg = Message(
                        role="attacker",
                        content=atk_resp,
                        turn_number=turn,
                        token_count=len(atk_resp.split()),
                    )
                    messages.append(msg)
                    if turn_callback:
                        turn_callback(msg)
                    logger.info(f"[{conversation_id}] Turn {turn}: Attacker responded")

            if outcome is None:
                outcome = ConversationOutcome.MAX_TURNS

        except LLMClientError as e:
            logger.error(f"[{conversation_id}] LLM error: {e}")
            outcome = ConversationOutcome.ERROR
        except Exception as e:
            logger.error(f"[{conversation_id}] Unexpected error: {e}")
            outcome = ConversationOutcome.ERROR

        duration = time.monotonic() - start_time
        total_tokens = sum(m.token_count for m in messages)

        return ConversationResult(
            conversation_id=conversation_id,
            victim_profile=victim_profile,
            attacker_config=attacker_config,
            messages=messages,
            outcome=outcome,
            model_name=self.attacker.model,
            total_tokens=total_tokens,
            total_duration_seconds=round(duration, 2),
        )

    def _retry_empty_response(
        self,
        agent_type: str,
        victim_profile: VictimProfile,
        messages: list[Message],
        turn: int,
    ) -> str:
        """Retry once with a nudge prompt when a response is empty."""
        logger.warning(f"Empty response from {agent_type} at turn {turn}, retrying...")

        if agent_type == "victim":
            nudge = Message(
                role="attacker",
                content="(Please respond in character.)",
                turn_number=turn,
            )
            resp = self.victim.get_response(messages + [nudge], turn)
        else:
            resp = self.attacker.get_response(messages, victim_profile, turn)

        return resp if resp and resp.strip() else "[No response generated]"
