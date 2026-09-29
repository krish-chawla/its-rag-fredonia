"""Shared human-escalation logic used by every bot page.

Detects two situations where the bot should hand off to a human instead of
just leaving an unhelpful answer on screen:
  1. The bot's own answer fell back to its "I don't have that information"
     line (it means it couldn't find anything relevant in the document).
  2. The user explicitly asked to talk to a human.
"""

import streamlit as st

# Phrases that indicate the bot gave up / couldn't answer. Checked against
# the bot's response, case-insensitively. Keep this in sync with whatever
# fallback wording each bot's prompt actually uses.
FALLBACK_PHRASES = [
    "i don't have that information",
    "i do not have that information",
]

# Phrases that indicate the user is explicitly asking for a person instead
# of the bot. Checked against the user's message, case-insensitively.
HUMAN_REQUEST_PHRASES = [
    "talk to a human",
    "speak to a human",
    "talk to a person",
    "speak to a person",
    "speak to someone",
    "talk to someone",
    "real person",
    "human being",
    "customer service rep",
    "representative",
    "agent",
    "speak to a rep",
]


def bot_could_not_answer(response_text: str) -> bool:
    lowered = response_text.lower()
    return any(phrase in lowered for phrase in FALLBACK_PHRASES)


def user_requested_human(user_text: str) -> bool:
    lowered = user_text.lower()
    return any(phrase in lowered for phrase in HUMAN_REQUEST_PHRASES)


def should_escalate(user_text: str, response_text: str) -> bool:
    return bot_could_not_answer(response_text) or user_requested_human(user_text)


def show_escalation(contact_name: str, contact_line: str, note: str = ""):
    """Renders a distinct escalation box. Call this right after the bot's
    own answer, inside the same st.chat_message('assistant') block."""
    message = f"**Need a real person?** You can reach {contact_name} directly:\n\n{contact_line}"
    if note:
        message += f"\n\n{note}"
    st.info(message, icon="🙋")