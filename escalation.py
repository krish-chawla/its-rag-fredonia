"""Human escalation for the ITS policy chatbot.

Two jobs:
  1. Detect when the bot can't (or shouldn't) handle a question on its own.
  2. Hand the conversation off to ITS staff as a ticket (local log + optional email).

This module has no LangChain/OpenAI dependency so it can be unit tested offline.
"""
import json
import os
import re
import smtplib
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from email.message import EmailMessage

# The answer prompt tells the LLM to reply with exactly this when the context
# doesn't cover the question — we detect it to trigger escalation.
NO_ANSWER_SENTINEL = "I don't have that information."

# Chroma relevance scores are 0-1 (higher = more similar). Below this, retrieval
# found nothing useful, so we skip the LLM and offer a handoff instead.
# Calibrate against real questions — the test run prints the top score.
DEFAULT_MIN_RELEVANCE = float(os.getenv("ESCALATION_MIN_RELEVANCE", "0.3"))

# Reasons, in the order they are checked
USER_REQUESTED = "user_requested"
SECURITY_INCIDENT = "security_incident"
LOW_RETRIEVAL_CONFIDENCE = "low_retrieval_confidence"
NO_ANSWER = "no_answer"

REASON_DESCRIPTIONS = {
    USER_REQUESTED: "User asked to speak with a person",
    SECURITY_INCIDENT: "Possible security incident reported",
    LOW_RETRIEVAL_CONFIDENCE: "No relevant policy content found",
    NO_ANSWER: "Bot could not answer from the policy document",
}

HUMAN_REQUEST_PATTERNS = [
    r"\b(talk|speak|chat)\s+(to|with)\s+(a\s+|an\s+|some\s+)?(real\s+|live\s+)?"
    r"(human|person|someone|somebody|agent|representative|staff|technician)\b",
    r"\b(real|live)\s+(person|human|agent)\b",
    r"\bhuman\s+(help|support|agent)\b",
    r"\b(open|create|submit|file)\s+(a\s+)?(support\s+|help\s*desk\s+)?ticket\b",
    r"\bcontact\s+(the\s+)?(help\s*desk|service\s*desk|its|it\s+support|support)\b",
    r"\bescalate\b",
]

# The policy says suspected unauthorized use must be reported immediately,
# so these always go to a human (at high priority) even if the bot answers.
SECURITY_PATTERNS = [
    r"\b(hacked|compromised|breached)\b",
    r"\bphish(ing|ed)?\b",
    r"\bransomware\b|\bmalware\b|\bvirus\b",
    r"\bsomeone\s+(else\s+)?(is\s+|has\s+been\s+)?(using|logged\s+into|accessing|got\s+into)\s+my\b",
    r"\bunauthori[sz]ed\s+(access|use|login|activity)\b",
    r"\b(stolen|leaked)\s+(password|credentials|account|laptop|device)\b",
    r"\bdata\s+breach\b",
]

_HUMAN_RE = re.compile("|".join(HUMAN_REQUEST_PATTERNS), re.IGNORECASE)
_SECURITY_RE = re.compile("|".join(SECURITY_PATTERNS), re.IGNORECASE)
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


@dataclass
class EscalationDecision:
    escalate: bool
    reason: str | None = None
    priority: str = "normal"      # "normal" | "high"
    answer_first: bool = False    # still give the bot's answer before handing off

    @property
    def description(self) -> str:
        return REASON_DESCRIPTIONS.get(self.reason, "")


NO_ESCALATION = EscalationDecision(escalate=False)


def wants_human(question: str) -> bool:
    return bool(_HUMAN_RE.search(question))


def is_security_incident(question: str) -> bool:
    return bool(_SECURITY_RE.search(question))


def check_before_answer(question: str, top_relevance: float | None,
                        min_relevance: float | None = None) -> EscalationDecision:
    """Run before calling the LLM. Decides whether to answer at all."""
    if min_relevance is None:
        min_relevance = DEFAULT_MIN_RELEVANCE
    if wants_human(question):
        return EscalationDecision(True, USER_REQUESTED)
    if is_security_incident(question):
        # Policy guidance (change password, report it) is still useful, so answer too
        return EscalationDecision(True, SECURITY_INCIDENT, priority="high", answer_first=True)
    if top_relevance is None or top_relevance < min_relevance:
        return EscalationDecision(True, LOW_RETRIEVAL_CONFIDENCE)
    return NO_ESCALATION


def is_no_answer(answer: str) -> bool:
    return NO_ANSWER_SENTINEL.lower().rstrip(".") in answer.lower()


def check_after_answer(answer: str) -> EscalationDecision:
    """Run after the LLM answers. Catches the 'not in context' fallback."""
    if is_no_answer(answer):
        return EscalationDecision(True, NO_ANSWER)
    return NO_ESCALATION


def is_valid_email(email: str) -> bool:
    return bool(_EMAIL_RE.match(email.strip()))


# ---------------------------------------------------------------------------
# Handoff
# ---------------------------------------------------------------------------

@dataclass
class Ticket:
    reason: str
    priority: str
    question: str
    contact_name: str
    contact_email: str
    bot_answer: str | None = None
    sources: list[str] = field(default_factory=list)
    transcript: list[dict] = field(default_factory=list)
    id: str = field(default_factory=lambda: f"ITS-{uuid.uuid4().hex[:8].upper()}")
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def subject(self) -> str:
        prefix = "[URGENT] " if self.priority == "high" else ""
        return f"{prefix}Chatbot escalation {self.id}: {REASON_DESCRIPTIONS.get(self.reason, self.reason)}"

    def body(self) -> str:
        lines = [
            f"Ticket: {self.id}",
            f"Created: {self.created_at}",
            f"Priority: {self.priority}",
            f"Reason: {REASON_DESCRIPTIONS.get(self.reason, self.reason)}",
            f"Contact: {self.contact_name} <{self.contact_email}>",
            "",
            f"Question: {self.question}",
        ]
        if self.bot_answer:
            lines += ["", f"Bot's answer: {self.bot_answer}"]
        if self.sources:
            lines += ["", "Policy sections retrieved:"] + [f"  - {s}" for s in self.sources]
        if self.transcript:
            lines += ["", "Conversation transcript:"]
            for turn in self.transcript:
                lines.append(f"  {turn['role'].upper()}: {turn['content']}")
        return "\n".join(lines)


class FileBackend:
    """Appends tickets to a JSONL file. Always on, so nothing is lost if email fails."""
    name = "log"

    def __init__(self, path: str | None = None):
        self.path = path or os.getenv("ESCALATION_LOG_PATH", "escalations/tickets.jsonl")

    def send(self, ticket: Ticket) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(ticket)) + "\n")


class EmailBackend:
    """Emails the ticket to the ITS help desk inbox over SMTP.

    Most help desk systems (TeamDynamix, ServiceNow, etc.) can turn emails to
    their intake address into tickets, so pointing ITS_HELPDESK_EMAIL at that
    address gives real ticket creation without a vendor-specific API.
    """
    name = "email"

    def __init__(self, to_addr: str, host: str, port: int = 587,
                 user: str | None = None, password: str | None = None,
                 from_addr: str | None = None):
        self.to_addr = to_addr
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.from_addr = from_addr or user or to_addr

    @classmethod
    def from_env(cls) -> "EmailBackend | None":
        to_addr = os.getenv("ITS_HELPDESK_EMAIL")
        host = os.getenv("ESCALATION_SMTP_HOST")
        if not (to_addr and host):
            return None
        return cls(
            to_addr=to_addr,
            host=host,
            port=int(os.getenv("ESCALATION_SMTP_PORT", "587")),
            user=os.getenv("ESCALATION_SMTP_USER"),
            password=os.getenv("ESCALATION_SMTP_PASSWORD"),
            from_addr=os.getenv("ESCALATION_FROM_EMAIL"),
        )

    def send(self, ticket: Ticket) -> None:
        msg = EmailMessage()
        msg["Subject"] = ticket.subject
        msg["From"] = self.from_addr
        msg["To"] = self.to_addr
        msg["Reply-To"] = ticket.contact_email
        msg.set_content(ticket.body())
        with smtplib.SMTP(self.host, self.port, timeout=15) as smtp:
            smtp.starttls()
            if self.user and self.password:
                smtp.login(self.user, self.password)
            smtp.send_message(msg)


def default_backends() -> list:
    backends = [FileBackend()]
    email = EmailBackend.from_env()
    if email:
        backends.append(email)
    return backends


def route_ticket(ticket: Ticket, backends: list | None = None) -> list[str]:
    """Send the ticket to every backend. Returns names of the ones that succeeded."""
    delivered = []
    for backend in backends if backends is not None else default_backends():
        try:
            backend.send(ticket)
            delivered.append(backend.name)
        except Exception as e:  # one failing channel shouldn't block the others
            print(f"[escalation] {backend.name} delivery failed: {e}")
    return delivered


def handoff_message(decision: EscalationDecision) -> str:
    """What the bot says to the user when it hands off."""
    if decision.reason == USER_REQUESTED:
        return "Sure — I'll connect you with the ITS Service Desk."
    if decision.reason == SECURITY_INCIDENT:
        return ("This sounds like a possible security issue, which ITS needs to know about "
                "right away. I'd like to send this to the ITS Service Desk as an urgent ticket.")
    return ("I'm not able to answer that from the ITS policy. "
            "I can pass your question to the ITS Service Desk so a staff member can follow up.")
