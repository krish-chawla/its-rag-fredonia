import json

import pytest

import escalation
from escalation import (
    EmailBackend, FileBackend, Ticket, check_after_answer, check_before_answer, route_ticket,
)

GOOD_SCORE = 0.8
BAD_SCORE = 0.1


@pytest.mark.parametrize("question", [
    "Can I talk to a real person?",
    "I want to speak with someone",
    "let me chat to a human",
    "Can you open a ticket for me?",
    "How do I contact the help desk?",
    "please escalate this",
])
def test_explicit_human_request(question):
    decision = check_before_answer(question, GOOD_SCORE)
    assert decision.escalate and decision.reason == escalation.USER_REQUESTED
    assert not decision.answer_first


@pytest.mark.parametrize("question", [
    "I think my account was hacked",
    "Someone is using my email account",
    "I clicked a phishing link",
    "There was unauthorized access to my files",
    "My laptop has a virus",
])
def test_security_incident_is_urgent_but_still_answered(question):
    decision = check_before_answer(question, GOOD_SCORE)
    assert decision.escalate and decision.reason == escalation.SECURITY_INCIDENT
    assert decision.priority == "high"
    assert decision.answer_first


@pytest.mark.parametrize("question", [
    "Can I share my password with a friend?",
    "What does Human Resources do in an investigation?",   # "human" alone must not trigger
    "Can Fredonia monitor my account without telling me?",
])
def test_normal_policy_questions_do_not_escalate(question):
    assert not check_before_answer(question, GOOD_SCORE).escalate


def test_low_or_missing_retrieval_score_escalates():
    assert check_before_answer("How do I fix my Xbox?", BAD_SCORE).reason == escalation.LOW_RETRIEVAL_CONFIDENCE
    assert check_before_answer("anything", None).reason == escalation.LOW_RETRIEVAL_CONFIDENCE


def test_threshold_is_configurable():
    assert not check_before_answer("question", 0.5, min_relevance=0.4).escalate
    assert check_before_answer("question", 0.5, min_relevance=0.6).escalate


def test_no_answer_sentinel_detected():
    assert check_after_answer("I don't have that information.").reason == escalation.NO_ANSWER
    assert check_after_answer("Sorry, i don't have that information").escalate
    assert not check_after_answer("No, passwords may not be shared.").escalate


def test_email_validation():
    assert escalation.is_valid_email("student@fredonia.edu")
    assert not escalation.is_valid_email("not an email")
    assert not escalation.is_valid_email("a@b")


def make_ticket(**kw):
    defaults = dict(reason=escalation.SECURITY_INCIDENT, priority="high", question="I was hacked",
                    contact_name="Sam", contact_email="sam@fredonia.edu",
                    transcript=[{"role": "user", "content": "I was hacked"}])
    return Ticket(**{**defaults, **kw})


def test_ticket_body_and_subject():
    t = make_ticket(bot_answer="Change your password.", sources=["SECTION II-C ..."])
    assert t.id.startswith("ITS-")
    assert t.subject.startswith("[URGENT]")
    body = t.body()
    for text in ("sam@fredonia.edu", "I was hacked", "Change your password.", "SECTION II-C", "USER:"):
        assert text in body
    assert not make_ticket(priority="normal").subject.startswith("[URGENT]")


def test_file_backend_appends_jsonl(tmp_path):
    path = tmp_path / "sub" / "tickets.jsonl"
    backend = FileBackend(str(path))
    t1, t2 = make_ticket(), make_ticket()
    assert route_ticket(t1, [backend]) == ["log"]
    route_ticket(t2, [backend])
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["id"] for r in rows] == [t1.id, t2.id]


def test_failing_backend_does_not_block_others(tmp_path):
    class Broken:
        name = "email"
        def send(self, ticket):
            raise ConnectionError("smtp down")
    delivered = route_ticket(make_ticket(), [Broken(), FileBackend(str(tmp_path / "t.jsonl"))])
    assert delivered == ["log"]


def test_email_backend_only_enabled_when_configured(monkeypatch):
    monkeypatch.delenv("ITS_HELPDESK_EMAIL", raising=False)
    monkeypatch.delenv("ESCALATION_SMTP_HOST", raising=False)
    assert EmailBackend.from_env() is None
    monkeypatch.setenv("ITS_HELPDESK_EMAIL", "helpdesk@example.edu")
    monkeypatch.setenv("ESCALATION_SMTP_HOST", "smtp.example.edu")
    backend = EmailBackend.from_env()
    assert backend.to_addr == "helpdesk@example.edu" and backend.port == 587


def test_email_backend_sends_message(monkeypatch):
    sent = {}

    class FakeSMTP:
        def __init__(self, host, port, timeout):
            sent["host"] = host
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def starttls(self): sent["tls"] = True
        def login(self, user, pw): sent["login"] = user
        def send_message(self, msg): sent["msg"] = msg

    monkeypatch.setattr(escalation.smtplib, "SMTP", FakeSMTP)
    backend = EmailBackend("helpdesk@example.edu", "smtp.example.edu", user="bot@example.edu", password="x")
    t = make_ticket()
    backend.send(t)
    msg = sent["msg"]
    assert sent["tls"] and sent["login"] == "bot@example.edu"
    assert msg["To"] == "helpdesk@example.edu"
    assert msg["Reply-To"] == "sam@fredonia.edu"
    assert t.id in msg["Subject"]
