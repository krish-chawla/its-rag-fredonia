import difflib
import os
from typing import List, Literal, Optional

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from scheduler import (
    DAYS,
    ROLES,
    SAMPLE_REQUESTS,
    SHIFT_TIMES,
    SHIFTS,
    StaffRequest,
    build_schedule,
    sample_coverage,
    sample_staff,
)

load_dotenv()

# Same local/deployed key handling as the other pages
try:
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except Exception:
    pass

st.set_page_config(page_title="Staff Scheduler", page_icon="🩺", layout="wide")
st.title("🩺 Healthcare Staff Scheduler")
st.caption(
    "Builds a weekly doctor and nurse schedule from staff preferences, coverage needs, "
    "and plain-English requests. Demo data: a fictional clinic."
)

with st.expander("How it works"):
    st.markdown(
        "1. **You enter the staff** — role, max hours per week, preferred shift, and regular days off.\n"
        "2. **You set coverage** — how many doctors and nurses each shift needs every day.\n"
        "3. **You type special requests in plain English** — e.g. *\"Nora can't work Friday\"*. "
        "The AI (GPT-4o-mini) converts them into structured rules and shows you how it understood each one.\n"
        "4. **A constraint solver (Google OR-Tools) builds the schedule.** It guarantees every hard rule "
        "is met — exact coverage, max hours, days off, at most one shift per day, and no 7am Day shift "
        "right after an overnight Night shift — and then maximizes preferences and fairness. "
        "If no valid schedule exists, it tells you why.\n\n"
        f"Shifts are 8 hours: Day {SHIFT_TIMES['Day']}, Evening {SHIFT_TIMES['Evening']}, "
        f"Night {SHIFT_TIMES['Night']}."
    )

# --- Session state defaults -------------------------------------------------
if "sched_staff" not in st.session_state:
    st.session_state.sched_staff = sample_staff()
    st.session_state.sched_coverage = sample_coverage()
    st.session_state.sched_requests_text = SAMPLE_REQUESTS
    st.session_state.sched_result = None
    st.session_state.sched_parsed = []
    st.session_state.sched_summary = ""

if st.button("↺ Reset to sample clinic data"):
    # The editors keep their own edit history under their keys, so those
    # have to be cleared too or old edits get re-applied to the fresh data.
    for k in ("staff_editor", "coverage_editor", "requests_box"):
        st.session_state.pop(k, None)
    st.session_state.sched_staff = sample_staff()
    st.session_state.sched_coverage = sample_coverage()
    st.session_state.sched_requests_text = SAMPLE_REQUESTS
    st.session_state.sched_result = None
    st.session_state.sched_parsed = []
    st.session_state.sched_summary = ""
    st.rerun()

# --- Step 1: staff roster ----------------------------------------------------
st.subheader("1. Staff roster")
st.caption("Edit any cell, or add/remove rows at the bottom. Days unavailable: comma-separated, e.g. `Sat, Sun`.")
staff_df = st.data_editor(
    st.session_state.sched_staff,
    num_rows="dynamic",
    width="stretch",
    key="staff_editor",
    column_config={
        "Role": st.column_config.SelectboxColumn(options=ROLES, required=True),
        "Max hours/week": st.column_config.NumberColumn(min_value=0, max_value=80, step=8),
        "Preferred shift": st.column_config.SelectboxColumn(options=SHIFTS + ["Any"], required=True),
        "Days unavailable": st.column_config.TextColumn(help="Comma-separated: Mon, Tue, Wed, Thu, Fri, Sat, Sun"),
    },
)

# --- Step 2: coverage ---------------------------------------------------------
st.subheader("2. Coverage needed (every day)")
coverage_df = st.data_editor(
    st.session_state.sched_coverage,
    width="stretch",
    hide_index=True,
    disabled=["Shift"],
    key="coverage_editor",
    column_config={
        "Doctors needed": st.column_config.NumberColumn(min_value=0, max_value=20, step=1),
        "Nurses needed": st.column_config.NumberColumn(min_value=0, max_value=50, step=1),
    },
)

# --- Step 3: free-text requests -----------------------------------------------
st.subheader("3. Special requests for this week")
requests_text = st.text_area(
    "One request per line, in plain English.",
    value=st.session_state.sched_requests_text,
    height=140,
    key="requests_box",
)


# --- LLM: free text -> structured requests -------------------------------------
class ParsedRequest(BaseModel):
    staff_name: str = Field(description="Exact name copied from the staff list")
    kind: Literal["unavailable", "avoid", "prefer", "max_shifts", "min_shifts"] = Field(
        description=(
            "unavailable = cannot work (hard rule: 'can't', 'is off', 'on vacation', 'has an appointment'); "
            "avoid = would rather not (soft: 'prefers not', 'if possible', 'would rather not', 'wants X off if possible'); "
            "prefer = would like to work it (soft); "
            "max_shifts = at most N shifts; min_shifts = at least N shifts"
        )
    )
    day: Optional[Literal["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]] = Field(
        default=None, description="Single day, or null for every day"
    )
    shift: Optional[Literal["Day", "Evening", "Night"]] = Field(
        default=None, description="Single shift, or null for every shift"
    )
    value: Optional[int] = Field(default=None, description="Number of shifts, only for max_shifts/min_shifts")
    source_text: str = Field(description="The part of the manager's text this rule came from")


class ParsedRequests(BaseModel):
    requests: List[ParsedRequest]


PARSE_PROMPT = ChatPromptTemplate.from_template("""
You convert a healthcare manager's scheduling requests into structured rules.

Staff list (use these exact names):
{staff_list}

Shifts: Day (7a-3p), Evening (3p-11p), Night (11p-7a). Days: Mon Tue Wed Thu Fri Sat Sun.

Rules:
- Match nicknames, first names, or last names to the exact staff name above.
- A request covering several days or shifts becomes several rules (one per day),
  e.g. "the weekend" = Sat and Sun; "weekdays" = Mon to Fri.
- "Can't", "unavailable", "off", "vacation", "appointment" = unavailable (hard).
- "If possible", "would rather not", "prefers not", "wants ... off if possible" = avoid (soft).
- Skip any request that names someone not in the staff list.

Manager's requests:
{requests_text}
""")


def parse_requests(text: str, names: list[str]) -> list[StaffRequest]:
    if not text.strip():
        return []
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(ParsedRequests, method="function_calling")
    result = (PARSE_PROMPT | llm).invoke(
        {"staff_list": "\n".join(f"- {n}" for n in names), "requests_text": text}
    )
    out = []
    for r in result.requests:
        name = r.staff_name
        if name not in names:  # safety net: snap near-misses to a real name
            close = difflib.get_close_matches(name, names, n=1, cutoff=0.5)
            if not close:
                continue
            name = close[0]
        out.append(StaffRequest(name, r.kind, r.day, r.shift, r.value, r.source_text))
    return out


SUMMARY_PROMPT = ChatPromptTemplate.from_template("""
You are helping a clinic manager review an automatically generated weekly schedule.
Write 4-6 short bullet points: who is at their max hours, any special requests that
could NOT be honored (compare the requests to the schedule), anyone working several
Night shifts, and anything the manager should double-check. Be specific with names
and days. Do not restate the whole schedule.

Requests as understood:
{requests}

Schedule (rows = staff, columns = days):
{schedule}

Hours per person:
{summary}
""")


def summarize(result, parsed) -> str:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    return (SUMMARY_PROMPT | llm).invoke(
        {
            "requests": "\n".join(r.describe() for r in parsed) or "(none)",
            "schedule": result.schedule.to_string(index=False),
            "summary": result.staff_summary.to_string(index=False),
        }
    ).content


# --- Generate ---------------------------------------------------------------------
if st.button("🗓️ Generate schedule", type="primary"):
    # Note: don't write staff_df back into sched_staff here. data_editor
    # already remembers the edits; feeding it the edited frame as new input
    # would apply the same edits twice (e.g. duplicate added rows).
    names = [n for n in staff_df["Name"].dropna().astype(str) if n.strip()]

    parsed = []
    with st.spinner("AI is reading your requests..."):
        try:
            parsed = parse_requests(requests_text, names)
        except Exception as e:
            st.warning(f"Couldn't interpret the special requests with the AI ({e}). Scheduling without them.")
    st.session_state.sched_parsed = parsed

    with st.spinner("Solver is building the schedule..."):
        st.session_state.sched_result = build_schedule(staff_df, coverage_df, parsed)

    st.session_state.sched_summary = ""
    if st.session_state.sched_result.ok:
        with st.spinner("AI is reviewing the schedule..."):
            try:
                st.session_state.sched_summary = summarize(st.session_state.sched_result, parsed)
            except Exception:
                st.session_state.sched_summary = ""

# --- Results -----------------------------------------------------------------------
result = st.session_state.sched_result
parsed = st.session_state.sched_parsed

if result is not None:
    st.divider()
    if parsed:
        st.subheader("How the AI understood your requests")
        st.dataframe(
            pd.DataFrame(
                [{"Rule": r.describe(), "From your text": r.source_text} for r in parsed]
            ),
            width="stretch",
            hide_index=True,
        )

    if not result.ok:
        st.error(result.message)
        for reason in result.diagnostics:
            st.markdown(f"- {reason}")
    else:
        st.success(result.message)
        st.subheader("Weekly schedule")

        shift_colors = {
            "Day": "background-color: #FFF3C4; color: #5C4400",
            "Evening": "background-color: #FFD9C2; color: #6B2E00",
            "Night": "background-color: #CBD5F5; color: #1E2A5A",
        }
        styled = result.schedule.style.map(lambda v: shift_colors.get(v, ""), subset=DAYS)
        st.dataframe(styled, width="stretch", hide_index=True)
        st.caption(" · ".join(f"**{s}** {t}" for s, t in SHIFT_TIMES.items()))

        if st.session_state.sched_summary:
            st.subheader("AI review")
            st.markdown(st.session_state.sched_summary)

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Hours per person")
            st.dataframe(result.staff_summary, width="stretch", hide_index=True)
        with c2:
            st.subheader("Coverage check")
            cov = result.coverage_check
            st.metric("Shifts fully staffed", f"{(cov['OK'] == '✅').sum()} / {len(cov)}")
            with st.expander("See every day / shift / role"):
                st.dataframe(cov, width="stretch", hide_index=True)

        st.download_button(
            "⬇️ Download schedule (CSV)",
            result.schedule.to_csv(index=False).encode("utf-8"),
            file_name="weekly_schedule.csv",
            mime="text/csv",
        )
