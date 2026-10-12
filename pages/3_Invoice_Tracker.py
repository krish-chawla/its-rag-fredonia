import json
import os
from datetime import date
from typing import Optional

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from escalation import should_escalate, show_escalation
from invoices import (
    LateFeePolicy,
    compute,
    invoice_record,
    kpis,
    load_csv,
    sample_invoices,
    search,
    template_csv,
)

load_dotenv()

# Same local/deployed key handling as the other pages
try:
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except Exception:
    pass

# PLACEHOLDER contact info for the fictional demo business's billing desk.
CONTACT_NAME = "the billing department"
CONTACT_LINE = "📞 (555) 123-4567 ext. 2  |  ✉️ billing@fredoniaflowers.example"

st.set_page_config(page_title="Invoice Tracker", page_icon="🧾", layout="wide")
st.title("🧾 Invoice Tracker")
st.caption(
    "Track payments, refund requests, and late fees — and ask questions in plain English. "
    "Demo data: Fredonia Flowers' corporate accounts (fictional)."
)

# --- Sidebar: data source + late-fee policy -----------------------------------
with st.sidebar:
    st.header("Invoice data")
    uploaded = st.file_uploader("Use your own invoices (CSV)", type="csv")
    st.download_button("⬇️ CSV template", template_csv(), "invoice_template.csv", "text/csv")
    as_of = st.date_input("Calculate as of", value=date.today())

    st.header("Late-fee policy")
    grace = st.number_input("Grace period (days)", 0, 60, 5)
    flat = st.number_input("One-time late fee ($)", 0.0, 500.0, 25.0, step=5.0)
    rate = st.number_input("Monthly interest on unpaid balance (%)", 0.0, 5.0, 1.5, step=0.25)
    st.caption(
        "Late fees must be in the customer's contract or invoice terms, and some states cap them — "
        "check local rules before charging."
    )

policy = LateFeePolicy(grace_days=int(grace), flat_fee=float(flat), monthly_rate=float(rate) / 100)

if uploaded is not None:
    try:
        raw = load_csv(uploaded)
    except ValueError as e:
        st.error(str(e))
        st.stop()
else:
    raw = sample_invoices()

data = compute(raw, as_of, policy)
k = kpis(data)

# --- Dashboard ------------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Outstanding balance", f"${k['outstanding_balance']:,.2f}", f"{k['invoices'] - k['paid_count']} unpaid invoices", delta_color="off")
c2.metric("Overdue", f"${k['overdue_balance']:,.2f}", f"{k['overdue_count']} invoices", delta_color="off")
c3.metric("Late fees accrued", f"${k['late_fees']:,.2f}")
c4.metric("Open refund requests", f"{k['open_refund_requests']}", f"${k['open_refund_amount']:,.2f}", delta_color="off")

with st.expander("Late-fee policy in effect"):
    st.write(policy.describe())

f1, f2 = st.columns([2, 1])
with f1:
    status_filter = st.multiselect(
        "Filter by status",
        sorted(data["Status"].unique()),
        placeholder="All statuses",
    )
with f2:
    refunds_only = st.checkbox("Only invoices with a refund request")

view = data
if status_filter:
    view = view[view["Status"].isin(status_filter)]
if refunds_only:
    view = view[view["Refund requested"]]


def color_status(v: str) -> str:
    v = str(v)
    if "overdue" in v.lower() and "grace" not in v.lower():
        return "background-color: #FAD4D4; color: #6B0F0F"
    if "grace" in v.lower() or "partially" in v.lower() or v == "Paid late":
        return "background-color: #FFF0C2; color: #5C4400"
    if v.startswith("Paid"):
        return "background-color: #D3F2DC; color: #0E4A22"
    return ""


cols = ["Invoice ID", "Customer", "Issue date", "Due date", "Amount", "Amount paid", "Balance",
        "Status", "Days overdue", "Late fee", "Total due now", "Refund requested", "Refund status",
        "Refund amount", "Refund reason"]
money = ["Amount", "Amount paid", "Balance", "Late fee", "Total due now", "Refund amount"]
st.dataframe(
    view[cols].style.map(color_status, subset=["Status"]).format({m: "${:,.2f}" for m in money}),
    width="stretch",
    hide_index=True,
)

# --- Chat assistant (LLM + tools) -----------------------------------------------
st.divider()
st.subheader("💬 Ask about your invoices")
st.caption(
    "Try: *Has INV-1004 been paid?* · *Which customers have requested refunds?* · "
    "*What's the penalty on Chautauqua Events' invoices?* · *How much is overdue in total?*"
)


def make_tools(df: pd.DataFrame):
    """Tools the model can call. Each one reads the already-computed table,
    so every number in an answer comes from invoices.py, not the LLM."""

    @tool
    def get_invoice(invoice_id: str) -> str:
        """Look up one invoice by ID (e.g. INV-1004): payment status, balance, due date,
        days overdue, late fee, total due now, and refund request details."""
        rec = invoice_record(df, invoice_id)
        return json.dumps(rec) if rec else f"No invoice found with ID {invoice_id}."

    @tool
    def find_invoices(customer: Optional[str] = None, status: Optional[str] = None,
                      refund_requested: Optional[bool] = None) -> str:
        """Search invoices. customer = part of a customer name; status = one of
        'Paid', 'Paid late', 'Open', 'Overdue', 'Partially paid', 'grace period';
        refund_requested = true to list only invoices with a refund request.
        Leave all empty to list every invoice."""
        rows = search(df, customer, status, refund_requested)
        return json.dumps(rows) if rows else "No matching invoices."

    @tool
    def account_totals() -> str:
        """Totals across all invoices: outstanding balance, overdue count and amount,
        late fees accrued, and open refund requests."""
        return json.dumps(kpis(df))

    @tool
    def late_fee_policy() -> str:
        """The late-fee / penalty rules currently in effect."""
        return policy.describe()

    return [get_invoice, find_invoices, account_totals, late_fee_policy]


SYSTEM = f"""You are an accounts-receivable assistant for a small business.
Today's date for all calculations is {as_of.isoformat()}.
Always use the tools to look up invoice data; never invent invoice IDs, amounts, dates, or fees.
Report money exactly as the tools return it, formatted like $1,234.56.
When asked about a penalty, give the late fee and briefly explain how the policy produced it.
Use a short markdown table when listing more than two invoices.
If a question can't be answered from the tools, say "I don't have that information."
"""


def answer(question: str, history: list[dict]) -> str:
    tools = make_tools(data)
    by_name = {t.name: t for t in tools}
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)

    messages = [SystemMessage(SYSTEM)]
    for m in history[-6:]:  # a little context for follow-ups like "and the penalty?"
        messages.append(HumanMessage(m["content"]) if m["role"] == "user" else AIMessage(m["content"]))
    messages.append(HumanMessage(question))

    for _ in range(5):  # tool-calling loop: model asks for data, we run it, repeat
        reply = llm.invoke(messages)
        messages.append(reply)
        if not reply.tool_calls:
            return reply.content
        for call in reply.tool_calls:
            result = by_name[call["name"]].invoke(call["args"])
            messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
    return "Sorry, I couldn't finish looking that up. Please try rephrasing."


if "invoice_messages" not in st.session_state:
    st.session_state.invoice_messages = []

for msg in st.session_state.invoice_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask about an invoice, a customer, refunds, or penalties...")
if question:
    history = list(st.session_state.invoice_messages)
    st.session_state.invoice_messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Looking it up..."):
            response = answer(question, history)
            st.markdown(response)
            if should_escalate(question, response):
                show_escalation(CONTACT_NAME, CONTACT_LINE)
    st.session_state.invoice_messages.append({"role": "assistant", "content": response})
