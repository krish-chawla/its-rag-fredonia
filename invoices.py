"""Invoice-tracking logic for the Invoice Tracker page.

All money math (balances, days overdue, late fees) happens here in plain
pandas, NOT in the LLM. The chat assistant on the page calls these
functions as tools and only phrases the answer, so the numbers it reports
are always exactly what this code computed.

No Streamlit or OpenAI imports, so it can be tested on its own:
    python invoices.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

# Columns a business must provide when uploading its own CSV
REQUIRED_COLUMNS = [
    "Invoice ID", "Customer", "Issue date", "Due date", "Amount",
    "Amount paid", "Payment date", "Refund requested", "Refund status",
    "Refund amount", "Refund reason",
]
REFUND_STATUSES = ["", "Pending", "Approved", "Denied", "Refunded"]


@dataclass
class LateFeePolicy:
    grace_days: int = 5          # no fee if paid within this many days after the due date
    flat_fee: float = 25.0       # one-time fee once the grace period ends
    monthly_rate: float = 0.015  # 1.5% of the unpaid balance per 30 days (or part) overdue

    def describe(self) -> str:
        return (
            f"Invoices are due on their due date. After a {self.grace_days}-day grace period, "
            f"a one-time ${self.flat_fee:,.2f} late fee applies, plus {self.monthly_rate * 100:.1f}% "
            f"of the unpaid balance for every 30 days (or part of 30 days) the invoice is overdue. "
            f"Invoices paid late are charged the same way, based on the date they were paid."
        )

    def fee(self, balance: float, days_overdue: int) -> float:
        if days_overdue <= self.grace_days or balance <= 0:
            return 0.0
        periods = math.ceil(days_overdue / 30)
        return round(self.flat_fee + balance * self.monthly_rate * periods, 2)


# ---------------------------------------------------------------------------
# Sample data — dates are relative to "today" so the demo never goes stale
# ---------------------------------------------------------------------------

def sample_invoices(today: date | None = None) -> pd.DataFrame:
    t = today or date.today()

    def d(offset):
        if offset is None or (isinstance(offset, float) and math.isnan(offset)):
            return None
        return t + timedelta(days=int(offset))

    # id, customer, issued(offset), due(offset), amount, paid, paid_on(offset),
    # refund_requested, refund_status, refund_amount, refund_reason
    rows = [
        ("INV-1001", "Lakeshore Community Clinic", -95, -65, 480.00, 480.00, -70, False, "", 0, ""),
        ("INV-1002", "Harbor View Hotel", -90, -60, 1250.00, 1250.00, -48, False, "", 0, ""),
        ("INV-1003", "Chautauqua Events Co.", -80, -50, 2200.00, 1000.00, -45, False, "", 0, ""),
        ("INV-1004", "Main Street Dental", -75, -45, 160.00, 0.00, None, False, "", 0, ""),
        ("INV-1005", "Blue Lake Wedding Venue", -70, -40, 3400.00, 3400.00, -41, True, "Pending", 400.00,
         "Centerpieces arrived with wilted hydrangeas"),
        ("INV-1006", "Lakeshore Community Clinic", -60, -30, 480.00, 480.00, -31, False, "", 0, ""),
        ("INV-1007", "Westfield Realty Group", -55, -25, 640.00, 0.00, None, False, "", 0, ""),
        ("INV-1008", "Harbor View Hotel", -50, -20, 1250.00, 1250.00, -22, True, "Refunded", 125.00,
         "One lobby arrangement delivered a day late"),
        ("INV-1009", "Campus Events Office (demo)", -45, -15, 900.00, 450.00, -14, False, "", 0, ""),
        ("INV-1010", "Main Street Dental", -40, -10, 160.00, 0.00, None, False, "", 0, ""),
        ("INV-1011", "Chautauqua Events Co.", -35, -5, 1750.00, 0.00, None, True, "Denied", 1750.00,
         "Event cancelled by customer within 30 days (non-refundable deposit)"),
        ("INV-1012", "Lakeshore Community Clinic", -30, -1, 480.00, 0.00, None, False, "", 0, ""),
        ("INV-1013", "Blue Lake Wedding Venue", -20, 10, 2800.00, 1400.00, -18, False, "", 0, ""),
        ("INV-1014", "Westfield Realty Group", -15, 15, 320.00, 0.00, None, True, "Approved", 80.00,
         "Ordered 8 arrangements, 7 delivered"),
        ("INV-1015", "Harbor View Hotel", -10, 20, 1250.00, 0.00, None, False, "", 0, ""),
        ("INV-1016", "Main Street Dental", -5, 25, 160.00, 160.00, -2, False, "", 0, ""),
    ]
    df = pd.DataFrame(rows, columns=REQUIRED_COLUMNS)
    df["Issue date"] = [d(o) for o in df["Issue date"]]
    df["Due date"] = [d(o) for o in df["Due date"]]
    df["Payment date"] = [d(o) for o in df["Payment date"]]
    return df


def load_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing these columns: {', '.join(missing)}")
    for c in ["Issue date", "Due date", "Payment date"]:
        df[c] = pd.to_datetime(df[c], errors="coerce").dt.date
    for c in ["Amount", "Amount paid", "Refund amount"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    df["Refund requested"] = df["Refund requested"].astype(str).str.strip().str.lower().isin(
        ["true", "yes", "y", "1"]
    )
    df["Refund status"] = df["Refund status"].fillna("").astype(str)
    df["Refund reason"] = df["Refund reason"].fillna("").astype(str)
    return df


def template_csv() -> bytes:
    return sample_invoices().head(3).to_csv(index=False).encode("utf-8")


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute(df: pd.DataFrame, as_of: date, policy: LateFeePolicy) -> pd.DataFrame:
    """Adds Balance, Status, Days overdue, Late fee, and Total due columns."""
    out = df.copy()
    balances, statuses, overdue_days, fees = [], [], [], []
    for _, r in out.iterrows():
        amount = float(r["Amount"])
        paid = float(r["Amount paid"])
        balance = round(max(amount - paid, 0.0), 2)
        due = r["Due date"]
        pay_date = r["Payment date"] if isinstance(r["Payment date"], date) else None

        if balance == 0:
            # fully paid: was it late?
            days = (pay_date - due).days if (pay_date and due) else 0
            fee = policy.fee(amount, days)
            status = "Paid late" if days > policy.grace_days else "Paid"
            days = max(days, 0)
        else:
            days = max((as_of - due).days, 0) if due else 0
            fee = policy.fee(balance, days)
            if days > 0:
                status = "Overdue" if paid == 0 else "Partially paid – overdue"
                if days <= policy.grace_days:
                    status += " (grace period)"
            else:
                status = "Open" if paid == 0 else "Partially paid"

        if str(r["Refund status"]) == "Refunded":
            status += " (refund issued)"

        balances.append(balance)
        statuses.append(status)
        overdue_days.append(days)
        fees.append(fee)

    out["Balance"] = balances
    out["Status"] = statuses
    out["Days overdue"] = overdue_days
    out["Late fee"] = fees
    out["Total due now"] = [
        round(b + f, 2) if b > 0 else (f if s.startswith("Paid late") else 0.0)
        for b, f, s in zip(balances, fees, statuses)
    ]
    return out


def kpis(c: pd.DataFrame) -> dict:
    open_mask = c["Balance"] > 0
    overdue_mask = c["Status"].str.contains("overdue", case=False)
    refund_open = c["Refund requested"] & c["Refund status"].isin(["Pending", "Approved"])
    return {
        "invoices": int(len(c)),
        "outstanding_balance": round(float(c.loc[open_mask, "Balance"].sum()), 2),
        "overdue_count": int(overdue_mask.sum()),
        "overdue_balance": round(float(c.loc[overdue_mask, "Balance"].sum()), 2),
        "late_fees": round(float(c["Late fee"].sum()), 2),
        "open_refund_requests": int(refund_open.sum()),
        "open_refund_amount": round(float(c.loc[refund_open, "Refund amount"].sum()), 2),
        "paid_count": int((~open_mask).sum()),
    }


def invoice_record(c: pd.DataFrame, invoice_id: str) -> dict | None:
    match = c[c["Invoice ID"].str.upper() == invoice_id.strip().upper()]
    if match.empty:
        return None
    return _to_jsonable(match.iloc[0].to_dict())


def search(c: pd.DataFrame, customer: str | None = None, status: str | None = None,
           refund_requested: bool | None = None) -> list[dict]:
    m = pd.Series(True, index=c.index)
    if customer:
        m &= c["Customer"].str.contains(customer, case=False, na=False)
    if status:
        m &= c["Status"].str.contains(status, case=False, na=False)
    if refund_requested is not None:
        m &= c["Refund requested"] == refund_requested
    cols = ["Invoice ID", "Customer", "Due date", "Amount", "Balance", "Status",
            "Days overdue", "Late fee", "Refund requested", "Refund status", "Refund amount"]
    return [_to_jsonable(r) for r in c.loc[m, cols].to_dict("records")]


def _to_jsonable(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, date):
            out[k] = v.isoformat()
        elif v is None or (isinstance(v, float) and math.isnan(v)):
            out[k] = None
        elif hasattr(v, "item"):
            out[k] = v.item()
        else:
            out[k] = v
    return out


if __name__ == "__main__":
    today = date(2026, 10, 11)
    pol = LateFeePolicy()
    c = compute(sample_invoices(today), today, pol)
    print(c[["Invoice ID", "Customer", "Balance", "Status", "Days overdue", "Late fee", "Total due now"]].to_string(index=False))
    print(kpis(c))
    # Hand-checked cases:
    r = invoice_record(c, "inv-1004")       # $160 unpaid, 45 days overdue -> 25 + 160*.015*2 = 29.80
    assert r["Late fee"] == 29.80 and r["Status"] == "Overdue", r
    r = invoice_record(c, "INV-1012")       # 1 day overdue, inside grace -> no fee
    assert r["Late fee"] == 0.0, r
    r = invoice_record(c, "INV-1002")       # paid 12 days late on $1250 -> 25 + 1250*.015*1 = 43.75
    assert r["Status"] == "Paid late" and r["Late fee"] == 43.75, r
    r = invoice_record(c, "INV-1003")       # $1200 balance, 50 days -> 25 + 1200*.015*2 = 61.00
    assert r["Late fee"] == 61.00, r
    print("all checks passed")
