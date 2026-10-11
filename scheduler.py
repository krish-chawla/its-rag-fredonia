"""Staff-scheduling engine for the Healthcare Staff Scheduler page.

How the "AI scheduling" is split up, and why:
  * The LLM (in pages/2_Staff_Scheduler.py) only does the language part:
    it turns a manager's free-text requests ("Nora can't work Friday",
    "Dr. Patel prefers nights") into structured StaffRequest objects.
  * This module does the math part: an OR-Tools CP-SAT constraint solver
    builds the actual weekly schedule. LLMs are unreliable at satisfying
    dozens of interlocking hard constraints (coverage, max hours, rest
    rules) at once; a solver is guaranteed to either satisfy all of them
    or prove that it can't.

Nothing in here touches Streamlit or OpenAI, so it can be unit-tested
on its own (see the bottom of this file).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
from ortools.sat.python import cp_model

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
SHIFTS = ["Day", "Evening", "Night"]
SHIFT_TIMES = {"Day": "7a–3p", "Evening": "3p–11p", "Night": "11p–7a"}
SHIFT_HOURS = 8
ROLES = ["Doctor", "Nurse"]

# Objective weights (bigger = the solver cares more).
W_PREFERRED_SHIFT = 3     # staff member's standing preferred shift
W_REQUEST_PREFER = 4      # "X would like to work ..." from free-text requests
W_REQUEST_AVOID = 6       # "X would rather not work ..." (soft)
W_FAIRNESS = 2            # per shift away from that person's fair share


# ---------------------------------------------------------------------------
# Sample data (fictional clinic) so the page works the moment it loads
# ---------------------------------------------------------------------------

def sample_staff() -> pd.DataFrame:
    rows = [
        # name, role, max hours/week, preferred shift, days unavailable
        ("Dr. Anika Patel", "Doctor", 40, "Night", ""),
        ("Dr. Marcus Reed", "Doctor", 40, "Day", "Sun"),
        ("Dr. Sofia Alvarez", "Doctor", 32, "Day", "Sat, Sun"),
        ("Dr. James Okafor", "Doctor", 40, "Evening", ""),
        ("Dr. Lena Kowalski", "Doctor", 24, "Any", "Mon, Tue"),
        ("Nora Bianchi, RN", "Nurse", 40, "Day", ""),
        ("Tyler Brooks, RN", "Nurse", 40, "Night", ""),
        ("Priya Shah, RN", "Nurse", 32, "Day", "Wed"),
        ("Ethan Miller, RN", "Nurse", 40, "Evening", ""),
        ("Grace Kim, RN", "Nurse", 40, "Any", "Sat"),
        ("Daniel Ortiz, RN", "Nurse", 24, "Evening", "Mon, Thu"),
        ("Hannah Weiss, RN", "Nurse", 40, "Day", "Sun"),
        ("Marcus Lee, RN", "Nurse", 40, "Night", ""),
        ("Olivia Grant, RN", "Nurse", 32, "Any", "Fri"),
        ("Samuel Park, RN", "Nurse", 40, "Any", "Tue"),
    ]
    return pd.DataFrame(
        rows,
        columns=["Name", "Role", "Max hours/week", "Preferred shift", "Days unavailable"],
    )


def sample_coverage() -> pd.DataFrame:
    """Minimum staff needed on EVERY day, per shift."""
    return pd.DataFrame(
        {
            "Shift": SHIFTS,
            "Doctors needed": [1, 1, 1],
            "Nurses needed": [3, 2, 1],
        }
    )


SAMPLE_REQUESTS = (
    "Nora is at a conference on Friday, so she can't work that day.\n"
    "Dr. Patel would like the weekend nights.\n"
    "Tyler should not work more than 3 night shifts this week.\n"
    "Grace would rather not work evenings.\n"
    "Dr. Reed wants Wednesday off if possible."
)


# ---------------------------------------------------------------------------
# Structured requests (what the LLM produces from free text)
# ---------------------------------------------------------------------------

@dataclass
class StaffRequest:
    staff_name: str
    kind: str                     # unavailable | avoid | prefer | max_shifts | min_shifts
    day: Optional[str] = None     # one of DAYS, or None = every day
    shift: Optional[str] = None   # one of SHIFTS, or None = every shift
    value: Optional[int] = None   # number of shifts for max_shifts / min_shifts
    source_text: str = ""

    def describe(self) -> str:
        when = " ".join(x for x in [self.day or "", self.shift or ""] if x) or "any shift"
        if self.kind == "unavailable":
            return f"{self.staff_name} CANNOT work {when} (hard rule)"
        if self.kind == "avoid":
            return f"{self.staff_name} would rather not work {when} (soft preference)"
        if self.kind == "prefer":
            return f"{self.staff_name} would like to work {when} (soft preference)"
        if self.kind == "max_shifts":
            what = f"{self.shift} shifts" if self.shift else "shifts"
            return f"{self.staff_name}: at most {self.value} {what} this week (hard rule)"
        if self.kind == "min_shifts":
            what = f"{self.shift} shifts" if self.shift else "shifts"
            return f"{self.staff_name}: at least {self.value} {what} this week (hard rule)"
        return f"{self.staff_name}: {self.kind}"


@dataclass
class ScheduleResult:
    ok: bool
    message: str
    schedule: Optional[pd.DataFrame] = None        # rows = staff, cols = days, cell = shift or "—"
    coverage_check: Optional[pd.DataFrame] = None  # assigned vs needed per day/shift/role
    staff_summary: Optional[pd.DataFrame] = None   # hours, preference hits per person
    diagnostics: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_days(text) -> set:
    """'Sat, Sun' -> {'Sat', 'Sun'}. Accepts full names and any case."""
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return set()
    out = set()
    for part in str(text).replace(";", ",").replace("/", ",").split(","):
        token = part.strip()[:3].title()
        if token in DAYS:
            out.add(token)
    return out


def _matches(req: StaffRequest, d: str, s: str) -> bool:
    return (req.day is None or req.day == d) and (req.shift is None or req.shift == s)


def clean_staff(staff: pd.DataFrame) -> pd.DataFrame:
    staff = staff.copy()
    staff = staff[staff["Name"].astype(str).str.strip() != ""]
    staff = staff.dropna(subset=["Name", "Role"])
    staff["Max hours/week"] = pd.to_numeric(staff["Max hours/week"], errors="coerce").fillna(40).astype(int)
    staff["Preferred shift"] = staff["Preferred shift"].fillna("Any")
    staff["Days unavailable"] = staff["Days unavailable"].fillna("")
    return staff.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Feasibility diagnostics (plain-English reasons when no schedule exists)
# ---------------------------------------------------------------------------

def diagnose(staff: pd.DataFrame, coverage: pd.DataFrame, requests: list[StaffRequest],
             capacity_only: bool = False) -> list[str]:
    """Plain-English reasons a schedule is impossible. With capacity_only=True
    it runs only the fast counting checks and returns [] when they pass —
    used as a pre-check so obviously impossible inputs fail instantly
    instead of making the solver search until its time limit."""
    staff = clean_staff(staff)
    problems = []
    need = {(row["Shift"], "Doctor"): int(row["Doctors needed"]) for _, row in coverage.iterrows()}
    need.update({(row["Shift"], "Nurse"): int(row["Nurses needed"]) for _, row in coverage.iterrows()})

    for role in ROLES:
        members = staff[staff["Role"] == role]
        required = sum(need[(s, role)] for s in SHIFTS) * len(DAYS)
        capacity = 0
        for _, p in members.iterrows():
            cap = min(p["Max hours/week"] // SHIFT_HOURS, len(DAYS) - len(parse_days(p["Days unavailable"])))
            for r in requests:
                if r.staff_name == p["Name"] and r.kind == "max_shifts" and r.shift is None and r.value is not None:
                    cap = min(cap, r.value)
            capacity += max(cap, 0)
        if capacity < required:
            problems.append(
                f"{role}s: the week needs {required} {role.lower()} shifts, but the {len(members)} "
                f"{role.lower()}(s) on the roster can cover at most {capacity} given their max hours "
                f"and days off. Add staff, raise max hours, or lower the coverage minimums."
            )

        for d in DAYS:
            for s in SHIFTS:
                if capacity_only:
                    continue
                if need[(s, role)] == 0:
                    continue
                available = 0
                for _, p in members.iterrows():
                    if d in parse_days(p["Days unavailable"]):
                        continue
                    if any(r.staff_name == p["Name"] and r.kind == "unavailable" and _matches(r, d, s) for r in requests):
                        continue
                    available += 1
                if available < need[(s, role)]:
                    problems.append(
                        f"{d} {s}: needs {need[(s, role)]} {role.lower()}(s) but only {available} are available that shift."
                    )
    if not problems and not capacity_only:
        problems.append(
            "No single obvious shortage. The combination of rules (rest between night and day shifts, "
            "per-person shift limits, and days off) can't all be met at once. Try removing one "
            "hard request at a time to find the conflict."
        )
    return problems


# ---------------------------------------------------------------------------
# The solver
# ---------------------------------------------------------------------------

def build_schedule(
    staff: pd.DataFrame,
    coverage: pd.DataFrame,
    requests: list[StaffRequest] | None = None,
    time_limit_s: float = 10.0,
) -> ScheduleResult:
    requests = requests or []
    staff = clean_staff(staff)
    if staff.empty:
        return ScheduleResult(False, "Add at least one staff member first.")

    # Fast pre-check: if there simply aren't enough staff-shifts available,
    # say so immediately instead of letting the solver search for 10 s.
    shortages = diagnose(staff, coverage, requests, capacity_only=True)
    if shortages:
        return ScheduleResult(
            ok=False,
            message="No schedule can satisfy every hard rule. Here is what's blocking it:",
            diagnostics=shortages,
        )

    names = list(staff["Name"])
    model = cp_model.CpModel()

    # x[p, d, s] == 1  <=>  person p works shift s on day d
    x = {
        (p, d, s): model.NewBoolVar(f"x_{p}_{d}_{s}")
        for p in range(len(names))
        for d in range(len(DAYS))
        for s in range(len(SHIFTS))
    }

    # --- HARD RULE 1: exact coverage for every day/shift/role --------------
    # Exactly the number needed (not "at least"): otherwise the preference
    # bonuses would reward the solver for scheduling extra, unneeded shifts,
    # which costs the clinic paid hours.
    for _, row in coverage.iterrows():
        s = SHIFTS.index(row["Shift"])
        for role, col in [("Doctor", "Doctors needed"), ("Nurse", "Nurses needed")]:
            members = [p for p in range(len(names)) if staff.loc[p, "Role"] == role]
            for d in range(len(DAYS)):
                model.Add(sum(x[p, d, s] for p in members) == int(row[col]))

    objective = []
    total_shifts = {p: sum(x[p, d, s] for d in range(len(DAYS)) for s in range(len(SHIFTS))) for p in range(len(names))}

    for p, name in enumerate(names):
        person = staff.loc[p]
        # --- HARD RULE 2: at most one shift per day ------------------------
        for d in range(len(DAYS)):
            model.Add(sum(x[p, d, s] for s in range(len(SHIFTS))) <= 1)

        # --- HARD RULE 3: max hours per week -------------------------------
        model.Add(total_shifts[p] * SHIFT_HOURS <= int(person["Max hours/week"]))

        # --- HARD RULE 4: standing days unavailable ------------------------
        for day in parse_days(person["Days unavailable"]):
            d = DAYS.index(day)
            for s in range(len(SHIFTS)):
                model.Add(x[p, d, s] == 0)

        # --- HARD RULE 5: rest — no Day shift (7a) right after a Night ----
        # shift that ended at 7a the same morning.
        night, day_shift = SHIFTS.index("Night"), SHIFTS.index("Day")
        for d in range(len(DAYS) - 1):
            model.Add(x[p, d, night] + x[p, d + 1, day_shift] <= 1)

        # --- SOFT: standing preferred shift --------------------------------
        pref = person["Preferred shift"]
        if pref in SHIFTS:
            s = SHIFTS.index(pref)
            objective += [W_PREFERRED_SHIFT * x[p, d, s] for d in range(len(DAYS))]

    # --- Requests parsed from the manager's free text ----------------------
    for req in requests:
        if req.staff_name not in names:
            continue
        p = names.index(req.staff_name)
        cells = [
            x[p, d, s]
            for d, dn in enumerate(DAYS)
            for s, sn in enumerate(SHIFTS)
            if _matches(req, dn, sn)
        ]
        if req.kind == "unavailable":
            for c in cells:
                model.Add(c == 0)
        elif req.kind == "avoid":
            objective += [-W_REQUEST_AVOID * c for c in cells]
        elif req.kind == "prefer":
            objective += [W_REQUEST_PREFER * c for c in cells]
        elif req.kind in ("max_shifts", "min_shifts") and req.value is not None:
            # day is ignored for counts; shift (if given) narrows what's counted
            counted = [x[p, d, s] for d in range(len(DAYS)) for s, sn in enumerate(SHIFTS)
                       if req.shift is None or req.shift == sn]
            if req.kind == "max_shifts":
                model.Add(sum(counted) <= int(req.value))
            else:
                model.Add(sum(counted) >= int(req.value))

    # --- SOFT: fairness — keep everyone near their fair share --------------
    # Fair share = that role's required shifts split in proportion to each
    # person's max hours (so a 24-hr part-timer gets fewer than a 40-hr).
    for role in ROLES:
        members = [p for p in range(len(names)) if staff.loc[p, "Role"] == role]
        if not members:
            continue
        col = "Doctors needed" if role == "Doctor" else "Nurses needed"
        role_required = int(coverage[col].sum()) * len(DAYS)
        total_max = sum(int(staff.loc[p, "Max hours/week"]) for p in members) or 1
        for p in members:
            target = round(role_required * int(staff.loc[p, "Max hours/week"]) / total_max)
            dev = model.NewIntVar(0, len(DAYS), f"dev_{p}")
            model.AddAbsEquality(dev, total_shifts[p] - target)
            objective.append(-W_FAIRNESS * dev)

    model.Maximize(sum(objective))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.num_workers = 4
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return ScheduleResult(
            ok=False,
            message="No schedule can satisfy every hard rule. Here is what's blocking it:",
            diagnostics=diagnose(staff, coverage, requests),
        )

    # --- Build the output tables -------------------------------------------
    grid = []
    summary = []
    for p, name in enumerate(names):
        row = {"Name": name, "Role": staff.loc[p, "Role"]}
        n = 0
        pref_hits = 0
        for d, dn in enumerate(DAYS):
            cell = "—"
            for s, sn in enumerate(SHIFTS):
                if solver.Value(x[p, d, s]):
                    cell = sn
                    n += 1
                    if staff.loc[p, "Preferred shift"] in (sn, "Any"):
                        pref_hits += 1
            row[dn] = cell
        grid.append(row)
        summary.append(
            {
                "Name": name,
                "Role": staff.loc[p, "Role"],
                "Shifts": n,
                "Hours": n * SHIFT_HOURS,
                "Max hours": int(staff.loc[p, "Max hours/week"]),
                "On preferred shift": f"{pref_hits}/{n}" if n else "—",
            }
        )

    cov_rows = []
    for _, row in coverage.iterrows():
        s = SHIFTS.index(row["Shift"])
        for d, dn in enumerate(DAYS):
            for role, col in [("Doctor", "Doctors needed"), ("Nurse", "Nurses needed")]:
                assigned = sum(
                    solver.Value(x[p, d, s]) for p in range(len(names)) if staff.loc[p, "Role"] == role
                )
                cov_rows.append(
                    {"Day": dn, "Shift": row["Shift"], "Role": role,
                     "Needed": int(row[col]), "Assigned": assigned,
                     "OK": "✅" if assigned >= int(row[col]) else "❌"}
                )

    quality = "optimal" if status == cp_model.OPTIMAL else "good (time limit reached before proving optimal)"
    return ScheduleResult(
        ok=True,
        message=f"Schedule found — {quality}. Every coverage minimum and hard rule is met.",
        schedule=pd.DataFrame(grid),
        coverage_check=pd.DataFrame(cov_rows),
        staff_summary=pd.DataFrame(summary),
    )


# ---------------------------------------------------------------------------
# Quick self-test:  python scheduler.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    reqs = [
        StaffRequest("Nora Bianchi, RN", "unavailable", day="Fri"),
        StaffRequest("Dr. Anika Patel", "prefer", day="Sat", shift="Night"),
        StaffRequest("Dr. Anika Patel", "prefer", day="Sun", shift="Night"),
        StaffRequest("Tyler Brooks, RN", "max_shifts", shift="Night", value=3),
        StaffRequest("Grace Kim, RN", "avoid", shift="Evening"),
        StaffRequest("Dr. Marcus Reed", "avoid", day="Wed"),
    ]
    res = build_schedule(sample_staff(), sample_coverage(), reqs)
    print(res.message)
    if res.ok:
        print(res.schedule.to_string(index=False))
        print(res.staff_summary.to_string(index=False))
        assert (res.coverage_check["OK"] == "✅").all()
    else:
        print("\n".join(res.diagnostics))
