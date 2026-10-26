import csv
import io
import math
import re
import pandas as pd
from pathlib import Path
import ollama
import warnings
import time
import datetime

from inventory_logic import (
    DEFAULT_LEAD_TIME_DAYS,
    DEFAULT_SERVICE_LEVEL,
    RISK_BOARD_RENAME,
    SERVICE_LEVELS,
    annual_demand_from_weekly,
    apply_service_level,
    attach_unit_cost,
    classify_abc,
    classify_cover_risk,
    describe_shock,
    forecast_mean_demand,
    format_weeks_of_cover,
    sort_risk_table,
    weekly_series,
    z_for_service_level,
)

warnings.filterwarnings("ignore")

# Directory setup. Cloud disks can reject writes; the UI still runs from memory.
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data" / "weekly"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "reports"
GEMMA_MODEL = "gemma2:2b"
# A missing local model must fail fast instead of hanging a public demo.
GEMMA_TIMEOUT_SECONDS = 8.0


def _ensure_dir(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        return True
    except OSError:
        return False


def _write_text(path: Path, text: str) -> bool:
    if not _ensure_dir(path.parent):
        return False
    try:
        path.write_text(text, encoding="utf-8")
        return True
    except OSError:
        return False


def _write_csv(frame: pd.DataFrame, path: Path) -> bool:
    if not _ensure_dir(path.parent):
        return False
    try:
        frame.to_csv(path, index=False)
        return True
    except OSError:
        return False


_ensure_dir(OUTPUT_DIR)


def _gemma_chat(messages, options=None):
    """Call local Gemma 2. Times out quickly when Ollama is not running."""
    client = ollama.Client(timeout=GEMMA_TIMEOUT_SECONDS)
    kwargs = {"model": GEMMA_MODEL, "messages": messages}
    # Copilot answers pass a token cap so a reply can finish inside the timeout.
    if options:
        kwargs["options"] = options
    return client.chat(**kwargs)

# Default service-level factor (95%). Other levels live in SERVICE_LEVELS.
Z_SCORE = SERVICE_LEVELS[DEFAULT_SERVICE_LEVEL]


# ====================== DATA INGESTION & CLEANING ======================
def load_and_clean_weekly_data(filename: str):
    file_path = DATA_DIR / filename
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return None

    df = pd.read_csv(file_path)
    df["week"] = pd.to_datetime(df["week"], errors="coerce")
    df = df.dropna(subset=["sku", "inventory_level", "consumption"])
    df = df.sort_values(["sku", "week"])

    df["daily_consumption"] = df["consumption"] / 7.0
    df["days_of_supply"] = df["inventory_level"] / df["daily_consumption"].replace(0, 1)

    print(f"✅ Loaded {len(df)} operational record(s).")
    return df


def _latest_number(sku_data: pd.DataFrame, column: str, default: float) -> float:
    if column not in sku_data.columns or sku_data.empty:
        return float(default)
    value = pd.to_numeric(sku_data[column], errors="coerce").iloc[-1]
    if pd.isna(value):
        return float(default)
    return float(value)


def _latest_text(sku_data: pd.DataFrame, column: str, default=""):
    if column not in sku_data.columns or sku_data.empty:
        return default
    value = sku_data[column].iloc[-1]
    if pd.isna(value):
        return default
    return value


# ====================== ADVANCED FORECASTING & METRICS ======================
def forecast_and_stockout(
    df,
    forecast_weeks: int = 8,
    service_level: str = DEFAULT_SERVICE_LEVEL,
):
    """Forecast each SKU and attach safety stock, cover risk, and ABC class.

    Safety stock and the reorder point use `service_level` (90%, 95%, or 99%).
    ABC class is assigned later from annual usage value, not from volume quantiles.
    """
    z_for_service_level(service_level)
    priced, unit_cost_source = attach_unit_cost(df)
    results = []

    for sku in priced["sku"].unique():
        sku_data = priced[priced["sku"] == sku].copy()
        sku_data["week"] = pd.to_datetime(sku_data["week"], errors="coerce")
        sku_data = sku_data.sort_values("week")
        series = weekly_series(sku_data)

        current_inv = round(_latest_number(sku_data, "inventory_level", 0.0), 1)
        # Round before cover and days-of-supply so a displayed 0.0 forecast
        # takes the zero-demand path instead of dividing by a hidden fraction.
        forecast_demand = round(
            forecast_mean_demand(series["consumption"], forecast_weeks), 1
        )
        weekly_std = (
            float(series["consumption"].std(ddof=1)) if len(series) > 1 else 0.0
        )
        if pd.isna(weekly_std):
            weekly_std = 0.0
        avg_weekly = float(series["consumption"].mean()) if len(series) else 0.0
        if pd.isna(avg_weekly):
            avg_weekly = 0.0

        lead_time = _latest_number(sku_data, "lead_time_days", DEFAULT_LEAD_TIME_DAYS)
        lt_weeks = lead_time / 7.0
        cover = classify_cover_risk(current_inv, forecast_demand, lt_weeks)

        daily_demand = forecast_demand / 7.0
        days_to_stockout = current_inv / daily_demand if daily_demand > 0 else 999.0

        annual = annual_demand_from_weekly(series["consumption"])
        unit_cost = _latest_number(sku_data, "unit_cost", 0.0)
        usage_value = max(annual, 0.0) * max(unit_cost, 0.0)

        fill_rate = _latest_number(sku_data, "vendor_fill_rate", 0.0)

        results.append(
            {
                "sku": sku,
                "current_inventory": current_inv,
                "forecast_weekly_demand": forecast_demand,
                "avg_weekly_demand": round(max(avg_weekly, 0.0), 4),
                "weekly_demand_sd": round(max(weekly_std, 0.0), 4),
                "lead_time_days": lead_time,
                "lead_time_weeks": round(lt_weeks, 4),
                "days_to_stockout": round(days_to_stockout, 1),
                "weeks_of_cover": cover["weeks_of_cover"],
                "stockout_risk": cover["stockout_risk"],
                "risk_reason": cover["risk_reason"],
                "annual_demand": round(annual, 2),
                "unit_cost": round(unit_cost, 2),
                "usage_value": round(usage_value, 2),
                "vendor_fill_rate": round(fill_rate, 2),
                "vendor_name": _latest_text(sku_data, "vendor_name", ""),
                "critical_item": _latest_text(sku_data, "critical", ""),
            }
        )

    summary = pd.DataFrame(results)
    if summary.empty:
        summary.attrs["unit_cost_source"] = unit_cost_source
        return summary

    summary = apply_service_level(summary, service_level)
    classified = classify_abc(summary[["sku", "usage_value"]])
    summary = summary.merge(
        classified[["sku", "abc_class", "usage_cum_pct"]],
        on="sku",
        how="left",
    )
    summary.attrs["unit_cost_source"] = unit_cost_source
    summary = sort_risk_table(summary)
    summary.attrs["unit_cost_source"] = unit_cost_source
    return summary


# ====================== FAST LLM REPORT GENERATION ======================
def generate_llm_report(full_summary):
    exception_summary = full_summary[
        full_summary["stockout_risk"].isin(["CRITICAL (Below ROP)", "High"])
    ]

    if exception_summary.empty:
        exception_summary = full_summary.head(10)

    summary_text = exception_summary.to_string(index=False)

    prompt = f"""You are a senior Supply Chain Inventory Management Specialist.
Analyze the following high-priority inventory exceptions (Reorder Points breached or high stockout risk).

Use ONLY this data:
{summary_text}

Write a clean Markdown report with these precise operational sections:
# Corporate Inventory Health & Optimization Report
## Executive Summary (Highlight operational health and cash flow efficiency)
## Critical Reorder Triggers (Items currently below their statistical Reorder Point)
## Stockout Risks & Supplier Vulnerabilities (Factoring in vendor fill rates)
## Prioritized Action Plan for Inventory Associates"""

    print("🤖 Processing telemetry exceptions with Gemma 2 (2B)...")
    start = time.time()
    try:
        response = _gemma_chat([{"role": "user", "content": prompt}])
        report = response["message"]["content"]
    except Exception as e:
        report = (
            f"# Operational Health Report\n\n"
            f"*Error connecting to Gemma 2 model: {str(e)}*\n\n"
            f"Forecasts and the risk board are still available without the model."
        )

    report_path = OUTPUT_DIR / "Inventory_Health_Risk_Report.md"
    if _write_text(report_path, report):
        print(f"✅ Report compiled in {time.time() - start:.1f} seconds → {report_path}")
    else:
        print(f"✅ Report ready in memory in {time.time() - start:.1f} seconds (disk write skipped).")
    return report


def generate_scenario_narrative(
    shock_df: pd.DataFrame,
    extra_lead_weeks: float,
    spike_pct: float,
    spike_weeks: float,
) -> dict:
    """Ask Gemma 2 for a short shock narrative. Fall back to the calculated summary.

    Returns source "gemma2:2b" or "rule". The rule-based text is always in
    `fallback` so the UI can show the numbers even when the model answers.
    """
    fallback = describe_shock(shock_df, extra_lead_weeks, spike_pct, spike_weeks)
    focus = shock_df.head(12)
    columns = [
        column
        for column in (
            "sku",
            "stockout_risk_before",
            "stockout_risk_after",
            "projected_stockout_week",
            "stockout_probability",
            "emergency_order_qty",
        )
        if column in focus.columns
    ]
    table = focus[columns].to_string(index=False) if len(focus) else "(no SKUs)"
    prompt = f"""You are a senior inventory planner. Write two short paragraphs about this supply shock.
Use ONLY the numbers below. Do not invent SKUs or quantities.
Name which SKUs get worse, the stockout probability, and the emergency order quantity.

Scenario: {extra_lead_weeks:.0f} extra weeks of supplier delay, {spike_pct:.0f}% demand spike for {spike_weeks:.0f} weeks.

Calculated summary:
{fallback}

Worst SKUs:
{table}
"""
    try:
        response = _gemma_chat([{"role": "user", "content": prompt}])
        narrative = (response.get("message") or {}).get("content") or ""
        narrative = narrative.strip()
        if not narrative:
            raise RuntimeError("Gemma 2 returned an empty narrative")
        return {"source": "gemma2:2b", "narrative": narrative, "fallback": fallback, "error": ""}
    except Exception as exc:
        return {
            "source": "rule",
            "narrative": fallback,
            "fallback": fallback,
            "error": str(exc),
        }


# ====================== GROUNDED COPILOT Q&A ======================
# Free-form questions go to local Gemma with a small slice of the forecast.
# Every number in the reply has to come from that slice. Otherwise the
# original keyword answer is what the user sees.
COPILOT_ROW_LIMIT = 15
# Compact on purpose: Gemma 2 has to read this and answer inside 8 seconds.
COPILOT_TABLE_COLUMNS = [
    "sku",
    "abc_class",
    "stockout_risk",
    "weeks_of_cover",
    "lead_time_weeks",
    "current_inventory",
    "reorder_point",
    "days_to_stockout",
    "vendor_fill_rate",
    "vendor_name",
]
COPILOT_ROUND = {
    "weeks_of_cover": 1,
    "current_inventory": 1,
    "forecast_weekly_demand": 1,
    "reorder_point": 1,
    "safety_stock": 1,
    "days_to_stockout": 1,
    "lead_time_days": 1,
    "lead_time_weeks": 2,
    "vendor_fill_rate": 2,
    "avg_weekly_demand": 2,
    "weekly_demand_sd": 2,
    "unit_cost": 2,
    "annual_demand": 2,
    "usage_value": 2,
}
# Gaps a planner might state without a new fact: on-hand versus reorder point,
# cover versus lead time, and the safety-stock piece of the reorder point.
COPILOT_DIFF_PAIRS = (
    ("current_inventory", "reorder_point"),
    ("current_inventory", "safety_stock"),
    ("reorder_point", "safety_stock"),
    ("weeks_of_cover", "lead_time_weeks"),
    ("days_to_stockout", "lead_time_days"),
    ("forecast_weekly_demand", "avg_weekly_demand"),
)
_NUMBER_RE = re.compile(
    r"(?<![\w.])(-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+\.\d+|-?\d+)%?(?!\w)"
)
_CLASS_PATTERNS = (
    re.compile(r"\b([ABCabc])\s*-\s*class\b", re.IGNORECASE),
    re.compile(r"\b([ABC])\s+class\b"),
    re.compile(r"\bclass\s*-?\s*([ABCabc])\b", re.IGNORECASE),
    re.compile(r"\babc\s*-?\s*([ABCabc])\b", re.IGNORECASE),
)


class CopilotReply:
    """One copilot turn: the text to show, where it came from, and the rows behind it.

    `source` is "grounded" when Gemma 2 answered from those rows and every
    number checked out, or "rule-based" when the keyword answer was used.
    """

    def __init__(self, answer: str, source: str, rows: pd.DataFrame, unverified=None, error: str = ""):
        self.answer = answer
        self.source = source
        self.rows = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame()
        self.unverified = list(unverified or [])
        self.error = error

    def __str__(self) -> str:
        return self.answer

    def __repr__(self) -> str:
        return f"CopilotReply(source={self.source!r}, unverified={self.unverified!r})"


def _extract_numbers(text: str):
    """Numbers in reading order. Integers stay ints; 1,220 and 86% count."""
    found = []
    if not text:
        return found
    for match in _NUMBER_RE.finditer(str(text)):
        token = match.group(1).replace(",", "")
        try:
            found.append(float(token) if "." in token else int(token))
        except ValueError:
            continue
    return found


def _nearly(left, right) -> bool:
    return abs(float(left) - float(right)) <= 1e-6 * max(1.0, abs(float(right)))


def _supports(number, base) -> bool:
    """True when `number` is `base`, a 0/1/2-decimal rounding of it, or a percent form."""
    if not math.isfinite(base) or not math.isfinite(float(number)):
        return False
    candidates = [float(base)]
    for digits in (0, 1, 2):
        candidates.append(float(round(float(base), digits)))
    # Fill rates are stored as 0.86. An answer that says 86% is the same fact.
    if abs(float(base)) <= 1.5:
        scaled = float(base) * 100.0
        candidates.append(scaled)
        for digits in (0, 1, 2):
            candidates.append(float(round(scaled, digits)))
    return any(_nearly(number, candidate) for candidate in candidates)


def _finite_values(series: pd.Series):
    values = []
    for value in pd.to_numeric(series, errors="coerce").tolist():
        if pd.isna(value):
            continue
        number = float(value)
        if math.isfinite(number):
            values.append(number)
    return values


def _base_numbers(rows: pd.DataFrame):
    """Cell values plus counts, column sums, and a few within-row gaps."""
    bases = []
    if rows is None or len(rows) == 0:
        return bases
    numeric = {column: _finite_values(rows[column]) for column in rows.columns}
    for values in numeric.values():
        bases.extend(values)
        if values:
            bases.append(float(sum(values)))
    for column in rows.columns:
        for raw in rows[column].tolist():
            if isinstance(raw, str):
                bases.extend(float(number) for number in _extract_numbers(raw))
    bases.append(float(len(rows)))
    for column in ("abc_class", "stockout_risk"):
        if column in rows.columns:
            bases.extend(float(count) for count in rows[column].astype(str).value_counts().tolist())
    if {"current_inventory", "reorder_point"}.issubset(rows.columns):
        inventory = pd.to_numeric(rows["current_inventory"], errors="coerce")
        reorder = pd.to_numeric(rows["reorder_point"], errors="coerce")
        comparable = inventory.notna() & reorder.notna()
        short = comparable & (inventory <= reorder)
        bases.append(float(short.sum()))
        bases.append(float((comparable & ~short).sum()))
    for left_name, right_name in COPILOT_DIFF_PAIRS:
        if left_name not in rows.columns or right_name not in rows.columns:
            continue
        left = pd.to_numeric(rows[left_name], errors="coerce")
        right = pd.to_numeric(rows[right_name], errors="coerce")
        for a, b in zip(left.tolist(), right.tolist()):
            if pd.isna(a) or pd.isna(b):
                continue
            if math.isfinite(float(a)) and math.isfinite(float(b)):
                bases.append(abs(float(a) - float(b)))
    return bases


def check_answer_grounding(answer: str, rows: pd.DataFrame):
    """Return 'grounded', or the numbers that are not in the supplied rows.

    A number counts when it appears in a supplied cell or is a trivial
    derivation: rounding to 0, 1, or 2 decimals, a fill rate written as a
    percent, the row count, a column sum, or a within-row gap such as
    on-hand minus reorder point.
    """
    numbers = _extract_numbers(answer)
    if not numbers:
        return "grounded"
    bases = _base_numbers(rows if isinstance(rows, pd.DataFrame) else pd.DataFrame())
    unverified = []
    for number in numbers:
        if any(_supports(number, base) for base in bases):
            continue
        if any(_nearly(number, earlier) for earlier in unverified):
            continue
        unverified.append(number)
    if not unverified:
        return "grounded"
    return unverified


def _mentioned_skus(frame: pd.DataFrame, question: str):
    text = (question or "").upper()
    found = set()
    skus = sorted({str(sku).upper() for sku in frame["sku"].dropna()}, key=len, reverse=True)
    for sku in skus:
        if re.search(rf"(?<![A-Z0-9]){re.escape(sku)}(?![A-Z0-9])", text):
            found.add(sku)
            continue
        tail = sku.split("-")[-1]
        prefix = "-".join(sku.split("-")[:-1])
        if prefix and tail.isdigit():
            loose = rf"(?<![A-Z0-9]){re.escape(prefix)}[-\s]*0*{int(tail)}(?![A-Z0-9])"
            if re.search(loose, text):
                found.add(sku)
    return found


def _mentioned_classes(question: str):
    found = set()
    for pattern in _CLASS_PATTERNS:
        for match in pattern.finditer(question or ""):
            letter = next(group for group in match.groups() if group)
            found.add(letter.upper())
    return found


def _risk_mask(frame: pd.DataFrame, question: str):
    """Rows the question is asking about, or None when it never mentions risk."""
    q = (question or "").lower()
    if frame.empty:
        return None
    parts = []
    if re.search(r"below\s+rop|\breorder\b|below\s+reorder|\bcritical\b", q):
        if {"current_inventory", "reorder_point"}.issubset(frame.columns):
            inventory = pd.to_numeric(frame["current_inventory"], errors="coerce")
            reorder = pd.to_numeric(frame["reorder_point"], errors="coerce")
            parts.append(inventory <= reorder)
        elif "stockout_risk" in frame.columns:
            parts.append(frame["stockout_risk"].astype(str).str.contains(r"High|CRITICAL", regex=True))
    if re.search(r"\bhigh\b|run\s*out|runs\s+out|running\s+out|stock-?out|\brisk\b", q):
        if "stockout_risk" in frame.columns:
            parts.append(frame["stockout_risk"].astype(str).str.contains(r"High|CRITICAL", regex=True))
    if re.search(r"\bmedium\b", q) and "stockout_risk" in frame.columns:
        parts.append(frame["stockout_risk"].astype(str).str.contains("Medium", regex=False))
    if re.search(r"\blow\s+risk\b", q) and "stockout_risk" in frame.columns:
        parts.append(frame["stockout_risk"].astype(str).str.contains("Low", regex=False))
    if not parts:
        return None
    mask = parts[0].fillna(False)
    for part in parts[1:]:
        mask = mask | part.fillna(False)
    return mask.reindex(frame.index, fill_value=False)


def _compact_rows(selected: pd.DataFrame) -> pd.DataFrame:
    if selected is None or len(selected) == 0:
        columns = [
            column
            for column in COPILOT_TABLE_COLUMNS
            if selected is not None and column in selected.columns
        ]
        return pd.DataFrame(columns=columns)
    columns = [column for column in COPILOT_TABLE_COLUMNS if column in selected.columns]
    out = selected.loc[:, columns].copy().reset_index(drop=True)
    for column, digits in COPILOT_ROUND.items():
        if column not in out.columns:
            continue
        numbers = pd.to_numeric(out[column], errors="coerce").round(digits)
        cleaned = []
        for value in numbers.tolist():
            if pd.isna(value):
                cleaned.append(math.nan)
                continue
            number = float(value)
            if math.isfinite(number):
                cleaned.append(number)
            elif column == "weeks_of_cover" and math.isinf(number):
                # Keep the sentinel so the prompt can say "No demand" instead of a fake quantity.
                cleaned.append(number)
            else:
                cleaned.append(math.nan)
        out[column] = cleaned
    return out


def select_copilot_rows(frame: pd.DataFrame, question: str, limit: int = COPILOT_ROW_LIMIT) -> pd.DataFrame:
    """Rows that can answer `question`.

    SKU codes, ABC class, and risk words in the question are applied together.
    A risk filter that matches nothing is dropped when a SKU or class was
    named, so the model can say those items do not run out. With no signal,
    the 15 highest-risk rows are used.
    """
    if not isinstance(frame, pd.DataFrame) or frame.empty or "sku" not in frame.columns:
        return _compact_rows(frame if isinstance(frame, pd.DataFrame) else pd.DataFrame())

    skus = _mentioned_skus(frame, question)
    classes = _mentioned_classes(question)
    risk_mask = _risk_mask(frame, question)
    selected = frame
    recognized = False

    if skus:
        recognized = True
        selected = selected.loc[selected["sku"].astype(str).str.upper().isin(skus)]

    if classes and "abc_class" in selected.columns:
        recognized = True
        class_rows = selected.loc[selected["abc_class"].astype(str).str.upper().isin(classes)]
        # A named SKU that is not in the requested class stays, so the answer
        # can correct the class instead of switching to an unrelated top-15.
        if not class_rows.empty or not skus:
            selected = class_rows

    if risk_mask is not None:
        recognized = True
        mask = risk_mask.reindex(selected.index, fill_value=False)
        risk_rows = selected.loc[mask.to_numpy()]
        if not risk_rows.empty or not (skus or classes):
            selected = risk_rows

    if not recognized:
        selected = sort_risk_table(frame)
    else:
        selected = sort_risk_table(selected)
    return _compact_rows(selected.head(limit))


def _format_cell(column: str, value) -> str:
    if isinstance(value, str):
        return " ".join(value.split())
    try:
        if value is None or pd.isna(value):
            return ""
    except (TypeError, ValueError):
        return " ".join(str(value).split())
    try:
        number = float(value)
    except (TypeError, ValueError):
        return " ".join(str(value).split())
    if not math.isfinite(number):
        return "No demand" if math.isinf(number) else ""
    digits = COPILOT_ROUND.get(column, 4)
    rounded = round(number, digits)
    if float(rounded).is_integer():
        return str(int(rounded))
    return f"{rounded:.{digits}f}".rstrip("0").rstrip(".")


def copilot_table_text(rows: pd.DataFrame) -> str:
    """Compact CSV of the rows Gemma is allowed to use."""
    if rows is None or len(rows) == 0:
        return "(no rows)"
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(list(rows.columns))
    for _, row in rows.iterrows():
        writer.writerow([_format_cell(column, row[column]) for column in rows.columns])
    return buffer.getvalue().strip()


_SKU_TOKEN_RE = re.compile(r"(?<![A-Z0-9])SKU-\d+(?![A-Z0-9])", re.IGNORECASE)
# "stockout_risk" is a column name, not a claim that something stocked out.
_RUNOUT_RE = re.compile(
    r"\brun\s*out\b|\bruns\s+out\b|\brunning\s+out\b|\bstock-?out\b|\bhigh risk\b",
    re.IGNORECASE,
)
_NEGATION_RE = re.compile(r"\b(none|no|not|n't|without)\b", re.IGNORECASE)
# Long enough for about three sentences, short enough to return before the 8s timeout.
COPILOT_MAX_TOKENS = 120


def _copilot_prompt(question: str, rows: pd.DataFrame) -> str:
    table = copilot_table_text(rows)
    # Kept short on purpose. Gemma 2 has to finish inside the 8-second helper.
    return f"""You are an inventory planner. Answer the question using ONLY the table below.
Do not invent SKUs or numbers. Quote stockout_risk for every SKU you name.
Say an item runs out only when its stockout_risk is High. Low and Medium do not run out before the next delivery.
If no row matches, say so. Write at most 3 short sentences.

Question: {question}

Rows:
{table}
"""


def answer_conflicts_with_rows(answer: str, rows: pd.DataFrame) -> str:
    """Return a reason to reject the reply, or '' when the claims match the rows.

    Number checks cannot see a Low SKU described as running out, because the
    SKU's own digits are already in the table. This catches that, and SKU
    codes that were not in the rows at all.
    """
    if not answer:
        return ""
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame()
    known = set()
    risk_by_sku = {}
    if not frame.empty and "sku" in frame.columns:
        known = {str(sku).upper() for sku in frame["sku"]}
        if "stockout_risk" in frame.columns:
            risk_by_sku = {
                str(sku).upper(): str(risk)
                for sku, risk in zip(frame["sku"], frame["stockout_risk"])
            }
    unknown = []
    for match in _SKU_TOKEN_RE.finditer(answer):
        token = match.group(0).upper()
        if token not in known and token not in unknown:
            unknown.append(token)
    if unknown:
        return "unknown sku: " + ", ".join(unknown[:5])

    for sentence in re.split(r"[.!?\n]", answer):
        if not _RUNOUT_RE.search(sentence) or _NEGATION_RE.search(sentence):
            continue
        named = [match.group(0).upper() for match in _SKU_TOKEN_RE.finditer(sentence)]
        if not named:
            high = any("High" in risk or "CRITICAL" in risk for risk in risk_by_sku.values())
            if risk_by_sku and not high:
                return "run-out claim without a high-risk row"
            continue
        for token in named:
            risk = risk_by_sku.get(token, "")
            if "High" not in risk and "CRITICAL" not in risk:
                return f"{token} is {risk or 'not high risk'}"
    return ""


def present_copilot_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Readable headers for the Rows used table in the Copilot tab."""
    if rows is None or len(rows) == 0:
        return pd.DataFrame()
    shown = rows.copy()
    if "weeks_of_cover" in shown.columns:
        def _cover_label(value):
            label = format_weeks_of_cover(value)
            return "" if str(label).lower() == "nan" else label

        shown["weeks_of_cover"] = shown["weeks_of_cover"].map(_cover_label)
    rename = {key: RISK_BOARD_RENAME[key] for key in shown.columns if key in RISK_BOARD_RENAME}
    return shown.rename(columns=rename)


# ====================== COGNITIVE INTERACTIVE AGENT ======================
class SupplyChainAgent:
    def __init__(self, data: pd.DataFrame):
        self.df = data

    def query(self, question: str) -> CopilotReply:
        """Answer a free-form question from the forecast, or fall back to keywords.

        Gemma 2 sees only the rows selected for the question. The reply is
        kept when every number in it is grounded in those rows. If Ollama is
        down, the call fails, or any number does not check out, the original
        keyword answer is returned with source "rule-based".
        """
        question = "" if question is None else str(question)
        rows = select_copilot_rows(self.df, question)
        keyword = self._keyword_answer(question)
        try:
            response = _gemma_chat(
                [{"role": "user", "content": _copilot_prompt(question, rows)}],
                options={"num_predict": COPILOT_MAX_TOKENS, "temperature": 0},
            )
            message = (response or {}).get("message") or {}
            content = str(message.get("content") or "").strip()
            if not content:
                raise RuntimeError("Gemma 2 returned an empty answer")
            verdict = check_answer_grounding(content, rows)
            conflict = answer_conflicts_with_rows(content, rows)
            if verdict != "grounded" or conflict:
                unverified = verdict if isinstance(verdict, list) else []
                return CopilotReply(
                    keyword,
                    "rule-based",
                    rows,
                    unverified=unverified,
                    error=conflict or "ungrounded",
                )
            return CopilotReply(content, "grounded", rows)
        except Exception as exc:
            return CopilotReply(keyword, "rule-based", rows, error=str(exc))

    def _keyword_answer(self, question: str) -> str:
        q = question.lower()

        if "critical" in q or "below rop" in q or "reorder" in q:
            triggered = self.df[self.df["current_inventory"] <= self.df["reorder_point"]]
            if triggered.empty:
                return "🟢 All inventory levels are above their statistical reorder points."

            res = "⚠️ **Action Required: SKUs Breaching Reorder Point:**\n"
            for _, row in triggered.iterrows():
                res += (
                    f"- **{row['sku']}** ({row['vendor_name']}): "
                    f"Current Stock: {row['current_inventory']} | "
                    f"ROP: {row['reorder_point']} | "
                    f"Days Left: {row['days_to_stockout']}\n"
                )
            return res

        if "risk" in q or "high risk" in q:
            high_risk = self.df[self.df["stockout_risk"].str.contains("High|CRITICAL")]
            if high_risk.empty:
                return "🟢 No high stockout risks identified."
            res = "📋 **High & Critical Risk SKUs:**\n"
            for _, row in high_risk.iterrows():
                res += (
                    f"- SKU **{row['sku']}**: Status: `{row['stockout_risk']}` | "
                    f"Vendor Fill Rate: {row['vendor_fill_rate'] * 100}%\n"
                )
            return res

        if "sku" in q:
            words = q.split()
            for word in words:
                clean_word = word.strip(",.?!").upper()
                match = self.df[self.df["sku"].str.upper() == clean_word]
                if not match.empty:
                    row = match.iloc[0]
                    cover_line = ""
                    if "weeks_of_cover" in row.index and "risk_reason" in row.index:
                        cover_line = (
                            f"  • Weeks of cover: {format_weeks_of_cover(row['weeks_of_cover'])}"
                            f" | Risk: {row['stockout_risk']}\n"
                            f"  • {row['risk_reason']}\n"
                        )
                    level = row["service_level"] if "service_level" in row.index else ""
                    level_bit = f" | Service level: {level}" if level else ""
                    return (
                        f"📊 **SKU Profile: {row['sku']}** ({row['vendor_name']})\n"
                        f"  • Class: {row['abc_class']} | Critical: {row['critical_item']}\n"
                        f"  • Inventory Level: {row['current_inventory']} units\n"
                        f"  • Safety Stock: {row['safety_stock']} | ROP: {row['reorder_point']} units"
                        f"{level_bit}\n"
                        f"  • Days to Stockout: {row['days_to_stockout']} "
                        f"(Lead Time: {row['lead_time_days']} days)\n"
                        f"{cover_line}"
                    ).rstrip()
            return "🔍 Couldn't match the specific SKU ID. Try entering the precise SKU code."

        return (
            "💡 *Copilot Query Examples:*\n"
            "  - 'Which items are below reorder point?'\n"
            "  - 'Show high risk items'\n"
            "  - 'Status for SKU [SKU_ID]'"
        )


# ====================== PIPELINE (CLI + UI) ======================
def run_pipeline(
    sample_file: str = "week_2026-04-13_demo.csv",
    forecast_weeks: int = 8,
    generate_report: bool = True,
    service_level: str = DEFAULT_SERVICE_LEVEL,
):
    """Run the full inventory intelligence pipeline. Returns None on failure."""
    df = load_and_clean_weekly_data(sample_file)
    if df is None:
        return None

    forecast_df = forecast_and_stockout(
        df,
        forecast_weeks=forecast_weeks,
        service_level=service_level,
    )
    summary_path = OUTPUT_DIR.parent / "latest_summary.csv"
    if not _write_csv(forecast_df, summary_path):
        summary_path = None

    report = generate_llm_report(forecast_df) if generate_report else ""

    return {
        "raw_df": df,
        "full_summary": forecast_df,
        "report": report,
        "summary_path": summary_path,
        "unit_cost_source": forecast_df.attrs.get("unit_cost_source", "sample"),
        "service_level": service_level,
        "source_file": sample_file,
    }


# ====================== MAIN EXECUTION CONTEXT ======================
if __name__ == "__main__":
    start_time = time.time()
    print(
        f"🚀 Supply Chain Intelligence Pipeline Active - "
        f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    )

    result = run_pipeline("week_2026-04-13_demo.csv")

    if result is not None:
        total_time = time.time() - start_time
        print("\n" + "=" * 60)
        print(f"🎉 ENGINE EXECUTION COMPLETE in {total_time:.1f} seconds")
        print("=" * 60)

        print("\n🤖 Inventory Copilot Online! (Type 'exit' to log out)")
        agent = SupplyChainAgent(result["full_summary"])
        while True:
            try:
                user_query = input("\n💬 Copilot Command: ")
                if user_query.lower() in ["exit", "quit"]:
                    print("Logging out safely. Have a great shift! 📦")
                    break
                if user_query.strip():
                    reply = agent.query(user_query)
                    print(reply)
                    print(f"[{reply.source}]")
            except (KeyboardInterrupt, EOFError):
                break
