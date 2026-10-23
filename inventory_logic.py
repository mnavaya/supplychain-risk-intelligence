"""Inventory math shared by the pipeline, Streamlit app, and tests.

Holt-Winters forecasts, service-level safety stock, weeks-of-cover alerts,
and usage-value ABC all live here so the UI and the CLI call one implementation.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Sidebar choices. Z values are the ones the product spec asks for.
SERVICE_LEVELS = {
    "90%": 1.28,
    "95%": 1.65,
    "99%": 2.33,
}
DEFAULT_SERVICE_LEVEL = "95%"

ACCURACY_HOLDOUT_WEEKS = 4
ACCURACY_MIN_TRAIN_WEEKS = 4

# Used only when a weekly file has no lead_time_days value.
DEFAULT_LEAD_TIME_DAYS = 7

# Illustrative planning prices (USD) used only when a file has no unit_cost.
# They are sample costs, not a vendor price list. A stable per-SKU factor
# (see sample_unit_cost) keeps two SKUs in one category from sharing a price.
SAMPLE_UNIT_COST_BY_CATEGORY = {
    "Reagent": 48.00,
    "Consumable": 6.50,
    "Equipment": 125.00,
    "Chemical": 22.00,
}
SAMPLE_UNIT_COST_DEFAULT = 15.00

RISK_SORT_ORDER = {
    "CRITICAL (Below ROP)": 0,
    "High": 1,
    "Medium": 2,
    "Low": 3,
}

RISK_BOARD_COLUMNS = [
    "sku",
    "abc_class",
    "stockout_risk",
    "weeks_of_cover",
    "risk_reason",
    "current_inventory",
    "forecast_weekly_demand",
    "service_level",
    "safety_stock",
    "reorder_point",
    "avg_weekly_demand",
    "weekly_demand_sd",
    "lead_time_days",
    "lead_time_weeks",
    "days_to_stockout",
    "unit_cost",
    "annual_demand",
    "usage_value",
    "usage_cum_pct",
    "vendor_fill_rate",
    "vendor_name",
    "critical_item",
]

RISK_BOARD_RENAME = {
    "sku": "SKU",
    "abc_class": "ABC",
    "stockout_risk": "Stockout risk",
    "weeks_of_cover": "Weeks of Cover",
    "risk_reason": "Risk Reason",
    "current_inventory": "On-hand",
    "forecast_weekly_demand": "Forecast weekly demand",
    "service_level": "Service level",
    "safety_stock": "Safety stock",
    "reorder_point": "Reorder point",
    "avg_weekly_demand": "Avg weekly demand",
    "weekly_demand_sd": "Weekly demand SD",
    "lead_time_days": "Lead time (days)",
    "lead_time_weeks": "Lead time (weeks)",
    "days_to_stockout": "Days to stockout",
    "unit_cost": "Unit cost",
    "annual_demand": "Annual demand",
    "usage_value": "Usage value",
    "usage_cum_pct": "Cumulative usage %",
    "vendor_fill_rate": "Vendor fill rate",
    "vendor_name": "Vendor",
    "critical_item": "Critical",
}


def z_for_service_level(service_level: str) -> float:
    key = str(service_level).strip()
    if key not in SERVICE_LEVELS:
        known = ", ".join(SERVICE_LEVELS)
        raise ValueError(f"Service level must be one of: {known}")
    return float(SERVICE_LEVELS[key])


def sample_unit_cost(sku, category=None) -> float:
    """Documented sample unit cost used when a CSV has no unit_cost column.

    Base price comes from SAMPLE_UNIT_COST_BY_CATEGORY. The SKU id then applies
    a stable multiplier from 0.85 to 1.25 so ABC is not only a sort on demand.
    """
    base = SAMPLE_UNIT_COST_BY_CATEGORY.get(str(category), SAMPLE_UNIT_COST_DEFAULT)
    spread = sum(ord(char) for char in str(sku)) % 41  # 0..40
    multiplier = 0.85 + (spread / 100.0)
    return round(float(base) * multiplier, 2)


def attach_unit_cost(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Return a copy with unit_cost filled in, plus where the prices came from.

    source is "file" (every row already had a cost), "sample" (none did),
    or "mixed" (gaps were filled with sample_unit_cost).
    """
    out = df.copy()
    if "category" in out.columns:
        categories = out["category"]
    else:
        categories = pd.Series([None] * len(out), index=out.index)
    sample = pd.Series(
        [
            sample_unit_cost(sku, category)
            for sku, category in zip(out["sku"], categories)
        ],
        index=out.index,
        dtype=float,
    )
    if "unit_cost" not in out.columns:
        out["unit_cost"] = sample
        return out, "sample"

    numeric = pd.to_numeric(out["unit_cost"], errors="coerce")
    missing = numeric.isna()
    if bool(missing.all()):
        out["unit_cost"] = sample
        return out, "sample"
    if bool(missing.any()):
        numeric = numeric.copy()
        numeric.loc[missing] = sample.loc[missing]
        out["unit_cost"] = numeric
        return out, "mixed"
    out["unit_cost"] = numeric
    return out, "file"


def weekly_series(df: pd.DataFrame, sku=None) -> pd.DataFrame:
    """One row per week for a SKU, sorted oldest to newest.

    Duplicate weeks are summed so a repeated stamp cannot be both a training
    week and a holdout week.
    """
    part = df if sku is None else df[df["sku"] == sku]
    if part.empty:
        return pd.DataFrame(columns=["week", "consumption"])
    work = part.copy()
    work["week"] = pd.to_datetime(work["week"], errors="coerce")
    work["consumption"] = pd.to_numeric(work["consumption"], errors="coerce")
    work = work.dropna(subset=["week", "consumption"])
    if work.empty:
        return pd.DataFrame(columns=["week", "consumption"])
    grouped = (
        work.groupby("week", as_index=False)["consumption"]
        .sum()
        .sort_values("week")
        .reset_index(drop=True)
    )
    return grouped


def annual_demand_from_weekly(consumption) -> float:
    """Most recent 52 weeks summed, or average week × 52 when history is shorter."""
    values = pd.to_numeric(pd.Series(consumption), errors="coerce").dropna()
    if values.empty:
        return 0.0
    if len(values) >= 52:
        annual = float(values.iloc[-52:].sum())
    else:
        annual = float(values.mean() * 52.0)
    return max(annual, 0.0)


def compute_safety_stock(weekly_demand_sd, lead_time_weeks, z) -> float:
    """Safety stock = Z × weekly demand SD × sqrt(lead time in weeks)."""
    sd = 0.0 if pd.isna(weekly_demand_sd) else max(float(weekly_demand_sd), 0.0)
    lead = 0.0 if pd.isna(lead_time_weeks) else max(float(lead_time_weeks), 0.0)
    return float(z) * sd * math.sqrt(lead)


def compute_reorder_point(avg_weekly_demand, lead_time_weeks, safety_stock) -> float:
    """Reorder point = average weekly demand × lead time in weeks + safety stock."""
    average = 0.0 if pd.isna(avg_weekly_demand) else max(float(avg_weekly_demand), 0.0)
    lead = 0.0 if pd.isna(lead_time_weeks) else max(float(lead_time_weeks), 0.0)
    stock = 0.0 if pd.isna(safety_stock) else float(safety_stock)
    return average * lead + stock


REORDER_NOTE = " Current stock is at or below the reorder point."


def annotate_reorder_status(summary: pd.DataFrame) -> pd.DataFrame:
    """Add or remove the reorder-point sentence so it matches the current ROP."""
    if summary.empty or "risk_reason" not in summary.columns:
        return summary
    out = summary.copy()
    out["risk_reason"] = (
        out["risk_reason"].astype(str).str.replace(REORDER_NOTE, "", regex=False)
    )
    if "current_inventory" not in out.columns or "reorder_point" not in out.columns:
        return out
    below = out["current_inventory"] <= out["reorder_point"]
    out.loc[below, "risk_reason"] = out.loc[below, "risk_reason"] + REORDER_NOTE
    return out


def apply_service_level(summary: pd.DataFrame, service_level: str) -> pd.DataFrame:
    """Recompute safety stock and reorder point from stored demand components."""
    out = summary.copy()
    required = {"weekly_demand_sd", "avg_weekly_demand", "lead_time_weeks"}
    if not required.issubset(out.columns):
        return out
    z = z_for_service_level(service_level)
    safety = [
        compute_safety_stock(sd, lead, z)
        for sd, lead in zip(out["weekly_demand_sd"], out["lead_time_weeks"])
    ]
    reorder = [
        compute_reorder_point(avg, lead, ss)
        for avg, lead, ss in zip(
            out["avg_weekly_demand"], out["lead_time_weeks"], safety
        )
    ]
    out["safety_stock"] = np.round(safety, 2)
    out["reorder_point"] = np.round(reorder, 2)
    out["service_level"] = service_level
    out["z_score"] = z
    return annotate_reorder_status(out)


def service_level_snapshot(summary: pd.DataFrame) -> pd.DataFrame:
    """Portfolio totals of safety stock and reorder point at 90%, 95%, and 99%."""
    rows = []
    for level in SERVICE_LEVELS:
        adjusted = apply_service_level(summary, level)
        if "safety_stock" not in adjusted.columns:
            safety_total = 0.0
            reorder_total = 0.0
        else:
            safety_total = float(pd.to_numeric(adjusted["safety_stock"], errors="coerce").fillna(0).sum())
            reorder_total = float(pd.to_numeric(adjusted["reorder_point"], errors="coerce").fillna(0).sum())
        rows.append(
            {
                "service_level": level,
                "z": SERVICE_LEVELS[level],
                "safety_stock": safety_total,
                "reorder_point": reorder_total,
            }
        )
    return pd.DataFrame(rows)


def service_level_message(snapshot: pd.DataFrame) -> str:
    ordered = snapshot.sort_values("z")
    safety = ordered["safety_stock"].tolist()
    reorder = ordered["reorder_point"].tolist()
    rises = all(
        safety[i] < safety[i + 1] and reorder[i] < reorder[i + 1]
        for i in range(len(safety) - 1)
    )
    if rises:
        return (
            "Higher service level produces higher safety stock and a higher reorder point."
        )
    flat = all(
        safety[i] <= safety[i + 1] and reorder[i] <= reorder[i + 1]
        for i in range(len(safety) - 1)
    )
    if flat:
        return (
            "Higher service level does not lower safety stock or the reorder point. "
            "Totals stay flat when weekly demand does not vary."
        )
    return "Safety stock should rise with the service level when weekly demand varies."


def format_service_snapshot(snapshot: pd.DataFrame) -> pd.DataFrame:
    display = snapshot.copy()
    display["safety_stock"] = display["safety_stock"].map(lambda v: round(float(v), 2))
    display["reorder_point"] = display["reorder_point"].map(lambda v: round(float(v), 2))
    display["z"] = display["z"].map(lambda v: f"{float(v):.2f}")
    return display.rename(
        columns={
            "service_level": "Service level",
            "z": "Z",
            "safety_stock": "Safety stock",
            "reorder_point": "Reorder point",
        }
    )


def weeks_of_cover(current_stock, forecast_weekly_demand) -> float:
    """On-hand ÷ forecast weekly demand. Zero demand does not divide."""
    demand = 0.0 if pd.isna(forecast_weekly_demand) else float(forecast_weekly_demand)
    stock = 0.0 if pd.isna(current_stock) else float(current_stock)
    if demand <= 0:
        return math.inf
    return stock / demand


def classify_cover_risk(current_stock, forecast_weekly_demand, lead_time_weeks) -> dict:
    """Compare weeks of cover with supplier lead time.

    High: cover runs out before replenishment arrives (cover < lead time).
    Medium: cover is within one week of the lead time (lead time <= cover < lead time + 1).
    Low: cover comfortably exceeds the lead time by at least a week.
    """
    cover = weeks_of_cover(current_stock, forecast_weekly_demand)
    lead = 0.0 if pd.isna(lead_time_weeks) else max(float(lead_time_weeks), 0.0)
    if not math.isfinite(cover):
        return {
            "weeks_of_cover": math.inf,
            "stockout_risk": "Low",
            "risk_reason": (
                "Forecast weekly demand is zero, so on-hand stock is not being drawn down."
            ),
        }
    if cover < lead:
        risk = "High"
        reason = (
            f"Runs out before replenishment arrives "
            f"({cover:.1f} weeks of cover vs {lead:.1f} weeks lead time)."
        )
    elif cover < lead + 1:
        risk = "Medium"
        reason = (
            f"Within one week of the supplier lead time "
            f"({cover:.1f} weeks of cover vs {lead:.1f} weeks lead time)."
        )
    else:
        risk = "Low"
        reason = (
            f"Comfortably covers the supplier lead time "
            f"({cover:.1f} weeks of cover vs {lead:.1f} weeks lead time)."
        )
    return {
        "weeks_of_cover": cover,
        "stockout_risk": risk,
        "risk_reason": reason,
    }


def sort_risk_table(summary: pd.DataFrame) -> pd.DataFrame:
    """High (and legacy critical) rows first, then the tightest weeks of cover."""
    if summary.empty or "stockout_risk" not in summary.columns:
        return summary.copy()
    out = summary.copy()
    out["_risk_rank"] = out["stockout_risk"].map(lambda value: RISK_SORT_ORDER.get(value, 9))
    if "weeks_of_cover" in out.columns:
        cover = pd.to_numeric(out["weeks_of_cover"], errors="coerce")
        cover = cover.mask(~np.isfinite(cover), np.inf)
        out["_cover_sort"] = cover.fillna(np.inf)
    else:
        out["_cover_sort"] = 0
    sku_col = out["sku"] if "sku" in out.columns else ""
    out["_sku_sort"] = sku_col.astype(str)
    out = out.sort_values(
        ["_risk_rank", "_cover_sort", "_sku_sort"],
        ascending=[True, True, True],
    )
    return out.drop(columns=["_risk_rank", "_cover_sort", "_sku_sort"]).reset_index(drop=True)


def format_weeks_of_cover(value):
    """Display text. Always a string so a 'No demand' row stays Arrow-safe."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "" if value is None else str(value)
    if math.isinf(number):
        return "No demand"
    return f"{number:.1f}"


def format_risk_board(summary: pd.DataFrame) -> pd.DataFrame:
    board = sort_risk_table(summary)
    if "weeks_of_cover" in board.columns:
        board["weeks_of_cover"] = board["weeks_of_cover"].map(format_weeks_of_cover)
    ordered = [column for column in RISK_BOARD_COLUMNS if column in board.columns]
    hidden = {"z_score"}
    leftovers = [
        column
        for column in board.columns
        if column not in ordered and column not in hidden and not str(column).startswith("_")
    ]
    board = board[ordered + leftovers]
    return board.rename(columns=RISK_BOARD_RENAME)


def classify_abc(usage: pd.DataFrame) -> pd.DataFrame:
    """Pareto ABC on usage value.

    Sorted descending. A is the first 80% of usage value, B the next 15%,
    C the remaining 5%. The SKU that crosses a cutoff stays in the earlier class.
    """
    out = usage[["sku", "usage_value"]].copy()
    out["usage_value"] = pd.to_numeric(out["usage_value"], errors="coerce").fillna(0.0)
    out["usage_value"] = out["usage_value"].clip(lower=0.0)
    out = out.sort_values(["usage_value", "sku"], ascending=[False, True]).reset_index(drop=True)
    total = float(out["usage_value"].sum())
    if total <= 0 or out.empty:
        out["usage_cum_pct"] = 0.0
        out["abc_class"] = "C"
        return out

    fraction = out["usage_value"].cumsum() / total
    previous = fraction.shift(1, fill_value=0.0)
    classes = []
    for prior in previous:
        if prior < 0.80:
            classes.append("A")
        elif prior < 0.95:
            classes.append("B")
        else:
            classes.append("C")
    out["abc_class"] = classes
    out["usage_cum_pct"] = (fraction * 100.0).round(2)
    return out


def abc_summary(classified: pd.DataFrame) -> pd.DataFrame:
    """SKU count, share of SKUs, usage value, and share of usage value by class."""
    frame = classified.copy()
    if "abc_class" not in frame.columns:
        frame = classify_abc(frame)
    frame["usage_value"] = pd.to_numeric(frame["usage_value"], errors="coerce").fillna(0.0).clip(lower=0)
    total_skus = int(len(frame))
    total_value = float(frame["usage_value"].sum())
    rows = []
    for label in ("A", "B", "C"):
        part = frame[frame["abc_class"] == label]
        value = float(part["usage_value"].sum())
        rows.append(
            {
                "abc_class": label,
                "sku_count": int(len(part)),
                "pct_of_skus": (len(part) / total_skus * 100.0) if total_skus else 0.0,
                "total_usage_value": value,
                "pct_of_usage_value": (value / total_value * 100.0) if total_value else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _round_percentages(values, places=2) -> list[float]:
    raw = np.asarray(values, dtype=float)
    rounded = np.round(raw, places)
    if raw.size and abs(float(raw.sum()) - 100.0) < 0.05:
        diff = round(100.0 - float(rounded.sum()), places)
        if diff != 0:
            idx = int(np.argmax(raw))
            rounded[idx] = round(float(rounded[idx]) + diff, places)
    return [float(value) for value in rounded]


def format_abc_summary(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty or "usage_value" not in summary.columns:
        return pd.DataFrame(
            columns=[
                "Class",
                "SKU count",
                "% of SKUs",
                "Total usage value",
                "% of usage value",
            ]
        )
    table = abc_summary(classify_abc(summary[["sku", "usage_value"]]))
    table["pct_of_skus"] = _round_percentages(table["pct_of_skus"])
    table["pct_of_usage_value"] = _round_percentages(table["pct_of_usage_value"])
    table["total_usage_value"] = table["total_usage_value"].map(lambda v: round(float(v), 2))
    return table.rename(
        columns={
            "abc_class": "Class",
            "sku_count": "SKU count",
            "pct_of_skus": "% of SKUs",
            "total_usage_value": "Total usage value",
            "pct_of_usage_value": "% of usage value",
        }
    )


def holt_winters_forecast(consumption, horizon: int) -> np.ndarray:
    """Forecast `horizon` weeks with the project's Holt-Winters model.

    This is statsmodels ExponentialSmoothing with an additive trend and no
    seasonal component — the same forecaster the weekly pipeline already uses.
    Callers must pass training weeks only.
    """
    series = np.asarray(
        pd.to_numeric(pd.Series(consumption), errors="coerce").dropna(),
        dtype=float,
    )
    horizon = int(horizon)
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    if len(series) == 0:
        return np.zeros(horizon, dtype=float)
    if len(series) < 2:
        return np.repeat(max(float(series[-1]), 0.0), horizon)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ExponentialSmoothing(series, trend="add", seasonal=None)
            fit = model.fit()
            forecast = np.asarray(fit.forecast(horizon), dtype=float)
    except Exception:
        window = series[-3:] if len(series) >= 3 else series
        forecast = np.repeat(float(np.mean(window)), horizon)

    if len(forecast) != horizon:
        forecast = np.resize(forecast, horizon)
    return np.maximum(forecast, 0.0)


def forecast_mean_demand(consumption, horizon: int) -> float:
    """Mean of the Holt-Winters path. Short histories use the latest week."""
    values = pd.to_numeric(pd.Series(consumption), errors="coerce").dropna()
    if values.empty:
        return 0.0
    if len(values) < 3:
        return float(max(values.iloc[-1], 0.0))
    return float(np.mean(holt_winters_forecast(values, horizon)))


def split_train_test(series: pd.DataFrame, holdout_weeks: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out the latest weeks. Raises if any test week is also in training."""
    if holdout_weeks < 1:
        raise ValueError("holdout_weeks must be at least 1")
    ordered = series.sort_values("week").reset_index(drop=True)
    if len(ordered) <= holdout_weeks:
        raise ValueError("Not enough weeks to hold out")
    train = ordered.iloc[:-holdout_weeks].copy()
    test = ordered.iloc[-holdout_weeks:].copy()
    if train.empty or len(test) != holdout_weeks:
        raise ValueError("Holdout split failed")
    if train["week"].max() >= test["week"].min():
        raise ValueError("Test weeks leaked into training")
    overlap = set(train["week"]).intersection(set(test["week"]))
    if overlap:
        raise ValueError(f"Test weeks leaked into training: {sorted(overlap)}")
    return train, test


def eligible_forecast_skus(
    df: pd.DataFrame,
    min_weeks: int = ACCURACY_HOLDOUT_WEEKS + ACCURACY_MIN_TRAIN_WEEKS,
) -> list:
    """SKUs with enough positive weekly history, most-sold first."""
    if df is None or len(df) == 0 or "sku" not in getattr(df, "columns", []):
        return []
    work = df.copy()
    work["week"] = pd.to_datetime(work["week"], errors="coerce")
    work["consumption"] = pd.to_numeric(work["consumption"], errors="coerce").fillna(0)
    work = work.dropna(subset=["sku", "week"])
    ranked = []
    for sku, part in work.groupby("sku", sort=False):
        weeks = int(part["week"].nunique())
        positive = int(part.loc[part["consumption"] > 0, "week"].nunique())
        sold = float(part["consumption"].sum())
        if weeks >= min_weeks and positive >= min_weeks and sold > 0:
            ranked.append((str(sku), sold))
    ranked.sort(key=lambda item: (-item[1], item[0]))
    return [sku for sku, _sold in ranked]


def _winner(sku: str, mae_hw: float, mae_naive: float) -> tuple[str, str]:
    if mae_hw < mae_naive:
        return (
            "Holt-Winters",
            (
                f"Holt-Winters won for {sku}. "
                f"MAE {mae_hw:.2f} vs naive baseline MAE {mae_naive:.2f}."
            ),
        )
    if mae_naive < mae_hw:
        return (
            "Naive baseline",
            (
                f"Naive baseline won for {sku}. "
                f"MAE {mae_naive:.2f} vs Holt-Winters MAE {mae_hw:.2f}."
            ),
        )
    return (
        "Tie",
        (
            f"Tie for {sku}. Holt-Winters and the naive baseline "
            f"both have MAE {mae_hw:.2f}."
        ),
    )


def forecast_accuracy_check(
    df: pd.DataFrame,
    sku: str,
    holdout_weeks: int = ACCURACY_HOLDOUT_WEEKS,
    min_train_weeks: int = ACCURACY_MIN_TRAIN_WEEKS,
) -> dict:
    """Hold out the latest weeks, fit Holt-Winters on earlier weeks, score MAE.

    The naive baseline repeats the previous week's demand (the last training
    week) across the holdout. Test weeks are never passed to the model.
    """
    series = weekly_series(df, sku)
    needed = int(holdout_weeks) + int(min_train_weeks)
    if len(series) < needed:
        return {
            "sku": sku,
            "status": "insufficient_history",
            "message": (
                f"{sku} needs at least {needed} weekly observations "
                f"for a {holdout_weeks}-week holdout."
            ),
            "winner": None,
        }

    train, test = split_train_test(series, holdout_weeks)
    hw = holt_winters_forecast(train["consumption"], holdout_weeks)
    naive_value = float(train["consumption"].iloc[-1])
    naive = np.repeat(naive_value, holdout_weeks)
    actual = test["consumption"].to_numpy(dtype=float)
    mae_hw = float(np.mean(np.abs(actual - hw)))
    mae_naive = float(np.mean(np.abs(actual - naive)))
    winner, message = _winner(str(sku), mae_hw, mae_naive)
    detail = pd.DataFrame(
        {
            "week": test["week"].dt.strftime("%Y-%m-%d"),
            "actual": actual,
            "holt_winters": hw,
            "naive_baseline": naive,
        }
    )
    return {
        "sku": sku,
        "status": "ok",
        "winner": winner,
        "message": message,
        "mae_holt_winters": mae_hw,
        "mae_naive": mae_naive,
        "naive_value": naive_value,
        "train_end": train["week"].max().isoformat(),
        "test_start": test["week"].min().isoformat(),
        "train_weeks": int(len(train)),
        "test_weeks": int(len(test)),
        "detail": detail,
    }


def accuracy_display_table(result: dict) -> pd.DataFrame:
    """Actuals, both forecasts, and a closing MAE row.

    Numeric columns stay numeric. The MAE row leaves Actual empty (NaN)
    so Streamlit can serialize the table without mixing text into a number column.
    """
    detail = result["detail"].copy()
    detail["actual"] = detail["actual"].map(lambda v: round(float(v), 2))
    detail["holt_winters"] = detail["holt_winters"].map(lambda v: round(float(v), 2))
    detail["naive_baseline"] = detail["naive_baseline"].map(lambda v: round(float(v), 2))
    mae_row = pd.DataFrame(
        [
            {
                "week": "MAE",
                "actual": np.nan,
                "holt_winters": round(float(result["mae_holt_winters"]), 2),
                "naive_baseline": round(float(result["mae_naive"]), 2),
            }
        ]
    )
    table = pd.concat([detail, mae_row], ignore_index=True)
    table = table.rename(
        columns={
            "week": "Week",
            "actual": "Actual",
            "holt_winters": "Holt-Winters",
            "naive_baseline": "Naive baseline",
        }
    )
    for column in ("Actual", "Holt-Winters", "Naive baseline"):
        table[column] = pd.to_numeric(table[column], errors="coerce")
    return table
