"""Streamlit app for KPI anomaly review using robust statistics."""

from __future__ import annotations

import io
from typing import Optional, Any

import altair as alt
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="KPI Review",
    page_icon="📈",
    layout="wide",
)

REQUIRED_COLUMNS = ["Period", "Metric", "Value", "Units"]
SEVERITY_ORDER = {
    "Severe": 0,
    "Mild": 1,
    "Normal": 2,
    "Stable": 3,
    "Insufficient History": 4,
    "No Baseline": 5,
}

SEVERITY_BADGE = {
    "Severe": "🔴 Severe",
    "Mild": "🟡 Mild",
    "Normal": "🟢 Normal",
    "Stable": "🔵 Stable",
    "Insufficient History": "⚪ Insufficient History",
    "No Baseline": "⚪ No Baseline",
}


@st.cache_data(show_spinner=False)
def load_dataset(upload: io.BytesIO) -> pd.DataFrame:
    """Load a CSV upload, validate schema, and return cleaned dataframe."""

    df = pd.read_csv(upload)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            "CSV is missing required columns: " + ", ".join(missing)
        )

    cleaned = df.copy()
    cleaned = cleaned[REQUIRED_COLUMNS]
    cleaned["Metric"] = cleaned["Metric"].astype(str).str.strip()
    cleaned["Period"] = cleaned["Period"].astype(str).str.strip()
    cleaned["Units"] = cleaned["Units"].astype(str).str.strip()
    cleaned["Value"] = pd.to_numeric(cleaned["Value"], errors="coerce")
    cleaned = cleaned.dropna(subset=["Value"])
    return cleaned.reset_index(drop=True)


def median_absolute_deviation(values: pd.Series) -> float:
    """Compute the Median Absolute Deviation (MAD)."""

    median = np.median(values)
    return float(np.median(np.abs(values - median)))


def build_historical_baseline(history: pd.DataFrame) -> pd.DataFrame:
    """Aggregate historical records by metric to derive robust baselines."""

    if history.empty:
        return pd.DataFrame(columns=[
            "Metric",
            "baseline_median",
            "baseline_mad",
            "history_count",
        ])

    stats = (
        history.groupby("Metric")
        .agg(
            baseline_median=("Value", "median"),
            baseline_mad=("Value", median_absolute_deviation),
            history_count=("Value", "count"),
            historical_periods=("Period", lambda x: sorted(set(x))),
            historical_units=("Units", lambda x: ", ".join(sorted(set(x)))),
        )
        .reset_index()
    )
    return stats


def robust_z_score(value: float, median: float, mad: float) -> Optional[float]:
    """Return a robust z-score using MAD. Returns None when not computable."""

    if mad is None or np.isnan(mad):
        return None
    if mad == 0:
        return None
    scaled_mad = mad * 1.4826  # consistent with standard deviation if normal
    if scaled_mad == 0:
        return None
    return float((value - median) / scaled_mad)


def sensitivity_thresholds(sensitivity: float) -> tuple[float, float]:
    factor = sensitivity if sensitivity and not np.isnan(sensitivity) else 1.0
    factor = max(factor, 0.1)
    mild_threshold = 2.5 / factor
    severe_threshold = 4.0 / factor
    # Ensure ordering even with extreme sensitivity values
    if mild_threshold > severe_threshold:
        mild_threshold, severe_threshold = severe_threshold, mild_threshold
    return mild_threshold, severe_threshold


def categorize_severity(
    robust_score: Optional[float],
    deviation: float,
    baseline_mad: float,
    history_count: int,
    sensitivity: float,
) -> str:
    """Map robust z-score and deviation to severity buckets."""

    if history_count < 3 or baseline_mad is None or np.isnan(baseline_mad):
        return "Insufficient History"

    if robust_score is None:
        return "Stable" if deviation == 0 else "Severe"

    distance = abs(robust_score)
    mild_threshold, severe_threshold = sensitivity_thresholds(sensitivity)

    if distance >= severe_threshold:
        return "Severe"
    if distance >= mild_threshold:
        return "Mild"
    return "Normal"


def describe_implication(
    severity: str,
    deviation: float,
    relative_change: Optional[float],
    robust_score: Optional[float],
    *,
    current_value: float,
    baseline_median: Optional[float],
    baseline_mad: Optional[float],
    units: str,
    history_count: int,
    period: str,
    mild_threshold: float,
    severe_threshold: float,
) -> str:
    """Create a plain-language explanation for the anomaly result."""

    severity_opening = {
        "Severe": "This result stands out sharply.",
        "Mild": "This period shows a noticeable shift.",
        "Normal": "This period looks routine compared to what we usually see.",
        "Stable": "The metric is essentially steady against history.",
        "Insufficient History": "We don't have enough history to judge how unusual this is yet.",
        "No Baseline": "We cannot rate this metric because there is no historical baseline.",
    }.get(severity, "Latest status update.")

    units_label = f" {units}" if units else ""
    sentences: list[str] = [severity_opening]

    if current_value is not None and not pd.isna(current_value):
        sentences.append(
            f"In {period}, we logged {format_value(current_value)}{units_label}."
        )

    has_baseline = baseline_median is not None and not np.isnan(baseline_median)
    has_deviation = deviation is not None and not np.isnan(deviation)

    if has_baseline:
        if has_deviation and not np.isclose(deviation, 0.0, atol=1e-9):
            direction = "higher" if deviation > 0 else "lower"
            pieces: list[str] = []
            pieces.append(
                f"about {format_value(baseline_median)}{units_label} is typical"
            )
            change_bits: list[str] = []
            if deviation is not None and not np.isnan(deviation):
                change_bits.append(f"{abs(deviation):,.4g}{units_label}".strip())
            if relative_change is not None and not np.isnan(relative_change):
                change_bits.append(f"{relative_change:+.1f}%")
            change_text = " / ".join(change_bits) if change_bits else "a small margin"
            sentences.append(
                f"That's {direction} than the usual level (around {format_value(baseline_median)}{units_label}) by {change_text}."
            )
        else:
            sentences.append(
                f"Performance is right around the usual level of {format_value(baseline_median)}{units_label}."
            )
    else:
        sentences.append("We need more history before we can compare against a typical level.")

    if (
        baseline_mad is not None
        and not np.isnan(baseline_mad)
        and baseline_mad != 0
        and has_deviation
    ):
        multiples = abs(deviation) / baseline_mad
        sentences.append(
            f"The move is roughly {multiples:.1f}× the size of a normal period-to-period wiggle."
        )

    if robust_score is not None and not np.isnan(robust_score):
        abs_z = abs(robust_score)
        if abs_z >= severe_threshold:
            z_context = "well past our alert level."
        elif abs_z >= mild_threshold:
            z_context = "noticeably outside the comfort zone."
        else:
            z_context = "comfortably within the expected band."
        sentences.append(
            f"The anomaly score is {abs_z:.1f}, which is {z_context}"
        )

    if history_count:
        sentences.append(
            f"This comparison draws on {history_count} prior periods."
        )

    closing_note = {
        "Severe": "Recommend a quick review to confirm the drivers behind this swing.",
        "Mild": "Keep an eye on the next update to see if the change sticks.",
        "Normal": "No immediate action needed unless other signals disagree.",
        "Stable": "Everything looks steady—no follow-up needed right now.",
    }.get(severity, "")
    if closing_note:
        sentences.append(closing_note)

    return " ".join(sentences)


def evaluate_new_points(
    history: pd.DataFrame,
    new_points: pd.DataFrame,
    sensitivity: float,
) -> pd.DataFrame:
    """Compare each new metric observation against historical context."""

    baselines = build_historical_baseline(history)
    merged = new_points.merge(baselines, on="Metric", how="left")

    results = []
    mild_threshold, severe_threshold = sensitivity_thresholds(sensitivity)
    for _, row in merged.iterrows():
        value = row["Value"]
        median = row.get("baseline_median")
        mad = row.get("baseline_mad")
        history_count = int(row.get("history_count", 0) or 0)

        if pd.isna(median):
            severity = "No Baseline"
            robust_score = None
            relative_change = None
            deviation = np.nan
            interpretation = "Metric not found in historical dataset"
        else:
            deviation = float(value - median)
            robust_score = robust_z_score(value, median, mad)
            relative_change = (
                (deviation / median) * 100 if median not in (0, np.nan) else np.nan
            )
            severity = categorize_severity(
                robust_score, deviation, mad, history_count, sensitivity
            )
            interpretation = describe_implication(
                severity,
                deviation,
                relative_change,
                robust_score,
                current_value=value,
                baseline_median=median,
                baseline_mad=mad,
                units=row.get("Units", ""),
                history_count=history_count,
                period=row.get("Period", ""),
                mild_threshold=mild_threshold,
                severe_threshold=severe_threshold,
            )

        results.append(
            {
                "Metric": row["Metric"],
                "Period": row["Period"],
                "Units": row["Units"],
                "Current Value": value,
                "Baseline Median": median,
                "Baseline MAD": mad,
                "History Points": history_count,
                "Robust Z": robust_score,
                "Absolute Change": deviation,
                "Percent Change": relative_change,
                "Severity": severity,
                "Interpretation": interpretation,
            }
        )

    results_df = pd.DataFrame(results)
    results_df["SortKey"] = results_df["Severity"].map(SEVERITY_ORDER).fillna(6)
    return results_df.sort_values(["SortKey", "Metric"]).drop(columns=["SortKey"])


def severity_color(severity: str) -> str:
    palette = {
        "Severe": "#c62828",
        "Mild": "#f6c344",
        "Normal": "#388e3c",
        "Stable": "#1976d2",
        "Insufficient History": "#6d6d6d",
        "No Baseline": "#6d6d6d",
    }
    return palette.get(severity, "#6d6d6d")


def render_results_table(results: pd.DataFrame) -> None:
    """Display anomaly assessment in an interactive grid."""

    display_cols = [
        "Metric",
        "Period",
        "Current Value",
        "Absolute Change",
        "Baseline Median",
        "Baseline MAD",
        "Robust Z",
        "Percent Change",
        "Severity",
        "Interpretation",
    ]
    formatted = results.copy()
    formatted["Current Value"] = formatted["Current Value"].map(lambda x: f"{x:,.4g}")
    formatted["Absolute Change"] = formatted["Absolute Change"].map(
        lambda x: "–" if pd.isna(x) else f"{x:+,.4g}"
    )
    formatted["Baseline Median"] = formatted["Baseline Median"].map(
        lambda x: "–" if pd.isna(x) else f"{x:,.4g}"
    )
    formatted["Baseline MAD"] = formatted["Baseline MAD"].map(
        lambda x: "–" if pd.isna(x) else f"{x:,.4g}"
    )
    formatted["Robust Z"] = formatted["Robust Z"].map(
        lambda x: "–" if pd.isna(x) else f"{x:.2f}"
    )
    formatted["Percent Change"] = formatted["Percent Change"].map(
        lambda x: "–" if pd.isna(x) else f"{x:+.1f}%"
    )

    def severity_style(value: str) -> str:
        color = severity_color(value)
        return (
            f"background-color: {color}22; color: {color}; font-weight: 600"
            if isinstance(value, str)
            else ""
        )

    styled = formatted[display_cols].style.map(severity_style, subset=["Severity"])

    st.dataframe(styled, use_container_width=True, hide_index=True)


def format_value(value: Optional[float]) -> str:
    if value is None or pd.isna(value):
        return "–"
    return f"{value:,.4g}"


def format_signed(value: Optional[float]) -> str:
    if value is None or pd.isna(value):
        return "–"
    return f"{value:+,.4g}"


def finalize_chart_layout(chart: alt.Chart, *, height: int) -> alt.Chart:
    """Apply common sizing/padding so chart text isn't clipped."""

    return (
        chart.properties(height=height, width="container", padding={"top": 12, "right": 12, "bottom": 12, "left": 12})
        .configure_view(continuousWidth=600, strokeOpacity=0)
        .configure_axis(labelLimit=220, titleLimit=220, labelPadding=6)
        .configure_legend(labelLimit=220, titleLimit=220)
    )


def collect_metric_frames(
    metric_name: str,
    historical_df: pd.DataFrame,
    new_df: pd.DataFrame,
    period: str,
    current_value: float,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Return filtered history/new frames and a stable period order for plotting."""

    history_points = (
        historical_df[historical_df["Metric"] == metric_name][["Period", "Value"]]
        .copy()
    )
    new_points = new_df[new_df["Metric"] == metric_name][["Period", "Value"]].copy()

    if new_points.empty:
        new_points = pd.DataFrame({"Period": [period], "Value": [current_value]})
    elif period not in new_points["Period"].tolist():
        new_points = pd.concat(
            [
                new_points,
                pd.DataFrame({"Period": [period], "Value": [current_value]}),
            ],
            ignore_index=True,
        )

    history_points["Period"] = history_points["Period"].astype(str)
    new_points["Period"] = new_points["Period"].astype(str)

    period_order = list(dict.fromkeys(history_points["Period"].tolist()))
    for item in new_points["Period"].tolist():
        if item not in period_order:
            period_order.append(item)
    if not period_order:
        combined = pd.concat(
            [df for df in (history_points, new_points) if not df.empty],
            ignore_index=True,
        )
        if not combined.empty:
            period_order = combined["Period"].tolist()
        else:
            period_order = []

    return history_points, new_points, period_order


def severity_from_z(
    z_value: Optional[float],
    *,
    mild_threshold: float = 2.5,
    severe_threshold: float = 4.0,
) -> str:
    if z_value is None or pd.isna(z_value):
        return "Stable"
    magnitude = abs(float(z_value))
    if magnitude >= severe_threshold:
        return "Severe"
    if magnitude >= mild_threshold:
        return "Mild"
    return "Normal"


def theil_sen_slope(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    """Compute a robust slope estimate using the Theil-Sen estimator."""

    if x.size == 0 or y.size == 0:
        return None

    valid = ~(np.isnan(x) | np.isnan(y))
    x = x[valid]
    y = y[valid]

    n = len(x)
    if n < 2:
        return None

    slopes: list[float] = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            dx = x[j] - x[i]
            if np.isclose(dx, 0.0):
                continue
            slopes.append(float((y[j] - y[i]) / dx))

    if not slopes:
        return None

    return float(np.median(slopes))


def build_metric_chart(
    *,
    units: str,
    history_points: pd.DataFrame,
    new_points: pd.DataFrame,
    period_order: list[str],
    baseline_median: Optional[float],
    baseline_mad: Optional[float],
    sensitivity: float,
) -> Optional[alt.Chart]:
    """Create a time-series chart with baseline and anomaly annotations."""

    if history_points.empty and new_points.empty:
        return None

    y_title = f"Value ({units})" if units else "Value"

    history_plot = history_points.copy()
    history_plot["Series"] = "Historical"
    new_plot = new_points.copy()
    new_plot["Series"] = "New"

    scaled_mad: Optional[float] = None
    if (
        baseline_median is not None
        and not pd.isna(baseline_median)
        and baseline_mad is not None
        and not pd.isna(baseline_mad)
        and not np.isclose(baseline_mad, 0.0)
    ):
        scaled_mad = baseline_mad * 1.4826
        if np.isclose(scaled_mad, 0.0):
            scaled_mad = None

    mild_threshold, severe_threshold = sensitivity_thresholds(sensitivity)

    if scaled_mad is not None:
        history_plot["RobustZ"] = (
            (history_plot["Value"] - baseline_median) / scaled_mad
        )
        new_plot["RobustZ"] = (new_plot["Value"] - baseline_median) / scaled_mad
    else:
        history_plot["RobustZ"] = np.nan
        new_plot["RobustZ"] = np.nan

    tooltip = [
        alt.Tooltip("Series:N", title="Type"),
        alt.Tooltip("Period:N", title="Period"),
        alt.Tooltip("Value:Q", title=y_title, format=".4g"),
    ]
    if scaled_mad is not None:
        tooltip.append(alt.Tooltip("RobustZ:Q", title="Robust z", format=".2f"))

    layers = []

    sort_order = period_order if period_order else None

    if not history_plot.empty:
        history_line = (
            alt.Chart(history_plot)
            .mark_line(color="#1f77b4")
            .encode(
                x=alt.X("Period:N", sort=sort_order, title="Period"),
                y=alt.Y("Value:Q", title=y_title),
            )
        )
        history_points_layer = (
            alt.Chart(history_plot)
            .mark_point(size=70, filled=True)
            .encode(
                x=alt.X("Period:N", sort=sort_order, title="Period"),
                y=alt.Y("Value:Q", title=y_title),
                tooltip=tooltip,
            )
        )
        layers.extend([history_line, history_points_layer])

    if not new_plot.empty:
        new_points_layer = (
            alt.Chart(new_plot)
            .mark_point(size=140, color="#d62728", filled=True)
            .encode(
                x=alt.X("Period:N", sort=sort_order, title="Period"),
                y=alt.Y("Value:Q", title=y_title),
                tooltip=tooltip,
            )
        )
        layers.append(new_points_layer)

    if baseline_median is not None and not pd.isna(baseline_median):
        baseline_rule = (
            alt.Chart(pd.DataFrame({"Value": [baseline_median]}))
            .mark_rule(color="#4c566a", strokeDash=[4, 4])
            .encode(y=alt.Y("Value:Q"))
        )
        layers.append(baseline_rule)

    if scaled_mad is not None:
        mild_delta = mild_threshold * scaled_mad
        severe_delta = severe_threshold * scaled_mad
        thresholds = [
            (baseline_median + mild_delta, severity_color("Mild")),
            (baseline_median - mild_delta, severity_color("Mild")),
            (baseline_median + severe_delta, severity_color("Severe")),
            (baseline_median - severe_delta, severity_color("Severe")),
        ]
        for value, color in thresholds:
            if value is None or pd.isna(value):
                continue
            threshold_rule = (
                alt.Chart(pd.DataFrame({"Value": [value]}))
                .mark_rule(color=color, strokeDash=[2, 2])
                .encode(y=alt.Y("Value:Q"))
            )
            layers.append(threshold_rule)

    if not layers:
        return None

    chart = alt.layer(*layers).resolve_scale(color="independent")
    return finalize_chart_layout(chart, height=260)


def build_zscore_timeline(
    *,
    history_points: pd.DataFrame,
    new_points: pd.DataFrame,
    period_order: list[str],
    baseline_median: Optional[float],
    baseline_mad: Optional[float],
    sensitivity: float,
) -> Optional[alt.Chart]:
    """Display robust z-scores over time with alert thresholds."""

    if (
        baseline_median is None
        or pd.isna(baseline_median)
        or baseline_mad is None
        or pd.isna(baseline_mad)
        or np.isclose(baseline_mad, 0.0)
    ):
        return None

    scaled_mad = baseline_mad * 1.4826
    if np.isclose(scaled_mad, 0.0):
        return None

    history_plot = history_points.copy()
    new_plot = new_points.copy()

    mild_threshold, severe_threshold = sensitivity_thresholds(sensitivity)

    history_plot["RobustZ"] = (
        (history_plot["Value"] - baseline_median) / scaled_mad
        if not history_plot.empty
        else pd.Series(dtype=float)
    )
    new_plot["RobustZ"] = (
        (new_plot["Value"] - baseline_median) / scaled_mad
        if not new_plot.empty
        else pd.Series(dtype=float)
    )

    history_plot["Severity"] = history_plot["RobustZ"].apply(
        lambda z: severity_from_z(
            z, mild_threshold=mild_threshold, severe_threshold=severe_threshold
        )
    )
    new_plot["Severity"] = new_plot["RobustZ"].apply(
        lambda z: severity_from_z(
            z, mild_threshold=mild_threshold, severe_threshold=severe_threshold
        )
    )

    y_field = alt.Y("RobustZ:Q", title="Robust z-score")
    x_field = alt.X("Period:N", sort=(period_order if period_order else None), title="Period")

    layers = []

    if not history_plot.empty:
        history_line = (
            alt.Chart(history_plot)
            .mark_line(color="#636efa")
            .encode(x=x_field, y=y_field)
        )
        history_points_layer = (
            alt.Chart(history_plot)
            .mark_point(size=60, filled=True)
            .encode(
                x=x_field,
                y=y_field,
                color=alt.Color(
                    "Severity:N",
                    scale=alt.Scale(
                        domain=["Severe", "Mild", "Normal", "Stable"],
                        range=[
                            severity_color("Severe"),
                            severity_color("Mild"),
                            severity_color("Normal"),
                            severity_color("Stable"),
                        ],
                    ),
                    legend=alt.Legend(title="Z flag"),
                ),
                tooltip=[
                    alt.Tooltip("Period:N", title="Period"),
                    alt.Tooltip("RobustZ:Q", title="Robust z", format=".2f"),
                    alt.Tooltip("Severity:N", title="Flag"),
                ],
            )
        )
        layers.extend([history_line, history_points_layer])

    if not new_plot.empty:
        new_points_layer = (
            alt.Chart(new_plot)
            .mark_point(size=140, color="#d62728", filled=True, shape="triangle-up")
            .encode(
                x=x_field,
                y=y_field,
                tooltip=[
                    alt.Tooltip("Period:N", title="Period"),
                    alt.Tooltip("RobustZ:Q", title="Robust z", format=".2f"),
                    alt.Tooltip("Severity:N", title="Flag"),
                ],
            )
        )
        layers.append(new_points_layer)

    zero_line = (
        alt.Chart(pd.DataFrame({"RobustZ": [0]}))
        .mark_rule(color="#4c566a", strokeDash=[4, 4])
        .encode(y=y_field)
    )
    layers.append(zero_line)

    for threshold, label_color in [
        (mild_threshold, severity_color("Mild")),
        (-mild_threshold, severity_color("Mild")),
        (severe_threshold, severity_color("Severe")),
        (-severe_threshold, severity_color("Severe")),
    ]:
        rule = (
            alt.Chart(pd.DataFrame({"RobustZ": [threshold]}))
            .mark_rule(color=label_color, strokeDash=[2, 2])
            .encode(y=y_field)
        )
        layers.append(rule)

    chart = alt.layer(*layers).resolve_scale(color="independent")
    return finalize_chart_layout(chart, height=260)


def build_distribution_snapshot(
    *,
    history_points: pd.DataFrame,
    baseline_median: Optional[float],
    baseline_mad: Optional[float],
    units: str,
    sensitivity: float,
) -> Optional[alt.Chart]:
    """Render a distribution view of historical values with thresholds."""

    if history_points.empty:
        return None

    y_title = "Frequency"
    x_title = f"Value ({units})" if units else "Value"

    chart = (
        alt.Chart(history_points)
        .mark_bar(color="#9ecae1", opacity=0.8)
        .encode(
            x=alt.X("Value:Q", bin=alt.Bin(maxbins=30), title=x_title),
            y=alt.Y("count():Q", title=y_title),
            tooltip=[
                alt.Tooltip("count():Q", title="Count"),
            ],
        )
    )

    layers = [chart]

    if baseline_median is not None and not pd.isna(baseline_median):
        median_rule = (
            alt.Chart(pd.DataFrame({"Value": [baseline_median]}))
            .mark_rule(color="#4c566a", strokeDash=[4, 4])
            .encode(x=alt.X("Value:Q", title=x_title))
        )
        layers.append(median_rule)

    if (
        baseline_mad is not None
        and not pd.isna(baseline_mad)
        and not np.isclose(baseline_mad, 0.0)
        and baseline_median is not None
        and not pd.isna(baseline_median)
    ):
        mild_threshold, severe_threshold = sensitivity_thresholds(sensitivity)
        scaled_mad = baseline_mad * 1.4826
        if not np.isclose(scaled_mad, 0.0):
            mild_delta = mild_threshold * scaled_mad
            severe_delta = severe_threshold * scaled_mad
            thresholds = [
                (baseline_median + mild_delta, severity_color("Mild")),
                (baseline_median - mild_delta, severity_color("Mild")),
                (baseline_median + severe_delta, severity_color("Severe")),
                (baseline_median - severe_delta, severity_color("Severe")),
            ]
            for value, color in thresholds:
                if value is None or pd.isna(value):
                    continue
                rule = (
                    alt.Chart(pd.DataFrame({"Value": [value]}))
                    .mark_rule(color=color, strokeDash=[2, 2])
                    .encode(x=alt.X("Value:Q", title=x_title))
                )
                layers.append(rule)

    chart = alt.layer(*layers)
    return finalize_chart_layout(chart, height=220)


def describe_metric_trend(
    *,
    history_points: pd.DataFrame,
    new_points: pd.DataFrame,
    period_order: list[str],
    baseline_mad: Optional[float],
    units: str,
) -> Optional[dict[str, Any]]:
    """Generate qualitative and quantitative trend context for the metric."""

    combined = pd.concat([history_points, new_points], ignore_index=True)
    combined = combined.dropna(subset=["Value"]).copy()
    if combined.empty:
        return None

    if period_order:
        order_map = {period: idx for idx, period in enumerate(period_order)}
        combined["Order"] = combined["Period"].map(order_map)
    else:
        combined = combined.reset_index(drop=True)
        combined["Order"] = combined.index.astype(float)

    combined = combined.dropna(subset=["Order"])
    if combined.empty:
        return None

    combined = (
        combined.groupby("Order", as_index=False)["Value"].mean().sort_values("Order")
    )

    order = combined["Order"].to_numpy(dtype=float)
    values = combined["Value"].to_numpy(dtype=float)

    slope = theil_sen_slope(order, values)
    units_label = f" {units}" if units else ""
    per_period_suffix = f"{units_label}/period" if units_label else " per period"

    direction_symbols = {
        "increasing": "↗",
        "decreasing": "↘",
        "steady": "→",
    }
    direction_titles = {
        "increasing": "Rising",
        "decreasing": "Falling",
        "steady": "Steady",
    }

    if slope is None or np.isclose(slope, 0.0, atol=1e-9):
        summary = "Trend: holding steady (slope ≈ 0)."
        delta_text = f"≈ 0{per_period_suffix}".strip()
        return {
            "summary": summary,
            "direction": "steady",
            "descriptor": "steady",
            "label": f"{direction_symbols['steady']} {direction_titles['steady']}",
            "slope": float(slope or 0.0),
            "slope_text": delta_text,
            "normalized": None,
        }

    span = order.max() - order.min()
    projected_shift = slope * span if span > 0 else slope

    normalized = None
    if (
        baseline_mad is not None
        and not pd.isna(baseline_mad)
        and not np.isclose(baseline_mad, 0.0)
    ):
        normalized = abs(projected_shift) / baseline_mad

    direction = "increasing" if slope > 0 else "decreasing"
    magnitude_descriptor = "gradually"

    if normalized is not None:
        if normalized >= 1.5:
            magnitude_descriptor = "strongly"
        elif normalized >= 0.5:
            magnitude_descriptor = "gradually"
        else:
            summary = (
                "Trend: holding steady (movement over the window is small relative to"
                " typical variability)."
            )
            delta_text = f"≈ 0{per_period_suffix}".strip()
            return {
                "summary": summary,
                "direction": "steady",
                "descriptor": "steady",
                "label": f"{direction_symbols['steady']} {direction_titles['steady']}",
                "slope": float(slope),
                "slope_text": delta_text,
                "normalized": normalized,
            }

    slope_text = f"{slope:+.4g}{per_period_suffix}".strip()
    span_periods = int(span) if span and span >= 1 else len(values) - 1
    if span_periods < 1:
        span_periods = len(values)

    summary = (
        f"Trend: {magnitude_descriptor} {direction} (≈ {slope_text} across the last"
        f" {span_periods if span_periods > 0 else len(values)} periods)."
    )

    return {
        "summary": summary,
        "direction": direction,
        "descriptor": magnitude_descriptor,
        "label": f"{direction_symbols[direction]} {direction_titles[direction]}",
        "slope": float(slope),
        "slope_text": slope_text,
        "normalized": normalized,
    }


def build_anomaly_gauge(
    robust_score: Optional[float],
    *,
    sensitivity: float,
) -> Optional[go.Figure]:
    """Render a semicircular gauge for the robust z-score magnitude."""

    if robust_score is None or pd.isna(robust_score):
        return None

    magnitude = float(abs(robust_score))
    mild_threshold, severe_threshold = sensitivity_thresholds(sensitivity)

    limit = max(
        4.5,
        magnitude * 1.2,
        severe_threshold * 1.2,
        mild_threshold * 2.0,
    )

    steps = []
    normal_cap = min(mild_threshold, limit)
    mild_cap = min(severe_threshold, limit)
    if normal_cap > 0:
        steps.append({"range": [0, normal_cap], "color": severity_color("Normal")})
    if mild_cap > normal_cap:
        steps.append({"range": [normal_cap, mild_cap], "color": severity_color("Mild")})
    if limit > mild_cap:
        steps.append({"range": [mild_cap, limit], "color": severity_color("Severe")})

    bar_color = "#333333"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=magnitude,
            number={"suffix": " |z|"},
            gauge={
                "shape": "angular",
                "axis": {"range": [0, limit], "tickwidth": 1, "tickcolor": "#666"},
                "bar": {"color": bar_color},
                "steps": steps,
                "threshold": {
                    "line": {"color": bar_color, "width": 4},
                    "thickness": 0.75,
                    "value": magnitude,
                },
            },
            title={"text": "Robust Anomaly Score", "font": {"size": 12}},
        )
    )

    fig.update_layout(
        margin=dict(l=8, r=8, t=55, b=20),
        height=220,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#222", size=14),
    )

    return fig


def summarize_changes(metric_row: pd.Series) -> str:
    value = metric_row.get("Current Value")
    median = metric_row.get("Baseline Median")
    units = metric_row.get("Units", "")
    abs_change = metric_row.get("Absolute Change")
    pct_change = metric_row.get("Percent Change")
    mad = metric_row.get("Baseline MAD")

    units_label = f" {units}" if units else ""
    sentences: list[str] = []

    if value is None or pd.isna(value):
        sentences.append("Current value is unavailable.")
    else:
        sentences.append(f"Current value: {format_value(value)}{units_label}.")

    if median is not None and not pd.isna(median):
        deviation_available = (
            abs_change is not None
            and not pd.isna(abs_change)
            and not np.isclose(abs_change, 0.0, atol=1e-9)
        )
        if deviation_available:
            direction = "higher" if abs_change > 0 else "lower"
            change_bits: list[str] = []
            if abs_change is not None and not pd.isna(abs_change):
                change_bits.append(f"{abs(abs_change):,.4g}{units_label}".strip())
            if pct_change is not None and not pd.isna(pct_change):
                change_bits.append(f"{pct_change:+.1f}%")
            change_text = " / ".join(change_bits) if change_bits else "a small margin"
            sentences.append(
                f"Typical level is about {format_value(median)}{units_label}; current is {direction} by {change_text}."
            )
        else:
            sentences.append(
                f"Current performance is right in line with the usual {format_value(median)}{units_label}."
            )
    else:
        sentences.append("We don't yet have enough history to say what's typical.")

    if (
        mad is not None
        and not pd.isna(mad)
        and mad != 0
        and abs_change is not None
        and not pd.isna(abs_change)
    ):
        multiples = abs(abs_change) / mad
        sentences.append(
            f"This is roughly {multiples:.1f}× the size of a normal period-to-period move."
        )

    return " ".join(sentences)


def main() -> None:
    st.title("KPI Anomaly Review")
    

    tab_upload, tab_results = st.tabs(["Upload & Overview", "Anomaly Review"])

    historical_df: Optional[pd.DataFrame] = None
    new_df: Optional[pd.DataFrame] = None
    history_error: Optional[str] = None
    new_error: Optional[str] = None

    history_empty = False
    new_empty = False

    with tab_upload:
        st.subheader("Upload Data")
        historical_file = st.file_uploader(
            "Historical data CSV",
            type="csv",
            key="history",
            help="Requires columns: Period, Metric, Value, Units",
        )
        new_data_file = st.file_uploader(
            "New data points CSV",
            type="csv",
            key="newdata",
        )

        st.markdown("**Detection Sensitivity**")
        st.slider(
            "Tune how quickly anomalies trigger",
            min_value=0.5,
            max_value=1.5,
            value=1.0,
            step=0.1,
            key="sensitivity_control",
            help="Higher sensitivity lowers alert thresholds; lower sensitivity raises them.",
        )

        if historical_file is not None:
            try:
                historical_df = load_dataset(historical_file)
            except ValueError as exc:
                history_error = str(exc)
                st.error(history_error)
            except Exception as exc:  # pragma: no cover
                history_error = str(exc)
                st.error(history_error)
            else:
                if historical_df.empty:
                    history_empty = True
                    st.warning("Historical dataset is empty; anomaly scoring is not possible.")
                else:
                    st.markdown("**Historical sample**")
                    st.dataframe(historical_df.head(200), use_container_width=True)

        if new_data_file is not None:
            try:
                new_df = load_dataset(new_data_file)
            except ValueError as exc:
                new_error = str(exc)
                st.error(new_error)
            except Exception as exc:  # pragma: no cover
                new_error = str(exc)
                st.error(new_error)
            else:
                if new_df.empty:
                    new_empty = True
                    st.warning("New data CSV is empty.")
                else:
                    st.markdown("**New data sample**")
                    st.dataframe(new_df.head(200), use_container_width=True)

    with tab_results:
        if history_error or new_error:
            st.info("Resolve upload issues in the other tab to see anomaly scores.")
            return

        if historical_df is None or new_df is None:
            st.info("Upload both historical and new datasets in the Upload tab.")
            return

        if history_empty:
            st.warning("Historical dataset must contain at least one row.")
            return
        if new_empty:
            st.warning("New dataset must contain at least one row.")
            return

        sensitivity_factor = st.session_state.get("sensitivity_control", 1.0)
        results = evaluate_new_points(historical_df, new_df, sensitivity_factor)
        ordered_results = results.assign(
            _sev=results["Severity"].map(SEVERITY_ORDER).fillna(6),
            _idx=np.arange(len(results)),
        ).sort_values(["_sev", "Metric", "_idx"])

        for _, metric_row in ordered_results.iterrows():
            sev = metric_row["Severity"]
            badge = SEVERITY_BADGE.get(sev, sev)
            title = f"{badge} · {metric_row['Metric']}"
            expand = sev in {"Severe", "Mild"}
            with st.expander(title, expanded=expand):
                history_points, new_points, period_order = collect_metric_frames(
                    metric_name=metric_row["Metric"],
                    historical_df=historical_df,
                    new_df=new_df,
                    period=metric_row["Period"],
                    current_value=metric_row["Current Value"],
                )

                timeseries_chart = build_metric_chart(
                    units=metric_row.get("Units", ""),
                    history_points=history_points,
                    new_points=new_points,
                    period_order=period_order,
                    baseline_median=metric_row.get("Baseline Median"),
                    baseline_mad=metric_row.get("Baseline MAD"),
                    sensitivity=sensitivity_factor,
                )
                z_chart = build_zscore_timeline(
                    history_points=history_points,
                    new_points=new_points,
                    period_order=period_order,
                    baseline_median=metric_row.get("Baseline Median"),
                    baseline_mad=metric_row.get("Baseline MAD"),
                    sensitivity=sensitivity_factor,
                )
                distribution_chart = build_distribution_snapshot(
                    history_points=history_points,
                    baseline_median=metric_row.get("Baseline Median"),
                    baseline_mad=metric_row.get("Baseline MAD"),
                    units=metric_row.get("Units", ""),
                    sensitivity=sensitivity_factor,
                )
                gauge_chart = build_anomaly_gauge(
                    metric_row.get("Robust Z"), sensitivity=sensitivity_factor
                )

                rendered = False
                if timeseries_chart is not None:
                    st.markdown("**Metric Trajectory**")
                    st.altair_chart(
                        timeseries_chart,
                        use_container_width=True,
                        key=f"metric-trajectory-{metric_row.get('Metric', '')}-{metric_row.get('Period', '')}",
                    )
                    rendered = True

                if z_chart is not None:
                    st.markdown("**Robust Z-Score Trend**")
                    st.altair_chart(
                        z_chart,
                        use_container_width=True,
                        key=f"robust-z-trend-{metric_row.get('Metric', '')}-{metric_row.get('Period', '')}",
                    )
                    rendered = True

                if not rendered:
                    st.info("No time-series diagnostics available for this metric.")

                if distribution_chart is not None:
                    st.markdown("**Historical Distribution**")
                    st.altair_chart(
                        distribution_chart,
                        use_container_width=True,
                        key=f"distribution-{metric_row.get('Metric', '')}-{metric_row.get('Period', '')}",
                    )

                trend_info = describe_metric_trend(
                    history_points=history_points,
                    new_points=new_points,
                    period_order=period_order,
                    baseline_mad=metric_row.get("Baseline MAD"),
                    units=metric_row.get("Units", ""),
                )
                if trend_info:
                    st.markdown("**Trend Insight**")
                    st.caption(trend_info["summary"])

                delta = metric_row.get("Percent Change")
                abs_delta = metric_row.get("Absolute Change")

                (
                    col_current,
                    col_abs,
                    col_pct,
                    col_median,
                    col_mad,
                    col_z,
                    col_trend,
                ) = st.columns((1, 1, 1, 1, 1, 1, 1.2))
                col_current.metric(
                    f"Current ({metric_row.get('Units', '')})".rstrip(" ()"),
                    format_value(metric_row.get("Current Value")),
                )
                col_abs.metric(
                    "Absolute change",
                    format_signed(abs_delta),
                )
                col_pct.metric(
                    "Percent change",
                    "–" if pd.isna(delta) else f"{delta:+.1f}%",
                )
                col_median.metric(
                    "Baseline median",
                    format_value(metric_row.get("Baseline Median")),
                )
                col_mad.metric(
                    "Baseline MAD",
                    format_value(metric_row.get("Baseline MAD")),
                )
                col_z.metric(
                    "Robust z-score",
                    "–"
                    if pd.isna(metric_row.get("Robust Z"))
                    else f"{metric_row.get('Robust Z'):.2f}",
                )
                if trend_info:
                    col_trend.metric(
                        "Trend",
                        trend_info["label"],
                        trend_info.get("slope_text", ""),
                    )
                else:
                    col_trend.metric("Trend", "–", "")

                if gauge_chart is not None:
                    st.markdown("**Anomaly Gauge**")
                    st.plotly_chart(
                        gauge_chart,
                        use_container_width=True,
                        key=f"gauge-{metric_row.get('Metric', '')}-{metric_row.get('Period', '')}",
                    )
                else:
                    st.caption("Robust z-score gauge will appear once available.")

                st.caption(
                    f"History points: {int(metric_row.get('History Points', 0) or 0)}"
                )
                st.markdown("**Change Summary**")
                st.write(summarize_changes(metric_row))
                st.markdown("**Interpretation**")
                st.write(metric_row.get("Interpretation", ""))

        st.subheader("Anomaly Summary Table")
        render_results_table(results)

        st.download_button(
            label="Download anomaly assessment",
            data=results.to_csv(index=False).encode("utf-8"),
            file_name="kpi_anomaly_results.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()

