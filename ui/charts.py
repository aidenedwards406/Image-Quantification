"""Agreement charts: model % vs hand-measured %."""

from __future__ import annotations

import altair as alt
import pandas as pd


def scatter(df: pd.DataFrame, x: str = "manual", y: str = "model", tol: float = 5.0, color: str | None = None,
            x_title: str = "Hand-measured % damage"):
    """Model vs manual % damage with the y = x line and a ±tol band."""
    hi = float(max(df[x].max(), df[y].max(), 5.0)) * 1.05
    line = pd.DataFrame({"a": [0, hi]})
    band = pd.DataFrame({"a": [0, hi], "lo": [-tol, hi - tol], "hi": [tol, hi + tol]})
    base = alt.Chart(df)
    enc = dict(x=alt.X(f"{x}:Q", title=x_title, scale=alt.Scale(domain=[0, hi])),
               y=alt.Y(f"{y}:Q", title="Model % damage", scale=alt.Scale(domain=[0, hi])),
               tooltip=[c for c in ("name", "vessel", "section", x, y) if c in df.columns])
    if color and color in df.columns:
        enc["color"] = alt.Color(f"{color}:N", title=color.capitalize(), legend=alt.Legend(orient="bottom"))
    pts = base.mark_circle(size=55, opacity=0.8).encode(**enc)
    b = alt.Chart(band).mark_area(opacity=0.12, color="#4C9F70").encode(x="a:Q", y="lo:Q", y2="hi:Q")
    ln = alt.Chart(line).mark_line(color="#888", strokeDash=[4, 4]).encode(x="a:Q", y="a:Q")
    return (b + ln + pts).properties(height=340)


def bland_altman(df: pd.DataFrame, x: str = "manual", y: str = "model", stats: dict | None = None):
    d = df.copy()
    d["mean"] = (d[x] + d[y]) / 2
    d["diff"] = d[y] - d[x]
    pts = alt.Chart(d).mark_circle(size=50, opacity=0.8).encode(
        x=alt.X("mean:Q", title="Average of model and hand-measured %"),
        y=alt.Y("diff:Q", title="Model − hand-measured (points)"),
        tooltip=[c for c in ("name", "vessel", "section", "mean", "diff") if c in d.columns])
    layers = [pts]
    if stats and stats.get("n"):
        lines = pd.DataFrame({"v": [stats["bias"], stats["loa_low"], stats["loa_high"], 5.0, -5.0],
                              "kind": ["bias", "95% limits", "95% limits", "±5 target", "±5 target"]})
        layers.append(alt.Chart(lines).mark_rule(strokeDash=[5, 4]).encode(
            y="v:Q", color=alt.Color("kind:N", title=None, legend=alt.Legend(orient="bottom"),
                                     scale=alt.Scale(domain=["bias", "95% limits", "±5 target"],
                                                     range=["#3B6FB6", "#C0504D", "#4C9F70"]))))
    return alt.layer(*layers).properties(height=340)


def calibration_curve(grid, mae, best, current):
    df = pd.DataFrame({"threshold": grid, "mae": mae})
    line = alt.Chart(df).mark_line().encode(x=alt.X("threshold:Q", title="Damage threshold"),
                                            y=alt.Y("mae:Q", title="Avg error (points)"))
    marks = pd.DataFrame({"threshold": [best, current], "label": ["best", "current"]})
    rules = alt.Chart(marks).mark_rule(strokeDash=[4, 3]).encode(
        x="threshold:Q", color=alt.Color("label:N", title=None, legend=alt.Legend(orient="bottom")))
    return (line + rules).properties(height=240)
