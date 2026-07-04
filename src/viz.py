"""可視化ヘルパー: 群別トレンドグラフと推定結果比較図（Altair 版）。

matplotlib / japanize-matplotlib を排除し、Altair Chart を返す純粋関数にした。
stlite (WASM) 環境での日本語豆腐化リスクを排除し、brand テーマと色を統一する。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import altair as alt
except ImportError as e:
    raise ImportError("altair が必要です: pip install altair") from e

from .data import DataConfig
from .estimation import EstimationResult

# デザインシステム §1.2 パレット（brand.py の PALETTE と同一）
_PALETTE = ["#0f766e", "#2563eb", "#d97706", "#be185d", "#7c3aed", "#64748b"]


def plot_group_trends(df: pd.DataFrame, cfg: DataConfig) -> alt.Chart:
    """群別の平均推移グラフ（Altair）。事前/事後を縦線で区切る。"""
    if df.empty:
        placeholder = pd.DataFrame({"x": [0.5], "y": [0.5], "text": ["データがありません"]})
        return (
            alt.Chart(placeholder)
            .mark_text(fontSize=14, color="#5b6474")
            .encode(
                x=alt.X("x:Q", axis=None),
                y=alt.Y("y:Q", axis=None),
                text=alt.Text("text:N"),
            )
            .properties(width="container", height=300)
        )

    summary = df.groupby(["time", "group"], observed=True)["y"].mean().reset_index()

    line = (
        alt.Chart(summary)
        .mark_line(strokeWidth=2.5, point=alt.OverlayMarkDef(filled=True, size=70))
        .encode(
            x=alt.X("time:Q", title="時間（期）", axis=alt.Axis(tickMinStep=1, format="d")),
            y=alt.Y("y:Q", title="平均アウトカム y", scale=alt.Scale(zero=False)),
            color=alt.Color(
                "group:N",
                scale=alt.Scale(range=_PALETTE[:2]),
                legend=alt.Legend(title="群"),
            ),
            tooltip=[
                alt.Tooltip("time:Q", title="時間"),
                alt.Tooltip("group:N", title="群"),
                alt.Tooltip("y:Q", title="平均アウトカム", format=".2f"),
            ],
        )
    )

    if cfg.n_pre >= 1 and cfg.n_post >= 1:
        boundary = cfg.n_pre - 0.5
        rule = (
            alt.Chart(pd.DataFrame({"x": [boundary]}))
            .mark_rule(color="#5b6474", strokeDash=[6, 3], strokeWidth=1.5)
            .encode(x=alt.X("x:Q"))
        )
        label = (
            alt.Chart(pd.DataFrame({"x": [boundary + 0.1], "text": ["施策導入"]}))
            .mark_text(align="left", color="#5b6474", fontSize=11)
            .encode(
                x=alt.X("x:Q"),
                y=alt.value(12),
                text=alt.Text("text:N"),
            )
        )
        chart = (line + rule + label).properties(
            title="群別トレンド: 施策前後の平均推移",
            width="container",
            height=300,
        )
    else:
        chart = line.properties(
            title="群別トレンド: 施策前後の平均推移",
            width="container",
            height=300,
        )

    return chart


def plot_estimate_comparison(results: list[EstimationResult]) -> alt.Chart:
    """推定値と真の効果を信頼区間付きで比較するドット図（Altair）。"""
    rows = [
        {
            "手法": r.name,
            "推定値": r.estimate,
            "下側CI": r.ci_lower,
            "上側CI": r.ci_upper,
        }
        for r in results
    ]
    df = pd.DataFrame(rows)
    true_eff = results[0].true_effect if results else 0.0

    # NaN/inf を除外して描画
    df_valid = df[df["推定値"].apply(np.isfinite)].copy()

    points = (
        alt.Chart(df_valid)
        .mark_point(filled=True, size=100, color=_PALETTE[0])
        .encode(
            y=alt.Y("手法:N", title=None, sort=None, axis=alt.Axis(labelLimit=260)),
            x=alt.X("推定値:Q", title="効果推定値（95% 信頼区間）"),
            tooltip=[
                alt.Tooltip("手法:N"),
                alt.Tooltip("推定値:Q", format=".3f"),
                alt.Tooltip("下側CI:Q", title="CI 下限", format=".3f"),
                alt.Tooltip("上側CI:Q", title="CI 上限", format=".3f"),
            ],
        )
    )

    errorbars = (
        alt.Chart(df_valid)
        .mark_errorbar(ticks=True, color=_PALETTE[0])
        .encode(
            y=alt.Y("手法:N", sort=None),
            x=alt.X("下側CI:Q", title="効果推定値（95% 信頼区間）"),
            x2=alt.X2("上側CI:Q"),
        )
    )

    rule = (
        alt.Chart(pd.DataFrame({"x": [true_eff]}))
        .mark_rule(color="#15803d", strokeDash=[6, 3], strokeWidth=2)
        .encode(x=alt.X("x:Q"))
    )

    return (points + errorbars + rule).properties(
        title=f"推定値と真の効果の比較（真の効果 = {true_eff:g}）",
        width="container",
        height=200,
    )
