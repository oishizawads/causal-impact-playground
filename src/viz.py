"""可視化ヘルパー: 群別トレンドグラフと推定結果比較図。"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import japanize_matplotlib  # noqa: F401  日本語ラベルの文字化け防止
    _JP_OK = True
except Exception:  # フォント未導入でも落ちない
    _JP_OK = False

from .data import DataConfig
from .estimation import EstimationResult


def plot_group_trends(df: pd.DataFrame, cfg: DataConfig) -> plt.Figure:
    """群別の平均推移グラフ。事前/事後を縦線で区切る。"""
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)

    if df.empty:
        ax.text(0.5, 0.5, "データがありません", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return fig

    summary = df.groupby(["time", "group"], observed=True)["y"].mean().reset_index()
    pivot = summary.pivot(index="time", columns="group", values="y")

    for col, color, marker in (("対照群", "#4C72B0", "o"), ("処置群", "#C44E52", "s")):
        if col in pivot.columns:
            ax.plot(
                pivot.index,
                pivot[col],
                marker=marker,
                color=color,
                label=col,
                linewidth=2,
            )

    if cfg.n_pre >= 1 and cfg.n_post >= 1:
        boundary = cfg.n_pre - 0.5
        ax.axvline(boundary, color="#888888", linestyle="--", linewidth=1.2)
        ymax = pivot.max().max() if not pivot.empty else 1.0
        ax.text(boundary + 0.1, ymax, "  施策導入", color="#555555", fontsize=9, va="top")

    ax.set_xlabel("時間（期）")
    ax.set_ylabel("平均アウトカム y")
    ax.set_title("群別トレンド: 施策前後の平均推移")
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    return fig


def plot_estimate_comparison(results: list[EstimationResult]) -> plt.Figure:
    """推定値と真の効果を信頼区間付きで比較する棒図（y軸0起点ではない点図）。"""
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)

    names = [r.name for r in results]
    estimates = [r.estimate for r in results]
    lowers = [r.ci_lower for r in results]
    uppers = [r.ci_upper for r in results]
    true_eff = results[0].true_effect if results else 0.0

    y_pos = np.arange(len(names))
    ax.axvline(true_eff, color="#55A868", linestyle="--", linewidth=2, label=f"真の効果 = {true_eff:g}")

    for i, (est, lo, hi) in enumerate(zip(estimates, lowers, uppers)):
        if np.isfinite(est):
            ax.errorbar(
                est,
                i,
                xerr=[[est - lo], [hi - est]],
                fmt="o",
                color="#4C72B0",
                capsize=5,
                markersize=8,
            )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel("効果推定値（95% 信頼区間）")
    ax.set_title("推定値と真の効果の比較")
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, axis="x", alpha=0.3)
    return fig
