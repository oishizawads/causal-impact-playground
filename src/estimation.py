"""推定ロジック: 単純前後差と DID。

処置群のみの前後差（トレンドを効果に混入するバイアス例）と、
対照群を用いた Difference-in-Differences (DID) を OLS で推定し、
信頼区間付きで返す。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


@dataclass(frozen=True)
class EstimationResult:
    """推定結果と診断量。"""

    name: str
    estimate: float
    se: float
    ci_lower: float
    ci_upper: float
    true_effect: float
    n_obs: int

    @property
    def bias(self) -> float:
        """推定値と真の効果の差（バイアス）。"""
        return float(self.estimate - self.true_effect)

    @property
    def covers_true(self) -> bool:
        """信頼区間が真の効果を含むか。"""
        return bool(self.ci_lower <= self.true_effect <= self.ci_upper)


def _empty_result(name: str, true_effect: float, message: str = "") -> EstimationResult:
    """推定不能時の空結果（UI が落ちないようにする）。"""
    nan = float("nan")
    return EstimationResult(
        name=f"{name}（{message}未推定）" if message else f"{name}（未推定）",
        estimate=nan,
        se=nan,
        ci_lower=nan,
        ci_upper=nan,
        true_effect=true_effect,
        n_obs=0,
    )


def pre_post_difference(df: pd.DataFrame, true_effect: float, alpha: float = 0.05) -> EstimationResult:
    """処置群のみの前後差を OLS で推定する。

    y ~ post （処置群のサブセット）。post 係数 = 前後差。
    トレンドが存在するときこの推定値は真の効果からバイアスされる。
    """
    treat_df = df.query("treat == 1")
    if treat_df.empty or treat_df["post"].nunique() < 2:
        return _empty_result("単純前後差", true_effect, "処置群の前後データ不足・")

    try:
        fit = smf.ols("y ~ post", data=treat_df).fit(cov_type="HC1")
    except Exception:  # 極端パラメータで数値発散した場合
        return _empty_result("単純前後差", true_effect, "計算失敗・")

    ci = fit.conf_int(alpha=alpha).loc["post"]
    return EstimationResult(
        name="単純前後差（処置群のみ）",
        estimate=float(fit.params["post"]),
        se=float(fit.bse["post"]),
        ci_lower=float(ci.iloc[0]),
        ci_upper=float(ci.iloc[1]),
        true_effect=float(true_effect),
        n_obs=int(fit.nobs),
    )


def did_estimate(df: pd.DataFrame, true_effect: float, alpha: float = 0.05) -> EstimationResult:
    """Difference-in-Differences を OLS で推定する。

    y ~ treat*post。treat:post 係数 = DID推定量。
    平行トレンド仮定が成立すれば真の効果に一致する。
    """
    if df.empty or df["treat"].nunique() < 2 or df["post"].nunique() < 2:
        return _empty_result("DID", true_effect, "群×期間データ不足・")

    try:
        fit = smf.ols("y ~ treat * post", data=df).fit(cov_type="HC1")
    except Exception:
        return _empty_result("DID", true_effect, "計算失敗・")

    key = "treat:post"
    if key not in fit.params.index:
        return _empty_result("DID", true_effect, "交互作用項なし・")

    ci = fit.conf_int(alpha=alpha).loc[key]
    return EstimationResult(
        name="DID（差の差）",
        estimate=float(fit.params[key]),
        se=float(fit.bse[key]),
        ci_lower=float(ci.iloc[0]),
        ci_upper=float(ci.iloc[1]),
        true_effect=float(true_effect),
        n_obs=int(fit.nobs),
    )


def parallel_trends_check(df: pd.DataFrame) -> dict[str, float]:
    """事前期間の群別トレンド傾きを比較し、平行トレンドの目安を返す。

    Returns:
        事前期間の処置群・対照群それぞれの平均経時変化（近似傾き）。
    """
    pre = df.query("post == 0")
    if pre.empty:
        return {"treat_slope": float("nan"), "control_slope": float("nan")}

    slopes: dict[str, float] = {}
    for treat_val, key in ((0, "control_slope"), (1, "treat_slope")):
        sub = pre.query("treat == @treat_val")
        if sub["time"].nunique() < 2:
            slopes[key] = 0.0
            continue
        x = sub["time"].to_numpy(dtype=float)
        y = sub["y"].to_numpy(dtype=float)
        slope = float(np.polyfit(x, y, 1)[0]) if x.var() > 0 else 0.0
        slopes[key] = slope

    return slopes
