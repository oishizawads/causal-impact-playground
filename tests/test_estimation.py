"""推定ロジックのテスト。"""
from __future__ import annotations

import numpy as np
import pytest

from src.data import DataConfig, generate_panel_data
from src.estimation import did_estimate, parallel_trends_check, pre_post_difference


def _big_clean_cfg(true_effect: float = 5.0, trend_diff: float = 0.0) -> DataConfig:
    """DIDが真値を良く復元する大きなサンプル・低ノイズ設定。"""
    return DataConfig(
        seed=0, n_pre=4, n_post=4, n_per_cell=500,
        baseline=100.0, trend=2.0, group_offset=5.0,
        trend_diff=trend_diff, true_effect=true_effect, noise_sd=1.0,
    )


def test_did_recovers_true_effect_under_parallel_trends():
    cfg = _big_clean_cfg(true_effect=5.0, trend_diff=0.0)
    df = generate_panel_data(cfg)
    res = did_estimate(df, cfg.true_effect)
    assert res.n_obs > 0
    assert res.estimate == pytest.approx(5.0, abs=0.5)
    assert res.covers_true


def test_pre_post_is_biased_when_trend_exists():
    cfg = _big_clean_cfg(true_effect=5.0, trend_diff=0.0)
    df = generate_panel_data(cfg)
    res = pre_post_difference(df, cfg.true_effect)
    # 共通トレンド2.0が4期間分=8程度のバイアスが入るはず
    assert res.estimate > cfg.true_effect + 3.0
    assert res.bias > 3.0


def test_did_is_biased_when_parallel_trends_violated():
    cfg = _big_clean_cfg(true_effect=0.0, trend_diff=3.0)
    df = generate_panel_data(cfg)
    res = did_estimate(df, cfg.true_effect)
    # 平行トレンド崩壊でDIDも0から大きく外れる
    assert abs(res.estimate - 0.0) > 2.0


def test_empty_data_returns_unestimated_result():
    import pandas as pd

    empty = pd.DataFrame({"y": [], "treat": [], "post": [], "time": []})
    res = did_estimate(empty, true_effect=1.0)
    assert np.isnan(res.estimate)
    assert res.n_obs == 0


def test_parallel_trends_check_keys():
    cfg = DataConfig(n_pre=3, n_post=3, n_per_cell=50, trend_diff=1.0)
    df = generate_panel_data(cfg)
    slopes = parallel_trends_check(df)
    assert set(slopes.keys()) == {"treat_slope", "control_slope"}
    assert slopes["treat_slope"] > slopes["control_slope"]


def test_result_fields_are_finite_for_normal_case():
    cfg = _big_clean_cfg()
    df = generate_panel_data(cfg)
    res = did_estimate(df, cfg.true_effect)
    for v in (res.estimate, res.se, res.ci_lower, res.ci_upper):
        assert np.isfinite(v)
    assert res.ci_lower <= res.estimate <= res.ci_upper
