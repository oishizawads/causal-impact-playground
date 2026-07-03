"""データ生成モジュールのテスト。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import DataConfig, generate_panel_data, validate_config


def test_generate_shape_and_columns():
    cfg = DataConfig(n_pre=4, n_post=3, n_per_cell=10)
    df = generate_panel_data(cfg)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2 * (4 + 3) * 10
    for col in ("y", "treat", "time", "post", "group", "period"):
        assert col in df.columns


def test_reproducibility_with_fixed_seed():
    cfg = DataConfig(seed=7)
    a = generate_panel_data(cfg)
    b = generate_panel_data(cfg)
    pd.testing.assert_frame_equal(a, b)


def test_different_seed_changes_data():
    a = generate_panel_data(DataConfig(seed=1, noise_sd=5.0))
    b = generate_panel_data(DataConfig(seed=2, noise_sd=5.0))
    assert not np.allclose(a["y"].to_numpy(), b["y"].to_numpy())


def test_true_effect_only_appears_in_treatment_post():
    cfg = DataConfig(
        n_pre=3, n_post=3, n_per_cell=200,
        baseline=100.0, trend=0.0, group_offset=0.0,
        trend_diff=0.0, true_effect=10.0, noise_sd=0.0,
    )
    df = generate_panel_data(cfg)
    # ノイズ0・トレンド0なので、処置群事後だけ10大きい
    means = df.groupby(["treat", "post"])["y"].mean()
    assert means.loc[(0, 0)] == pytest.approx(100.0)
    assert means.loc[(0, 1)] == pytest.approx(100.0)
    assert means.loc[(1, 0)] == pytest.approx(100.0)
    assert means.loc[(1, 1)] == pytest.approx(110.0)


def test_trend_diff_creates_parallel_trends_violation():
    cfg = DataConfig(
        n_pre=4, n_post=4, n_per_cell=200,
        trend=0.0, trend_diff=2.0, true_effect=0.0, noise_sd=0.0,
    )
    df = generate_panel_data(cfg)
    pre = df.query("post == 0")
    control_slope = np.polyfit(pre.query("treat==0")["time"], pre.query("treat==0")["y"], 1)[0]
    treat_slope = np.polyfit(pre.query("treat==1")["time"], pre.query("treat==1")["y"], 1)[0]
    assert control_slope == pytest.approx(0.0)
    assert treat_slope == pytest.approx(2.0)


def test_validation_rejects_invalid_params():
    bad = DataConfig(n_pre=0, n_post=0, n_per_cell=0, noise_sd=-1.0)
    errors = validate_config(bad)
    assert len(errors) >= 4
    with pytest.raises(ValueError):
        generate_panel_data(bad)


def test_zero_noise_does_not_crash():
    cfg = DataConfig(noise_sd=0.0)
    df = generate_panel_data(cfg)
    assert len(df) > 0
