"""viz.py の Altair 化テスト。"""
from __future__ import annotations

import pandas as pd
import pytest

from src.data import DataConfig, generate_panel_data
from src.estimation import did_estimate, pre_post_difference
from src.viz import plot_estimate_comparison, plot_group_trends


def _default_cfg() -> DataConfig:
    return DataConfig(seed=0, n_pre=3, n_post=3, n_per_cell=20, noise_sd=1.0)


def test_plot_group_trends_returns_altair_chart():
    import altair as alt

    cfg = _default_cfg()
    df = generate_panel_data(cfg)
    chart = plot_group_trends(df, cfg)
    assert hasattr(chart, "to_dict"), "Altair chart が返されるべき"
    spec = chart.to_dict()
    assert "layer" in spec or "encoding" in spec or "spec" in spec


def test_plot_group_trends_empty_df():
    import altair as alt

    cfg = _default_cfg()
    chart = plot_group_trends(pd.DataFrame(), cfg)
    assert hasattr(chart, "to_dict"), "空 DataFrame でも Altair chart が返されるべき"


def test_plot_estimate_comparison_returns_altair_chart():
    import altair as alt

    cfg = _default_cfg()
    df = generate_panel_data(cfg)
    pp = pre_post_difference(df, cfg.true_effect)
    did = did_estimate(df, cfg.true_effect)
    chart = plot_estimate_comparison([pp, did])
    assert hasattr(chart, "to_dict"), "Altair chart が返されるべき"


def test_plot_estimate_comparison_nan_estimates_do_not_crash():
    """NaN 推定値が混在しても描画クラッシュしない。"""
    import altair as alt
    from src.estimation import EstimationResult

    nan_result = EstimationResult(
        name="未推定",
        estimate=float("nan"),
        se=float("nan"),
        ci_lower=float("nan"),
        ci_upper=float("nan"),
        true_effect=5.0,
        n_obs=0,
    )
    cfg = _default_cfg()
    df = generate_panel_data(cfg)
    did = did_estimate(df, cfg.true_effect)
    chart = plot_estimate_comparison([nan_result, did])
    assert hasattr(chart, "to_dict")
