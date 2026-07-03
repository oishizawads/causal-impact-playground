"""合成パネルデータの生成。

処置群と対照群、施策前後の期間を持つ個体レベルの合成データを生成する。
真の効果 (true_effect) は処置群の事後期間にのみ加算される。
trend_diff != 0 のとき平行トレンド仮定が崩れる。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DataConfig:
    """合成データ生成パラメータ。"""

    seed: int = 42
    n_pre: int = 5
    n_post: int = 5
    n_per_cell: int = 30
    baseline: float = 100.0
    trend: float = 1.0
    group_offset: float = 0.0
    trend_diff: float = 0.0
    true_effect: float = 5.0
    noise_sd: float = 2.0
    # 事前/事後の期間長。n_pre + n_post が全期間長。
    periods: tuple[int, ...] = field(default_factory=tuple)

    @property
    def n_periods(self) -> int:
        return self.n_pre + self.n_post


def validate_config(cfg: DataConfig) -> list[str]:
    """パラメータの入力値バリデーション。問題メッセージのリストを返す（空ならOK）。"""
    errors: list[str] = []
    if cfg.n_pre < 1:
        errors.append("施策前の期間数は 1 以上にしてください。")
    if cfg.n_post < 1:
        errors.append("施策後の期間数は 1 以上にしてください。")
    if cfg.n_per_cell < 1:
        errors.append("1セルあたりのサンプル数は 1 以上にしてください。")
    if cfg.noise_sd < 0:
        errors.append("ノイズ標準偏差は 0 以上にしてください。")
    if not np.isfinite(cfg.baseline):
        errors.append("ベースラインに有限な数値を指定してください。")
    if not np.isfinite(cfg.true_effect):
        errors.append("真の効果に有限な数値を指定してください。")
    if not np.isfinite(cfg.noise_sd):
        errors.append("ノイズ標準偏差に有限な数値を指定してください。")
    return errors


def generate_panel_data(cfg: DataConfig) -> pd.DataFrame:
    """DataConfig から個体レベルのパネルデータを生成する。

    モデル: y = baseline + trend*t + group_offset*treat + trend_diff*treat*t
            + true_effect*treat*post + N(0, noise_sd)
    """
    errors = validate_config(cfg)
    if errors:
        raise ValueError("; ".join(errors))

    rng = np.random.default_rng(cfg.seed)
    rows: list[dict[str, float]] = []
    t_index = np.arange(cfg.n_periods)
    post_mask = t_index >= cfg.n_pre

    for treat in (0, 1):
        for t in t_index:
            post = int(post_mask[t])
            mean = (
                cfg.baseline
                + cfg.trend * t
                + cfg.group_offset * treat
                + cfg.trend_diff * treat * t
                + cfg.true_effect * treat * post
            )
            sd = cfg.noise_sd if cfg.noise_sd > 0 else 0.0
            draws = rng.normal(loc=mean, scale=sd, size=cfg.n_per_cell)
            for y in draws:
                rows.append(
                    {
                        "y": float(y),
                        "treat": treat,
                        "time": int(t),
                        "post": post,
                        "group": "処置群" if treat == 1 else "対照群",
                        "period": "事後" if post else "事前",
                    }
                )

    return pd.DataFrame(rows)
