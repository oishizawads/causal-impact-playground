"""因果推論 Playground の推定ロジック。

UI (Streamlit) から独立した純粋関数としてデータ生成・推定を提供する。
"""
from __future__ import annotations

from .data import DataConfig, generate_panel_data, validate_config
from .estimation import EstimationResult, did_estimate, pre_post_difference

__all__ = [
    "DataConfig",
    "generate_panel_data",
    "validate_config",
    "EstimationResult",
    "did_estimate",
    "pre_post_difference",
]
