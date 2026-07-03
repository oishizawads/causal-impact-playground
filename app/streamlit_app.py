"""Causal Impact Playground の Streamlit エントリポイント。

UIのみを担い、推定ロジックは src/ に委譲する。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import DataConfig, did_estimate, generate_panel_data, pre_post_difference
from src.data import validate_config
from src.estimation import parallel_trends_check
from src.viz import plot_estimate_comparison, plot_group_trends

st.set_page_config(
    page_title="Causal Impact Playground",
    page_icon=":bar_chart:",
    layout="centered",
)

from src.brand import apply_brand, hero
apply_brand(st)
hero(st, "Causal Inference", "Causal Impact Playground", "施策前後比較・A/Bテスト・DID の違いを合成データで比較し、因果推論と実験設計を体験します。")


@st.cache_data(show_spinner=False)
def _build_dataset(cfg_tuple: tuple) -> pd.DataFrame:
    """キャッシュ付きデータ生成。引数はハッシュ可能なtupleに変換して渡す。"""
    cfg = DataConfig(*cfg_tuple)
    return generate_panel_data(cfg)


@st.cache_data(show_spinner=False)
def _build_results(cfg_tuple: tuple) -> dict:
    """推定結果一式をキャッシュ付きで構築。入力変化で再計算される。"""
    cfg = DataConfig(*cfg_tuple)
    df = generate_panel_data(cfg)
    pp = pre_post_difference(df, cfg.true_effect)
    did = did_estimate(df, cfg.true_effect)
    slopes = parallel_trends_check(df)
    return {
        "df": df,
        "pre_post": pp,
        "did": did,
        "slopes": slopes,
        "true_effect": cfg.true_effect,
    }


with st.sidebar:
    st.header("データ生成パラメータ")
    seed = st.number_input("乱数シード（再現性）", value=42, step=1)
    n_pre = st.number_input("施策前の期間数", min_value=1, value=5, step=1)
    n_post = st.number_input("施策後の期間数", min_value=1, value=5, step=1)
    n_per_cell = st.number_input("1セルあたりサンプル数", min_value=1, value=30, step=1)
    baseline = st.number_input("ベースライン（y切片）", value=100.0, step=1.0)
    trend = st.number_input("共通トレンド（期ごとの増分）", value=1.0, step=0.5)
    group_offset = st.number_input("群のレベル差（group_offset）", value=0.0, step=1.0)
    trend_diff = st.number_input(
        "群別トレンド差（0=平行トレンド成立）",
        value=0.0,
        step=0.5,
        help="0 以外にすると平行トレンド仮定が崩れ、DID にバイアスが入ります。",
    )
    true_effect = st.number_input("真の施策効果", value=5.0, step=1.0)
    noise_sd = st.number_input("ノイズ標準偏差", min_value=0.0, value=2.0, step=0.5)
    alpha = st.slider("信頼区間の有意水準 (1 - alpha)", min_value=0.01, max_value=0.20, value=0.05, step=0.01)

cfg = DataConfig(
    seed=int(seed),
    n_pre=int(n_pre),
    n_post=int(n_post),
    n_per_cell=int(n_per_cell),
    baseline=float(baseline),
    trend=float(trend),
    group_offset=float(group_offset),
    trend_diff=float(trend_diff),
    true_effect=float(true_effect),
    noise_sd=float(noise_sd),
)

# バリデーション表示（極端値でも落ちない）
errors = validate_config(cfg)
if errors:
    for msg in errors:
        st.error(msg)
    st.stop()

cfg_tuple = (
    cfg.seed, cfg.n_pre, cfg.n_post, cfg.n_per_cell, cfg.baseline,
    cfg.trend, cfg.group_offset, cfg.trend_diff, cfg.true_effect, cfg.noise_sd,
)

bundle = _build_results(cfg_tuple)
df: pd.DataFrame = bundle["df"]
pp = bundle["pre_post"]
did = bundle["did"]
slopes = bundle["slopes"]

# --- 群別トレンドグラフ ---
st.subheader("群別トレンドグラフ")
st.pyplot(plot_group_trends(df, cfg), use_container_width=True)

# --- 平行トレンドの診断 ---
st.subheader("平行トレンドの目安（事前期間）")
col_a, col_b, col_c = st.columns(3)
col_a.metric("対照群の事前傾き", f"{slopes['control_slope']:.2f}")
col_b.metric("処置群の事前傾き", f"{slopes['treat_slope']:.2f}")
slope_gap = abs(slopes["treat_slope"] - slopes["control_slope"])
col_c.metric("傾きの差", f"{slope_gap:.2f}")

if abs(cfg.trend_diff) > 1e-9:
    st.warning(
        "群別トレンド差が 0 以外です。平行トレンド仮定が崩れているため、"
        "DID 推定量にもバイアスが入ります。下の推定値で bias を確認してください。"
    )
else:
    st.success("群別トレンド差 = 0。平行トレンド仮定が成立する設定です。")

# --- 推定結果 ---
st.subheader("推定結果（信頼区間付き）")
st.pyplot(plot_estimate_comparison([pp, did]), use_container_width=True)

rows = [
    {
        "手法": r.name,
        "推定値": r.estimate,
        "標準誤差": r.se,
        "下側CI": r.ci_lower,
        "上側CI": r.ci_upper,
        "真の効果": r.true_effect,
        "バイアス": r.bias,
        "CIが真値を包含": r.covers_true,
        "サンプル数": r.n_obs,
    }
    for r in (pp, did)
]
st.dataframe(pd.DataFrame(rows).set_index("手法"), use_container_width=True)

# --- 解釈メモ ---
st.subheader("解釈メモ")
st.markdown(
    f"""
- **単純前後差**: 処置群の事後-事前の差。共通トレンドが {cfg.trend:g} のため、
  真の効果 {cfg.true_effect:g} に対してトレンド分だけ上方バイアスが入りやすい典型例。
- **DID**: 対照群でトレンドを差し引くことで、平行トレンドが成立すれば真の効果に近づく。
- 現在の設定: トレンド差 = {cfg.trend_diff:g}、ノイズSD = {cfg.noise_sd:g}、
  セルサイズ = {cfg.n_per_cell}。
"""
)

# --- 注意文（過剰解釈の防止） ---
st.subheader(":warning: 注意")
st.info(
    "これは合成データの実験であり、実在の因果効果を示すものではありません。"
    "DID は平行トレンド仮定の下で成立する手法であり、実際の施策評価では"
    "事前トレンドの検査・共変量バランス・介入タイミングの精査が不可欠です。"
    "ノイズやサンプルサイズによっては信頼区間が真値を外れることもあります。"
)
