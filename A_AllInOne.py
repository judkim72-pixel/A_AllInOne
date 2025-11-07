
# A_AllInOne.py
# 하나의 페이지에서 A항목(최대 5개 데이터셋)을 탭 형태로 모두 확인/분석할 수 있는 통합 페이지
#
# 각 탭 공통 기능
# - CSV 업로드(Company 필수, Category 선택, 나머지 수치형 자동 인식)
# - 1~10 가정 스케일 → 0~100 재스케일 옵션
# - 가중치 기반 CompositeScore 계산 및 분위 등급(Tier) 산출
# - 분포/박스플롯/레이더 차트 + 자동 해설
# - 결과 CSV 다운로드
#
# 사용
# $ streamlit run A_AllInOne.py

import io
import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="A: All-in-One(5 datasets)", layout="wide")

# -------------------------------
# 공통 유틸
# -------------------------------
def detect_numeric_columns(df, exclude_cols):
    numeric_cols = []
    for c in df.columns:
        if c in exclude_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_cols.append(c)
    return numeric_cols

def to_0_100_scale(series, assumed_min=1.0, assumed_max=10.0):
    s = series.astype(float)
    return (s - assumed_min) / (assumed_max - assumed_min) * 100.0

def safe_quantile(series, q):
    try:
        return float(series.quantile(q))
    except Exception:
        return float('nan')

def summarize_distribution(values, title_label=""):
    s = pd.Series(values).dropna()
    if len(s) == 0:
        return "데이터가 부족하여 분포 해설을 생성할 수 없습니다."
    mean = s.mean()
    med = s.median()
    q1 = safe_quantile(s, 0.25)
    q3 = safe_quantile(s, 0.75)
    msg = f"{title_label} 평균은 {mean:.1f}점, 중앙값은 {med:.1f}점입니다. 사분위 범위(IQR)는 {q1:.1f}~{q3:.1f}입니다."
    if mean > med + 2:
        msg += " 평균이 중앙값보다 높아 고득점 쪽으로 꼬리가 긴 분포일 수 있습니다."
    elif med > mean + 2:
        msg += " 중앙값이 평균보다 높아 저득점 쪽으로 꼬리가 긴 분포일 수 있습니다."
    else:
        msg += " 평균과 중앙값이 비슷하여 대체로 대칭적인 분포로 보입니다."
    return msg

def make_radar(ax, metrics, values, baseline, title):
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    v = values.tolist() + values.tolist()[:1]
    b = baseline.tolist() + baseline.tolist()[:1]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=9)
    ax.set_rlabel_position(0)
    ax.set_yticks([20, 40, 60, 80])
    ax.set_yticklabels(["20", "40", "60", "80"], fontsize=8)
    ax.set_ylim(0, 100)

    ax.plot(angles, v, linewidth=2)
    ax.fill(angles, v, alpha=0.1)
    ax.plot(angles, b, linewidth=1, linestyle='--')
    ax.set_title(title, pad=10, fontsize=12)

def process_single_dataset(slot_name:str):
    st.markdown(f"### 데이터 업로드 – {slot_name}")
    uploaded = st.file_uploader(f"{slot_name} CSV 업로드 (필수: Company, 선택: Category)", type=["csv"], key=f"fu_{slot_name}")
    assumed_min = st.number_input(f"[{slot_name}] 지표 최소값(스케일 가정)", 0.0, 1000.0, 1.0, 0.5, key=f"min_{slot_name}")
    assumed_max = st.number_input(f"[{slot_name}] 지표 최대값(스케일 가정)", 1.0, 1000.0, 10.0, 0.5, key=f"max_{slot_name}")
    rescale = st.checkbox(f"[{slot_name}] 0~100 점수로 재스케일", value=True, key=f"rescale_{slot_name}")

    if uploaded is None:
        st.info("CSV를 업로드하면 분석 결과를 볼 수 있습니다.")
        return

    df = pd.read_csv(uploaded)
    if "Company" not in df.columns:
        st.error("필수 컬럼 'Company'가 없습니다.")
        return
    has_category = "Category" in df.columns
    exclude = ["Company", "Category"]
    metric_cols = detect_numeric_columns(df, exclude)
    if len(metric_cols) == 0:
        st.error("수치형 지표 컬럼이 감지되지 않았습니다.")
        return

    st.subheader("데이터 미리보기")
    st.dataframe(df.head(20), use_container_width=True)

    # 가중치
    st.markdown("#### 지표 가중치 설정")
    st.write("가중치는 합=1로 자동 정규화됩니다.")
    default_weight = 1.0 / len(metric_cols)
    weight_vals = {}
    for mc in metric_cols:
        weight_vals[mc] = st.number_input(f"[{slot_name}] {mc} 가중치", 0.0, 100.0, float(default_weight), 0.1, key=f"w_{slot_name}_{mc}")
    w_total = sum(weight_vals.values()) if sum(weight_vals.values()) > 0 else 1.0
    weights = {k: v / w_total for k, v in weight_vals.items()}

    # 점수 계산
    score_df = df.copy()
    for mc in metric_cols:
        if rescale:
            score_df[mc + "_scaled"] = to_0_100_scale(score_df[mc], assumed_min, assumed_max)
        else:
            score_df[mc + "_scaled"] = score_df[mc].astype(float)

    scaled_cols = [c for c in score_df.columns if c.endswith("_scaled") and c[:-7] in metric_cols]
    score_df["CompositeScore"] = 0.0
    for mc in metric_cols:
        score_df["CompositeScore"] += score_df[f"{mc}_scaled"] * weights[mc]

    s = score_df["CompositeScore"]
    q33, q66 = s.quantile(0.33), s.quantile(0.66)
    def label_tier(v):
        if v >= q66:
            return "Top"
        elif v >= q33:
            return "Mid"
        else:
            return "Low"
    score_df["Tier"] = score_df["CompositeScore"].apply(label_tier)

    # 결과 테이블 + 다운로드
    show_cols = ["Company"] + (["Category"] if has_category else []) + metric_cols + scaled_cols + ["CompositeScore", "Tier"]
    st.markdown("#### 결과 테이블")
    st.dataframe(score_df[show_cols].sort_values("CompositeScore", ascending=False), use_container_width=True)
    csv_buf = io.StringIO()
    score_df[show_cols].to_csv(csv_buf, index=False, encoding="utf-8-sig")
    st.download_button(f"[{slot_name}] 결과 CSV 다운로드", data=csv_buf.getvalue(), file_name=f"{slot_name}_results.csv", mime="text/csv")

    st.markdown("---")
    st.markdown("#### 분포 시각화 & 자동 해설")
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.hist(score_df["CompositeScore"], bins=12)
    ax1.set_title("Composite Score Distribution")
    ax1.set_xlabel("Score (0~100)")
    ax1.set_ylabel("Count")
    st.pyplot(fig1, use_container_width=False)
    st.caption(summarize_distribution(score_df["CompositeScore"], f"{slot_name} 종합점수"))

    st.markdown("#### 지표별 분포 (최대 6개)")
    max_charts = min(6, len(metric_cols))
    grid_cols = st.columns(max_charts)
    for i, mc in enumerate(metric_cols[:max_charts]):
        with grid_cols[i]:
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.hist(score_df[f"{mc}_scaled"], bins=10)
            ax.set_title(f"{mc} (scaled)")
            ax.set_xlabel("0~100")
            ax.set_ylabel("Count")
            st.pyplot(fig, use_container_width=True)
            st.caption(summarize_distribution(score_df[f"{mc}_scaled"], f"{mc}"))

    if has_category:
        st.markdown("#### 카테고리별 박스플롯 & 해설")
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        cats = list(score_df["Category"].dropna().unique())
        data_by_cat = [score_df.loc[score_df["Category"] == c, "CompositeScore"].dropna().values for c in cats]
        ax2.boxplot(data_by_cat, labels=cats, vert=True, showmeans=True)
        ax2.set_title("Composite Score by Category")
        ax2.set_ylabel("Score (0~100)")
        st.pyplot(fig2, use_container_width=False)
        medians = {c: np.nanmedian(score_df.loc[score_df["Category"] == c, "CompositeScore"]) for c in cats}
        best_cat = max(medians, key=medians.get) if len(medians) else None
        cat_msg = "카테고리 간 중앙값 차이를 통해 상대적 성과를 비교할 수 있습니다."
        if best_cat is not None:
            cat_msg += f" 현재 중앙값이 가장 높은 카테고리는 **{best_cat}** 입니다."
        st.caption(cat_msg)

    # 레이더 차트
    st.markdown("#### 레이더 차트 (기업 vs. 기준선)")
    colL, colR = st.columns([1,2])
    with colL:
        selected_company = st.selectbox(f"[{slot_name}] 기업 선택", score_df["Company"].tolist(), key=f"sel_{slot_name}")
        baseline_mode = st.radio(f"[{slot_name}] 기준선 선택", ["전체 평균", "카테고리 중앙값(있을 경우)", "전체 중앙값"], index=0, key=f"base_{slot_name}")

    with colR:
        sel_row = score_df.loc[score_df["Company"] == selected_company].iloc[0]
        values = pd.Series([sel_row[f"{mc}_scaled"] for mc in metric_cols], index=metric_cols)
        if baseline_mode == "전체 평균":
            baseline = score_df[[f"{mc}_scaled" for mc in metric_cols]].mean().values
            baseline = pd.Series(baseline, index=metric_cols)
            baseline_label = "전체 평균"
        elif baseline_mode == "카테고리 중앙값(있을 경우)" and has_category and pd.notna(sel_row.get("Category", np.nan)):
            cat = sel_row["Category"]
            baseline = score_df.loc[score_df["Category"] == cat, [f"{mc}_scaled" for mc in metric_cols]].median().values
            baseline = pd.Series(baseline, index=metric_cols)
            baseline_label = f"{cat} 중앙값"
        else:
            baseline = score_df[[f"{mc}_scaled" for mc in metric_cols]].median().values
            baseline = pd.Series(baseline, index=metric_cols)
            baseline_label = "전체 중앙값"

        figR = plt.figure(figsize=(6, 6))
        axR = plt.subplot(111, polar=True)
        make_radar(axR, metric_cols, values, baseline, f"{selected_company} vs. {baseline_label}")
        st.pyplot(figR, use_container_width=False)
        diffs = (values - baseline).sort_values(ascending=False)
        top_strengths = ", ".join([f"{k} (+{v:.1f})" for k, v in diffs.head(3).items()])
        top_weak = ", ".join([f"{k} ({v:.1f})" for k, v in diffs.tail(3).items()])
        st.caption(f"**요약:** 선택 기업은 {baseline_label} 대비 **강점**이 {top_strengths} 이며, **보완 필요** 영역은 {top_weak} 입니다.")

# -------------------------------
# 레이아웃
# -------------------------------
st.title("A 항목 All‑in‑One 대시보드 (최대 5개 데이터셋)")
st.caption("한 페이지에서 A1~A5(또는 5개 데이터셋)를 탭으로 전환하며 동일한 분석 파이프라인으로 확인합니다.")

tabs = st.tabs(["A1", "A2", "A3", "A4", "A5"])
slot_names = ["A1", "A2", "A3", "A4", "A5"]
for tab, name in zip(tabs, slot_names):
    with tab:
        process_single_dataset(name)

st.markdown("---")
st.markdown("##### 사용 팁")
st.write("""
- 각 탭은 완전히 독립적으로 동작합니다. 서로 다른 CSV를 올려 비교하거나 동일 CSV를 여러 탭에 올려 설정(가중치/스케일)을 달리하여 비교할 수 있습니다.
- 결과 테이블/분포/박스플롯/레이더 차트 하단의 캡션은 자동 해설을 제공합니다.
- 레이더 차트의 기준선(전체 평균/중앙값, 동일 카테고리 중앙값)을 바꾸며 상대 비교를 빠르게 확인하세요.
""")
