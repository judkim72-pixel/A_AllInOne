
# A_AllInOne.py
# All‑in‑One dashboard that reads a fixed Excel file (ux_100_dataset.xlsx) from repo root.
# Tabs (A1~A5) run the same analysis pipeline on chosen sheets.
#
# Run:
#   streamlit run A_AllInOne.py

from pathlib import Path
import io
import numpy as np
import pandas as pd
import streamlit as st

# Headless-safe Matplotlib import
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MPL_OK = True
    _MPL_ERR = None
except Exception as _e:
    _MPL_OK = False
    _MPL_ERR = _e

st.set_page_config(page_title="A: All‑in‑One (Excel Source)", layout="wide")

# ---- Environment info (helps debugging on Streamlit Cloud) ----
with st.sidebar:
    st.markdown("### Environment")
    try:
        import sys
        st.write("Python", sys.version.split()[0])
    except Exception:
        pass
    try:
        import streamlit as _st
        st.write("streamlit", _st.__version__)
    except Exception:
        pass
    try:
        st.write("pandas", pd.__version__)
    except Exception:
        pass
    try:
        import numpy as _np
        st.write("numpy", _np.__version__)
    except Exception:
        pass
    if not _MPL_OK:
        st.warning(f"Matplotlib import failed: {_MPL_ERR}")

# ---- Config ----
EXCEL_FILENAME = "ux_100_dataset.xlsx"
EXCEL_PATH = Path(__file__).resolve().parent / EXCEL_FILENAME

# ---- Utils ----
def detect_numeric_columns(df: pd.DataFrame, exclude_cols: list[str]) -> list[str]:
    cols = []
    for c in df.columns:
        if c in exclude_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols

def to_0_100_scale(series: pd.Series, assumed_min: float = 1.0, assumed_max: float = 10.0) -> pd.Series:
    s = series.astype(float)
    return (s - assumed_min) / (assumed_max - assumed_min) * 100.0

def safe_quantile(series: pd.Series, q: float) -> float:
    try:
        return float(series.quantile(q))
    except Exception:
        return float("nan")

def summarize_distribution(values: pd.Series, title_label: str = "") -> str:
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

def make_radar(ax, metrics, values, baseline, title: str):
    import numpy as _np
    angles = _np.linspace(0, 2*_np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    v = values.tolist() + values.tolist()[:1]
    b = baseline.tolist() + baseline.tolist()[:1]

    ax.set_theta_offset(_np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=9)
    ax.set_rlabel_position(0)
    ax.set_yticks([20, 40, 60, 80])
    ax.set_yticklabels(["20", "40", "60", "80"], fontsize=8)
    ax.set_ylim(0, 100)

    ax.plot(angles, v, linewidth=2)
    ax.fill(angles, v, alpha=0.1)
    ax.plot(angles, b, linewidth=1, linestyle="--")
    ax.set_title(title, pad=10, fontsize=12)

@st.cache_data(show_spinner=False)
def load_all_sheets(xlsx_path: Path) -> dict[str, pd.DataFrame]:
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Excel 파일을 찾지 못했습니다: {xlsx_path}")
    # engine openpyxl
    return pd.read_excel(xlsx_path, sheet_name=None, engine="openpyxl")

def process_dataset(slot_key: str, df: pd.DataFrame):
    # Options
    st.markdown(f"#### 분석 옵션 – {slot_key}")
    assumed_min = st.number_input(f"[{slot_key}] 지표 최소값(스케일 가정)", 0.0, 1000.0, 1.0, 0.5, key=f"min_{slot_key}")
    assumed_max = st.number_input(f"[{slot_key}] 지표 최대값(스케일 가정)", 1.0, 1000.0, 10.0, 0.5, key=f"max_{slot_key}")
    rescale = st.checkbox(f"[{slot_key}] 0~100 점수로 재스케일", value=True, key=f"rescale_{slot_key}")

    # Schema
    if "Company" not in df.columns:
        st.error("필수 컬럼 'Company'가 없습니다.")
        return
    has_category = "Category" in df.columns
    metric_cols = detect_numeric_columns(df, exclude_cols=["Company", "Category"])
    if len(metric_cols) == 0:
        st.error("수치형 지표 컬럼이 감지되지 않았습니다.")
        return

    # Preview
    st.markdown("##### 데이터 미리보기")
    st.dataframe(df.head(20), use_container_width=True)

    # Weights
    st.markdown("##### 지표 가중치 설정")
    st.write("가중치는 합=1로 자동 정규화됩니다.")
    default_weight = 1.0 / len(metric_cols)
    weight_vals = {}
    for mc in metric_cols:
        weight_vals[mc] = st.number_input(f"[{slot_key}] {mc} 가중치", 0.0, 100.0, float(default_weight), 0.1, key=f"w_{slot_key}_{mc}")
    w_total = sum(weight_vals.values()) if sum(weight_vals.values()) > 0 else 1.0
    weights = {k: v / w_total for k, v in weight_vals.items()}

    # Scores
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

    # Tiers
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

    # Table + Download
    show_cols = ["Company"] + (["Category"] if has_category else []) + metric_cols + scaled_cols + ["CompositeScore", "Tier"]
    st.markdown("##### 결과 테이블")
    st.dataframe(score_df[show_cols].sort_values("CompositeScore", ascending=False), use_container_width=True)
    csv_buf = io.StringIO()
    score_df[show_cols].to_csv(csv_buf, index=False, encoding="utf-8-sig")
    st.download_button(f"[{slot_key}] 결과 CSV 다운로드", data=csv_buf.getvalue(), file_name=f"{slot_key}_results.csv", mime="text/csv")

    # Plots
    if not _MPL_OK:
        st.warning(f"Matplotlib 불러오기 실패: {_MPL_ERR}. requirements.txt에 matplotlib를 추가해 주세요.")
        return

    st.markdown("---")
    st.markdown("##### 분포 시각화 & 자동 해설")
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.hist(score_df["CompositeScore"], bins=12)
    ax1.set_title("Composite Score Distribution")
    ax1.set_xlabel("Score (0~100)")
    ax1.set_ylabel("Count")
    st.pyplot(fig1, use_container_width=False)
    st.caption(summarize_distribution(score_df["CompositeScore"], f"{slot_key} 종합점수"))

    st.markdown("##### 지표별 분포 (최대 6개)")
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
        st.markdown("##### 카테고리별 박스플롯 & 해설")
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

    # Radar
    st.markdown("##### 레이더 차트 (기업 vs. 기준선)")
    colL, colR = st.columns([1, 2])
    with colL:
        selected_company = st.selectbox(f"[{slot_key}] 기업 선택", score_df["Company"].tolist(), key=f"sel_{slot_key}")
        baseline_mode = st.radio(f"[{slot_key}] 기준선 선택", ["전체 평균", "카테고리 중앙값(있을 경우)", "전체 중앙값"], index=0, key=f"base_{slot_key}")

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


# ---- Layout ----
st.title("A 항목 All-in-One (Excel 고정 데이터 소스)")
st.caption("리포지토리 루트의 ux_100_dataset.xlsx를 기반으로 A1~A5 탭 분석을 제공합니다.")

# Load Excel
try:
    sheets = load_all_sheets(EXCEL_PATH)
except FileNotFoundError as e:
    st.error(f"{e}\n\n*리포지토리 루트에 '{EXCEL_FILENAME}' 파일을 넣어 주세요.*")
    st.stop()
except Exception as e:
    st.error(f"Excel 로드 중 오류: {e}")
    st.stop()

sheet_names = list(sheets.keys())

# --- Sidebar: per-tab sheet & column mapping ---
with st.sidebar:
    st.header("시트 & 컬럼 매핑")

    # Default sheet mapping
    default_map = {name: name if name in sheet_names else (sheet_names[0] if sheet_names else None) for name in ["A1","A2","A3","A4","A5"]}

    chosen_sheet = {}
    company_col_map = {}
    category_col_map = {}

    # Known synonyms
    company_synonyms = ["Company", "company", "기업명", "기업", "회사", "회사명", "Name", "name"]
    category_synonyms = ["Category", "category", "카테고리", "분류", "업종", "그룹", "Group", "group", "Sector", "sector"]

    for key in ["A1","A2","A3","A4","A5"]:
        st.markdown(f"**탭 {key}**")
        # Sheet select
        if sheet_names:
            idx = sheet_names.index(default_map[key]) if default_map[key] in sheet_names else 0
            chosen_sheet[key] = st.selectbox(f"{key} 탭 시트", sheet_names, index=idx, key=f"map_sheet_{key}")
        else:
            chosen_sheet[key] = None

        # Infer candidate columns
        df_tmp = sheets[chosen_sheet[key]].head(1) if chosen_sheet[key] else None
        cols = list(df_tmp.columns) if df_tmp is not None else []

        # Company column
        comp_default = None
        for cand in company_synonyms:
            if cand in cols:
                comp_default = cand
                break
        company_col_map[key] = st.selectbox(f"{key} - Company 컬럼", options=cols, index=(cols.index(comp_default) if comp_default in cols else 0) if cols else 0, key=f"map_company_{key}") if cols else None

        # Category column (optional)
        cat_default = None
        for cand in category_synonyms:
            if cand in cols:
                cat_default = cand
                break
        category_col_map[key] = st.selectbox(f"{key} - Category 컬럼(선택)", options=["(없음)"] + cols, index=( (["(없음)"]+cols).index(cat_default) if cat_default in cols else 0), key=f"map_category_{key}") if cols else "(없음)"
        st.divider()

def coerce_numeric(df, exclude_cols):
    df_out = df.copy()
    numeric_cols = []
    for c in df_out.columns:
        if c in exclude_cols:
            continue
        # try numeric coercion
        coerced = pd.to_numeric(df_out[c], errors="coerce")
        # consider numeric if at least one non-null after coercion
        if coerced.notna().sum() > 0:
            df_out[c] = coerced
            numeric_cols.append(c)
    return df_out, numeric_cols

tabs = st.tabs(["A1", "A2", "A3", "A4", "A5"])
for tab_key, tab in zip(["A1","A2","A3","A4","A5"], tabs):
    with tab:
        if chosen_sheet.get(tab_key) is None:
            st.warning("선택할 시트가 없습니다.")
            continue

        df_tab_raw = sheets[chosen_sheet[tab_key]].copy()

        if df_tab_raw.empty:
            st.error("시트에 데이터가 없습니다.")
            continue

        # Rename columns to canonical names based on mapping
        comp_col = company_col_map.get(tab_key)
        cat_col = category_col_map.get(tab_key)
        if comp_col is None:
            st.error("Company 컬럼을 선택해 주세요.")
            continue

        rename_map = {comp_col: "Company"}
        if cat_col and cat_col != "(없음)":
            rename_map[cat_col] = "Category"
        df_tab = df_tab_raw.rename(columns=rename_map)

        if "Company" not in df_tab.columns:
            st.error("필수 컬럼 'Company'가 없습니다. 사이드바에서 매핑을 확인해 주세요.")
            continue

        # Coerce numerics & let user choose metric columns
        df_coerced, numeric_candidates = coerce_numeric(df_tab, exclude_cols=["Company", "Category"])
        st.markdown("#### 지표 컬럼 선택")
        if len(numeric_candidates) == 0:
            st.error("수치형으로 판별되는 지표 컬럼이 없습니다. 엑셀 데이터 타입을 확인해 주세요.")
            continue
        selected_metrics = st.multiselect("분석에 사용할 지표 컬럼", options=numeric_candidates, default=numeric_candidates, key=f"metric_sel_{tab_key}")
        if len(selected_metrics) == 0:
            st.warning("선택된 지표가 없습니다. 최소 1개 이상 선택해 주세요.")
            continue

        # Keep only selected metrics + Company/Category
        keep_cols = ["Company"] + (["Category"] if "Category" in df_coerced.columns else []) + selected_metrics
        df_ready = df_coerced[keep_cols].copy()

        st.markdown(f"### 탭: {tab_key}  |  시트: **{chosen_sheet[tab_key]}**")
        process_dataset(tab_key, df_ready)
