import os
import json
import re
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Optional: OpenAI (only used if key is provided)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# -----------------------------
# App config
# -----------------------------
st.set_page_config(page_title="환경 데이터 분석 (AI 포함)", layout="wide")

st.title("🌍 환경 데이터 분석 대시보드 (Streamlit + OpenAI API)")
st.caption("업로드 데이터 자동 EDA + 통계 분석 + (선택) AI 구조화 인사이트/보고서 생성")

# -----------------------------
# Utilities
# -----------------------------
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(\+?\d{1,3}[-\s]?)?(\d{2,4}[-\s]?\d{3,4}[-\s]?\d{4})")

def mask_pii_text(s: str) -> str:
    s = EMAIL_RE.sub("[EMAIL]", s)
    s = PHONE_RE.sub("[PHONE]", s)
    return s

def mask_pii_df(df: pd.DataFrame, max_cells: int = 20000) -> pd.DataFrame:
    out = df.copy()
    # Cost/latency guard
    if out.size > max_cells:
        out = out.head(max(50, min(500, len(out))))
    obj_cols = [c for c in out.columns if out[c].dtype == "object"]
    for c in obj_cols:
        out[c] = out[c].astype(str).map(mask_pii_text)
    return out

def safe_get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    if key in st.secrets:
        return st.secrets.get(key, default)
    return os.getenv(key, default)

@st.cache_resource
def get_openai_client(api_key: str):
    if OpenAI is None:
        raise RuntimeError("openai 패키지를 불러오지 못했습니다. requirements.txt 설치를 확인하세요.")
    return OpenAI(api_key=api_key)

@st.cache_data
def load_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)

@st.cache_data
def load_excel(file) -> pd.DataFrame:
    return pd.read_excel(file)

def infer_datetime(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Try to create a datetime column from common patterns.
    Returns (df, datetime_col_name or None).
    """
    out = df.copy()
    # If already has a datetime-like column
    for c in out.columns:
        if np.issubdtype(out[c].dtype, np.datetime64):
            return out, c

    # Common patterns: Year/Month/Day
    cols = {c.lower(): c for c in out.columns}
    if {"year", "month", "day"}.issubset(cols.keys()):
        y, m, d = cols["year"], cols["month"], cols["day"]
        out["date"] = pd.to_datetime(dict(year=out[y], month=out[m], day=out[d]), errors="coerce")
        return out, "date"

    # Single date column named like date/time
    for key in ["date", "datetime", "time", "timestamp"]:
        if key in cols:
            c = cols[key]
            out[c] = pd.to_datetime(out[c], errors="coerce")
            return out, c

    return out, None

def numeric_cols(df: pd.DataFrame):
    return df.select_dtypes(include=[np.number]).columns.tolist()

def make_eda(df: pd.DataFrame) -> Dict[str, Any]:
    eda = {}
    eda["shape"] = list(df.shape)
    eda["columns"] = []
    for c in df.columns:
        eda["columns"].append({
            "name": c,
            "dtype": str(df[c].dtype),
            "missing": int(df[c].isna().sum()),
            "n_unique": int(df[c].nunique(dropna=True))
        })
    num = numeric_cols(df)
    if num:
        desc = df[num].describe().T
        eda["numeric_describe"] = desc[["count","mean","std","min","25%","50%","75%","max"]].replace({np.nan: None}).to_dict()
    else:
        eda["numeric_describe"] = {}

    cat = [c for c in df.columns if df[c].dtype == "object"]
    top = {}
    for c in cat[:30]:
        vc = df[c].astype(str).value_counts(dropna=False).head(12)
        top[c] = [{"value": str(i), "count": int(vc[i])} for i in vc.index]
    eda["top_categories"] = top
    return eda

def plot_time_series(df: pd.DataFrame, date_col: str, y_col: str, group_col: Optional[str] = None):
    fig, ax = plt.subplots()
    if group_col and group_col in df.columns:
        for g, gdf in df.sort_values(date_col).groupby(group_col):
            ax.plot(gdf[date_col], gdf[y_col], label=str(g))
        ax.legend()
    else:
        d = df.sort_values(date_col)
        ax.plot(d[date_col], d[y_col])
    ax.set_xlabel(date_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"{y_col} over time")
    st.pyplot(fig, clear_figure=True)

def plot_monthly_climatology(df: pd.DataFrame, date_col: str, y_col: str, group_col: Optional[str] = None):
    d = df.dropna(subset=[date_col, y_col]).copy()
    d["month"] = pd.to_datetime(d[date_col]).dt.month
    fig, ax = plt.subplots()
    if group_col and group_col in d.columns:
        for g, gdf in d.groupby(group_col):
            m = gdf.groupby("month")[y_col].mean()
            ax.plot(m.index, m.values, label=str(g))
        ax.legend()
    else:
        m = d.groupby("month")[y_col].mean()
        ax.plot(m.index, m.values)
    ax.set_xlabel("month")
    ax.set_ylabel(f"mean({y_col})")
    ax.set_title("Monthly climatology (mean by month)")
    st.pyplot(fig, clear_figure=True)

def compute_anomaly(df: pd.DataFrame, date_col: str, y_col: str, group_col: Optional[str] = None) -> pd.DataFrame:
    d = df.dropna(subset=[date_col, y_col]).copy()
    d["month"] = pd.to_datetime(d[date_col]).dt.month
    if group_col and group_col in d.columns:
        clim = d.groupby([group_col, "month"])[y_col].mean().rename("clim").reset_index()
        out = d.merge(clim, on=[group_col, "month"], how="left")
        out["anomaly"] = out[y_col] - out["clim"]
    else:
        clim = d.groupby("month")[y_col].mean().rename("clim").reset_index()
        out = d.merge(clim, on="month", how="left")
        out["anomaly"] = out[y_col] - out["clim"]
    return out

# -----------------------------
# OpenAI Structured Output
# -----------------------------
INSIGHT_SCHEMA = {
  "name": "environment_analysis_report",
  "schema": {
    "type": "object",
    "additionalProperties": False,
    "properties": {
      "one_line_summary": {"type": "string"},
      "key_findings": {
        "type": "array",
        "items": {
          "type": "object",
          "additionalProperties": False,
          "properties": {
            "title": {"type": "string"},
            "evidence": {"type": "string"},
            "impact": {"type": "string"},
            "next_step": {"type": "string"}
          },
          "required": ["title", "evidence", "impact", "next_step"]
        }
      },
      "data_quality_warnings": {"type": "array", "items": {"type": "string"}},
      "statistical_notes": {"type": "array", "items": {"type": "string"}},
      "recommended_models": {
        "type": "array",
        "items": {
          "type": "object",
          "additionalProperties": False,
          "properties": {
            "model": {"type": "string"},
            "why": {"type": "string"},
            "how": {"type": "string"}
          },
          "required": ["model", "why", "how"]
        }
      },
      "executive_report_md": {"type": "string"}
    },
    "required": ["one_line_summary", "key_findings", "data_quality_warnings", "statistical_notes", "recommended_models", "executive_report_md"]
  }
}

def ai_report(
    df: pd.DataFrame,
    eda: Dict[str, Any],
    domain_context: str,
    date_col: Optional[str],
    y_col: Optional[str],
    group_col: Optional[str],
    model_name: str,
    api_key: str,
    user_requirements: str
) -> Dict[str, Any]:
    """
    Sends ONLY masked sample rows + summary stats to the model.
    """
    masked = mask_pii_df(df)
    sample = masked.sample(min(40, len(masked)), random_state=42).to_dict(orient="records")

    payload = {
        "domain_context": domain_context,
        "user_requirements": user_requirements,
        "dataset_shape": eda.get("shape"),
        "columns": eda.get("columns"),
        "numeric_describe": eda.get("numeric_describe"),
        "top_categories": eda.get("top_categories"),
        "selected_datetime_col": date_col,
        "selected_target_col": y_col,
        "selected_group_col": group_col,
        "masked_sample_rows": sample
    }

    system = (
        "당신은 환경 데이터(기후/해양/대기/설문 포함) 분석을 총괄하는 수석 데이터 과학자입니다. "
        "사용자가 제공한 user_requirements를 최우선으로 준수하세요. "
        "과장 금지. 관측/표본/측정의 한계를 명확히 지적하고, 통계적 함정(상관=인과, 계절성, 자기상관, 이상치, 결측, 표본 편향)을 반드시 언급하세요. "
        "출력은 반드시 주어진 JSON Schema를 만족해야 합니다. "
        "executive_report_md는 1~2페이지 분량의 마크다운 보고서로 작성하세요(제목/요약/핵심 결과/권고/한계)."
    )

    client = get_openai_client(api_key)

    # 사용자 요구 사항 우선 준수
 (openai>=1.40 권장)
    resp = client.responses.create(
        model=model_name,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}
        ],
        text={
            "format": {
                "type": "json_schema",
                "json_schema": INSIGHT_SCHEMA
            }
        }
    )

    # Prefer output_text when available
    out_text = getattr(resp, "output_text", None)
    if not out_text:
        # Fallback: try to find text in resp.model_dump()
        dump = resp.model_dump()
        out_text = json.dumps(dump, ensure_ascii=False)

    return json.loads(out_text)

# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    st.header("⚙️ 설정")
    domain_context = st.text_area(
        "도메인 컨텍스트(분석 목적/배경)",
        value="해빙(Sea Ice) 또는 환경 관측 데이터의 장기 추세/계절성/변동성 분석",
        height=110
    )

    
    st.subheader("요구 사항(분석/보고서 지시)")
    user_requirements = st.text_area(
        "요구 사항을 자유롭게 입력하세요 (예: 반드시 포함할 지표/표/그래프/해석 관점/톤)",
        value=(
            "- 결과는 정책 제언 중심으로\n"
            "- 표본 편향/결측 처리/계절성/자기상관을 반드시 언급\n"
            "- 핵심 그래프 3개(시계열, 월별 climatology, anomaly) 해석 포함\n"
            "- 결론은 5줄 이내 요약 + 다음 액션 3개"
        ),
        height=170
    )
st.subheader("OpenAI (선택)")
    default_model = safe_get_secret("OPENAI_MODEL", "gpt-4.1-mini")
    model_name = st.text_input("모델", value=default_model)
    api_key = safe_get_secret("OPENAI_API_KEY", "")
    api_key = st.text_input("OPENAI_API_KEY", value=api_key, type="password", help="Streamlit secrets 또는 환경변수로 설정 권장")
    enable_ai = st.toggle("AI 인사이트 사용", value=bool(api_key))

    st.divider()
    st.subheader("데이터")
    uploaded = st.file_uploader("CSV/Excel 업로드", type=["csv", "xlsx", "xls"])
    use_sample = st.checkbox("샘플 데이터(북극/남극 해빙) 사용", value=(uploaded is None))

# -----------------------------
# Load data
# -----------------------------
df = None
if uploaded is not None:
    name = uploaded.name.lower()
    if name.endswith(".csv"):
        df = load_csv(uploaded)
    else:
        df = load_excel(uploaded)
elif use_sample:
    # packaged sample (seaice.csv) should be next to this file when deployed
    sample_path = os.path.join(os.path.dirname(__file__), "seaice.csv")
    if os.path.exists(sample_path):
        df = pd.read_csv(sample_path)
    else:
        st.warning("샘플 seaice.csv가 앱 폴더에 없습니다. 파일 업로드를 사용하세요.")

if df is None:
    st.stop()

# Infer datetime
df, dt_col = infer_datetime(df)

# -----------------------------
# Basic EDA
# -----------------------------
st.subheader("1) 데이터 요약 (EDA)")
eda = make_eda(df)

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("행", eda["shape"][0])
with c2:
    st.metric("열", eda["shape"][1])
with c3:
    miss_total = sum(x["missing"] for x in eda["columns"])
    st.metric("결측치 총합", miss_total)

with st.expander("열 메타정보(결측/유형/고유값)"):
    st.dataframe(pd.DataFrame(eda["columns"]).sort_values("missing", ascending=False), use_container_width=True)

if st.checkbox("데이터 미리보기", value=True):
    st.dataframe(df.head(50), use_container_width=True)

# -----------------------------
# Analysis controls
# -----------------------------
st.subheader("2) 분석 설정")

num = numeric_cols(df)
all_cols = df.columns.tolist()

colA, colB, colC = st.columns([1,1,1])
with colA:
    date_col = st.selectbox("시간 컬럼", options=[None] + all_cols, index=(1 if dt_col in all_cols else 0))
with colB:
    y_col = st.selectbox("분석 대상(수치형)", options=[None] + num, index=(1 if len(num) else 0))
with colC:
    possible_groups = [c for c in all_cols if df[c].dtype == "object"][:30]
    group_col = st.selectbox("그룹 컬럼(선택)", options=[None] + possible_groups, index=(1 if "hemisphere" in possible_groups else 0))

dff = df.copy()
if date_col and date_col in dff.columns:
    dff = dff.dropna(subset=[date_col])
    dff[date_col] = pd.to_datetime(dff[date_col], errors="coerce")
    dff = dff.dropna(subset=[date_col])

    min_d, max_d = dff[date_col].min(), dff[date_col].max()
    rng = st.slider("기간 필터", min_value=min_d.to_pydatetime(), max_value=max_d.to_pydatetime(),
                    value=(min_d.to_pydatetime(), max_d.to_pydatetime()))
    dff = dff[(dff[date_col] >= pd.Timestamp(rng[0])) & (dff[date_col] <= pd.Timestamp(rng[1]))]

if group_col and group_col in dff.columns:
    groups = sorted(dff[group_col].dropna().astype(str).unique().tolist())
    picked = st.multiselect("그룹 선택", options=groups, default=groups[:min(len(groups), 6)])
    if picked:
        dff = dff[dff[group_col].astype(str).isin(picked)]

# -----------------------------
# Core plots
# -----------------------------
st.subheader("3) 시각화 & 핵심 통계")

if date_col and y_col:
    plot_time_series(dff, date_col, y_col, group_col=group_col)
    plot_monthly_climatology(dff, date_col, y_col, group_col=group_col)

    st.markdown("**월별 기준선(climatology) 대비 이상(anomaly)**")
    anom = compute_anomaly(dff, date_col, y_col, group_col=group_col)

    fig, ax = plt.subplots()
    if group_col and group_col in anom.columns:
        for g, gdf in anom.sort_values(date_col).groupby(group_col):
            ax.plot(gdf[date_col], gdf["anomaly"], label=str(g))
        ax.legend()
    else:
        ad = anom.sort_values(date_col)
        ax.plot(ad[date_col], ad["anomaly"])
    ax.axhline(0)
    ax.set_xlabel(date_col)
    ax.set_ylabel("anomaly")
    ax.set_title("Anomaly over time (de-seasonalized)")
    st.pyplot(fig, clear_figure=True)

    # Trend (simple) - annual aggregation helps reduce autocorrelation/seasonality
    st.markdown("**연도별 집계 + 단순 추세(선형회귀) 참고**")
    anom["year"] = pd.to_datetime(anom[date_col]).dt.year
    if group_col and group_col in anom.columns:
        rows = []
        for g, gdf in anom.groupby(group_col):
            annual = gdf.groupby("year")[y_col].mean().dropna()
            if len(annual) >= 5:
                x = annual.index.values.astype(float)
                y = annual.values.astype(float)
                # simple OLS
                b1, b0 = np.polyfit(x, y, 1)
                rows.append({"group": str(g), "years": int(len(annual)), "slope_per_year": float(b1), "intercept": float(b0)})
        if rows:
            st.dataframe(pd.DataFrame(rows).sort_values("slope_per_year"), use_container_width=True)
        else:
            st.info("연도별 집계 후 추세 계산에 필요한 표본이 부족합니다.")
    else:
        annual = anom.groupby("year")[y_col].mean().dropna()
        if len(annual) >= 5:
            x = annual.index.values.astype(float)
            y = annual.values.astype(float)
            b1, b0 = np.polyfit(x, y, 1)

            fig, ax = plt.subplots()
            ax.plot(annual.index, annual.values)
            ax.plot(annual.index, b1 * annual.index + b0)
            ax.set_xlabel("year")
            ax.set_ylabel(f"annual mean({y_col})")
            ax.set_title("Annual mean + linear trend (reference)")
            st.pyplot(fig, clear_figure=True)

            st.write({"years": int(len(annual)), "slope_per_year": float(b1)})
        else:
            st.info("연도별 집계 후 추세 계산에 필요한 표본이 부족합니다.")

else:
    st.info("시간 컬럼과 수치형 분석 대상을 선택하면 시계열/계절성/이상(anomaly) 분석이 활성화됩니다.")

# -----------------------------
# AI report
# -----------------------------
st.subheader("4) AI 인사이트/보고서 (선택)")

if not enable_ai:
    st.info("왼쪽에서 OPENAI_API_KEY를 설정하고 'AI 인사이트 사용'을 켜면 활성화됩니다.")
else:
    if st.button("🧠 AI 분석 보고서 생성 (구조화 출력)"):
        if not api_key:
            st.error("OPENAI_API_KEY가 필요합니다.")
        else:
            with st.spinner("AI 보고서 생성 중..."):
                report = ai_report(
                    df=dff,
                    eda=eda,
                    domain_context=domain_context,
                    date_col=date_col,
                    y_col=y_col,
                    group_col=group_col,
                    model_name=model_name,
                    api_key=api_key,
                    user_requirements=user_requirements
                )

            st.success("완료!")

            st.markdown("### 한 줄 요약")
            st.write(report["one_line_summary"])

            st.markdown("### 핵심 결과")
            for i, f in enumerate(report["key_findings"], 1):
                with st.expander(f"{i}. {f['title']}"):
                    st.markdown(f"**근거**: {f['evidence']}")
                    st.markdown(f"**영향/의미**: {f['impact']}")
                    st.markdown(f"**다음 단계**: {f['next_step']}")

            st.markdown("### 데이터 품질 경고")
            st.write(report["data_quality_warnings"])

            st.markdown("### 통계적 유의사항")
            st.write(report["statistical_notes"])

            st.markdown("### 추천 모델/분석 프레임")
            st.write(report["recommended_models"])

            st.markdown("### Executive Report (Markdown)")
            st.markdown(report["executive_report_md"])

            st.download_button(
                "📥 AI 보고서(JSON) 다운로드",
                data=json.dumps(report, ensure_ascii=False, indent=2),
                file_name="ai_environment_report.json",
                mime="application/json"
            )

st.caption("Tip: 배포 환경에서는 secrets 관리(OPENAI_API_KEY)와 데이터 보안(PII 마스킹/샘플링)을 꼭 확인하세요.")
