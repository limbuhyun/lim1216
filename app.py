import os
import json
import re
from typing import Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# OpenAI is optional (app still runs without it)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="요구사항 기반 AI 환경분석", layout="wide")
st.title("🌍 요구사항 기반 AI 환경 데이터 분석 (Streamlit)")
st.caption("요구사항을 입력 → 버튼 실행 → 해당 요구사항을 반영한 분석/보고서 생성")


# -----------------------------
# Helpers: secrets/env
# -----------------------------
def get_setting(key: str, default: str = "") -> str:
    if key in st.secrets:
        return str(st.secrets.get(key, default))
    return os.getenv(key, default)


# -----------------------------
# Helpers: PII masking
# -----------------------------
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(\+?\d{1,3}[-\s]?)?(\d{2,4}[-\s]?\d{3,4}[-\s]?\d{4})")

def mask_pii_text(s: str) -> str:
    s = EMAIL_RE.sub("[EMAIL]", s)
    s = PHONE_RE.sub("[PHONE]", s)
    return s

def mask_pii_df(df: pd.DataFrame, max_rows: int = 200) -> pd.DataFrame:
    out = df.copy()
    # keep small sample to reduce cost and protect privacy
    out = out.head(max_rows)
    obj_cols = out.select_dtypes(include="object").columns.tolist()
    for c in obj_cols:
        out[c] = out[c].astype(str).map(mask_pii_text)
    return out


# -----------------------------
# Data loading
# -----------------------------
@st.cache_data
def load_file(file) -> pd.DataFrame:
    name = file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(file)
    raise ValueError("CSV 또는 Excel만 지원합니다.")


def infer_datetime(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[str]]:
    out = df.copy()

    # Already datetime?
    for c in out.columns:
        if np.issubdtype(out[c].dtype, np.datetime64):
            return out, c

    cols = {c.lower(): c for c in out.columns}

    # Year/Month/Day
    if {"year", "month", "day"}.issubset(cols.keys()):
        y, m, d = cols["year"], cols["month"], cols["day"]
        out["date"] = pd.to_datetime(dict(year=out[y], month=out[m], day=out[d]), errors="coerce")
        return out, "date"

    # common single date col
    for key in ["date", "datetime", "time", "timestamp"]:
        if key in cols:
            c = cols[key]
            out[c] = pd.to_datetime(out[c], errors="coerce")
            return out, c

    return out, None


def numeric_cols(df: pd.DataFrame):
    return df.select_dtypes(include=[np.number]).columns.tolist()


def make_eda(df: pd.DataFrame) -> Dict[str, Any]:
    meta = []
    for c in df.columns:
        meta.append({
            "name": c,
            "dtype": str(df[c].dtype),
            "missing": int(df[c].isna().sum()),
            "n_unique": int(df[c].nunique(dropna=True))
        })

    num = numeric_cols(df)
    desc = {}
    if num:
        d = df[num].describe().T
        desc = d[["count","mean","std","min","25%","50%","75%","max"]].replace({np.nan: None}).to_dict()

    return {
        "shape": list(df.shape),
        "columns": meta,
        "numeric_describe": desc
    }


# -----------------------------
# Simple plots
# -----------------------------
def plot_time_series(df: pd.DataFrame, date_col: str, y_col: str, group_col: Optional[str] = None):
    fig, ax = plt.subplots()
    d = df.dropna(subset=[date_col, y_col]).sort_values(date_col)
    if group_col and group_col in d.columns:
        for g, gdf in d.groupby(group_col):
            ax.plot(gdf[date_col], gdf[y_col], label=str(g))
        ax.legend()
    else:
        ax.plot(d[date_col], d[y_col])
    ax.set_xlabel(date_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"{y_col} over time")
    st.pyplot(fig, clear_figure=True)


# -----------------------------
# OpenAI: Structured Output schema
# -----------------------------
INSIGHT_SCHEMA = {
    "name": "requirements_based_report",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "one_line_summary": {"type": "string"},
            "requirements_interpretation": {"type": "array", "items": {"type": "string"}},
            "key_findings": {"type": "array", "items": {"type": "string"}},
            "data_quality_warnings": {"type": "array", "items": {"type": "string"}},
            "statistical_notes": {"type": "array", "items": {"type": "string"}},
            "recommended_next_steps": {"type": "array", "items": {"type": "string"}},
            "executive_report_md": {"type": "string"}
        },
        "required": [
            "one_line_summary",
            "requirements_interpretation",
            "key_findings",
            "data_quality_warnings",
            "statistical_notes",
            "recommended_next_steps",
            "executive_report_md"
        ]
    }
}


def run_ai_report(
    df: pd.DataFrame,
    eda: Dict[str, Any],
    requirements: str,
    domain_context: str,
    date_col: Optional[str],
    y_col: Optional[str],
    group_col: Optional[str],
    api_key: str,
    model: str
) -> Dict[str, Any]:
    if OpenAI is None:
        raise RuntimeError("openai 패키지를 찾지 못했습니다. requirements.txt 설치를 확인하세요.")

    client = OpenAI(api_key=api_key)

    masked_sample = mask_pii_df(df, max_rows=200).to_dict(orient="records")

    payload = {
        "domain_context": domain_context,
        "user_requirements": requirements,
        "selected_datetime_col": date_col,
        "selected_target_col": y_col,
        "selected_group_col": group_col,
        "eda_summary": eda,
        "masked_sample_rows": masked_sample
    }

    system = (
        "당신은 환경/기후/해양 데이터 분석을 총괄하는 수석 분석가입니다. "
        "사용자의 user_requirements를 최우선으로 반영하여 결과를 작성하세요. "
        "과장 금지, 근거 기반. 상관=인과 오류 경고, 계절성/자기상관/결측/이상치/표본편향의 가능성을 반드시 언급하세요. "
        "출력은 반드시 주어진 JSON Schema를 만족해야 합니다. "
        "executive_report_md는 1~2페이지 분량의 마크다운 보고서로 작성하세요."
    )

    resp = client.responses.create(
        model=model,
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

    return json.loads(resp.output_text)


# -----------------------------
# Sidebar: inputs (requirements included)
# -----------------------------
with st.sidebar:
    st.header("⚙️ 실행 설정")

    domain_context = st.text_area(
        "도메인 컨텍스트(분석 목적/배경)",
        value="환경/기후 관측 데이터(예: 해빙, 온도, 강수)에서 장기 추세와 계절성을 점검하고 정책적 시사점을 도출",
        height=110
    )

    requirements = st.text_area(
        "요구 사항(여기에 입력한 내용이 AI 보고서에 반영됨)",
        value=(
            "- 결과는 정책 제언 중심으로\n"
            "- 결측치 처리/표본편향/자기상관/계절성 한계를 반드시 언급\n"
            "- 핵심 인사이트 5개 이내로 요약\n"
            "- 다음 실행(추가 분석/검정/시각화) 3개 제안"
        ),
        height=170
    )

    st.subheader("OpenAI (선택)")
    api_key_default = get_setting("OPENAI_API_KEY", "")
    model_default = get_setting("OPENAI_MODEL", "gpt-4.1-mini")

    api_key = st.text_input("OPENAI_API_KEY", value=api_key_default, type="password")
    model = st.text_input("MODEL", value=model_default)

    st.divider()
    st.subheader("데이터")
    uploaded = st.file_uploader("CSV/Excel 업로드", type=["csv", "xlsx", "xls"])
    use_sample = st.checkbox("샘플 데이터 사용(seaice.csv)", value=(uploaded is None))


# -----------------------------
# Load dataset
# -----------------------------
df = None
if uploaded is not None:
    df = load_file(uploaded)
elif use_sample:
    sample_path = os.path.join(os.path.dirname(__file__), "seaice.csv")
    if os.path.exists(sample_path):
        df = pd.read_csv(sample_path)
    else:
        st.error("샘플 seaice.csv가 앱 폴더에 없습니다. 파일 업로드를 사용하세요.")
        st.stop()

if df is None:
    st.info("왼쪽에서 파일을 업로드하거나 샘플 데이터를 선택하세요.")
    st.stop()

df, inferred_dt = infer_datetime(df)

# -----------------------------
# EDA
# -----------------------------
st.subheader("1) 데이터 요약 (EDA)")
eda = make_eda(df)

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("행", eda["shape"][0])
with c2:
    st.metric("열", eda["shape"][1])
with c3:
    st.metric("결측치 총합", sum(x["missing"] for x in eda["columns"]))

with st.expander("열 메타정보"):
    st.dataframe(pd.DataFrame(eda["columns"]).sort_values("missing", ascending=False), use_container_width=True)

if st.checkbox("데이터 미리보기", value=True):
    st.dataframe(df.head(50), use_container_width=True)

# -----------------------------
# Analysis selection
# -----------------------------
st.subheader("2) 분석 설정")
all_cols = df.columns.tolist()
num = numeric_cols(df)

colA, colB, colC = st.columns([1, 1, 1])
with colA:
    date_col = st.selectbox("시간 컬럼", options=[None] + all_cols, index=(1 if inferred_dt in all_cols else 0))
with colB:
    y_col = st.selectbox("분석 대상(수치형)", options=[None] + num, index=(1 if len(num) else 0))
with colC:
    group_candidates = [c for c in all_cols if df[c].dtype == "object"][:50]
    group_col = st.selectbox("그룹 컬럼(선택)", options=[None] + group_candidates, index=(1 if "hemisphere" in group_candidates else 0))

dff = df.copy()
if date_col and date_col in dff.columns:
    dff[date_col] = pd.to_datetime(dff[date_col], errors="coerce")
    dff = dff.dropna(subset=[date_col])

# -----------------------------
# Plots
# -----------------------------
st.subheader("3) 시각화")
if date_col and y_col:
    plot_time_series(dff, date_col, y_col, group_col=group_col)
else:
    st.info("시간 컬럼과 수치형 변수를 선택하면 시계열 그래프가 생성됩니다.")

# -----------------------------
# Run button: requirements-driven execution
# -----------------------------
st.subheader("4) 요구 사항 기반 실행")

st.markdown("아래 버튼을 누르면 **요구 사항(requirements)**을 포함해 AI가 보고서를 생성합니다.")
run = st.button("🚀 요구 사항 반영 AI 보고서 생성")

if run:
    if not api_key:
        st.error("OPENAI_API_KEY가 필요합니다. (Streamlit secrets 또는 환경변수로 설정 가능)")
        st.stop()
    with st.spinner("AI 보고서 생성 중..."):
        report = run_ai_report(
            df=dff,
            eda=eda,
            requirements=requirements,
            domain_context=domain_context,
            date_col=date_col,
            y_col=y_col,
            group_col=group_col,
            api_key=api_key,
            model=model
        )

    st.success("완료!")

    st.markdown("### ✅ 한 줄 요약")
    st.write(report["one_line_summary"])

    st.markdown("### 🧾 요구 사항 해석")
    st.write(report["requirements_interpretation"])

    st.markdown("### 📌 핵심 결과")
    st.write(report["key_findings"])

    st.markdown("### ⚠️ 데이터 품질 경고")
    st.write(report["data_quality_warnings"])

    st.markdown("### 🧠 통계적 유의사항")
    st.write(report["statistical_notes"])

    st.markdown("### ✅ 다음 실행(권고)")
    st.write(report["recommended_next_steps"])

    st.markdown("### 📄 Executive Report (Markdown)")
    st.markdown(report["executive_report_md"])

    st.download_button(
        "📥 AI 보고서(JSON) 다운로드",
        data=json.dumps(report, ensure_ascii=False, indent=2),
        file_name="requirements_based_env_report.json",
        mime="application/json"
    )
