import os
import re
import json
import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAI

# -----------------------------
# 0) 기본 설정
# -----------------------------
st.set_page_config(page_title="AI 분석 대시보드", layout="wide")

def get_secret(key: str, default=None):
    # Streamlit secrets -> env 순으로 읽기
    if key in st.secrets:
        return st.secrets[key]
    return os.getenv(key, default)

MODEL = get_secret("OPENAI_MODEL", "gpt-4.1-mini")

@st.cache_resource
def get_client():
    # OpenAI SDK는 OPENAI_API_KEY 환경변수도 자동 인식 가능
    # (여기서는 st.secrets/env에서 읽어 직접 주입)
    api_key = get_secret("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY가 설정되지 않았습니다.")
    return OpenAI(api_key=api_key)

client = get_client()

# -----------------------------
# 1) 유틸: PII 마스킹
# -----------------------------
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(\+?\d{1,3}[-\s]?)?(\d{2,4}[-\s]?\d{3,4}[-\s]?\d{4})")

def mask_pii_text(s: str) -> str:
    s = EMAIL_RE.sub("[EMAIL]", s)
    s = PHONE_RE.sub("[PHONE]", s)
    return s

def mask_pii_df(df: pd.DataFrame, max_cells: int = 20000) -> pd.DataFrame:
    # 큰 데이터는 비용/지연 때문에 "일부만" 문자열 마스킹
    out = df.copy()
    # 문자열 칼럼만
    obj_cols = [c for c in out.columns if out[c].dtype == "object"]
    # 너무 크면 앞부분만
    if out.size > max_cells:
        out = out.head(max(50, min(500, len(out))))
    for c in obj_cols:
        out[c] = out[c].astype(str).map(mask_pii_text)
    return out

# -----------------------------
# 2) 데이터 로딩
# -----------------------------
@st.cache_data
def load_data(file) -> pd.DataFrame:
    name = file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(file)
    raise ValueError("CSV 또는 Excel만 지원합니다.")

# -----------------------------
# 3) EDA 요약 생성
# -----------------------------
def make_eda_summary(df: pd.DataFrame) -> dict:
    summary = {}
    summary["shape"] = list(df.shape)
    summary["columns"] = [{"name": c, "dtype": str(df[c].dtype), "missing": int(df[c].isna().sum())}
                          for c in df.columns]
    # 수치 요약
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        desc = df[num_cols].describe().T.replace({np.nan: None})
        summary["numeric_describe"] = desc[["count","mean","std","min","25%","50%","75%","max"]].to_dict()
    else:
        summary["numeric_describe"] = {}

    # 범주형 상위 값
    cat_cols = [c for c in df.columns if df[c].dtype == "object"]
    topcats = {}
    for c in cat_cols[:30]:
        vc = df[c].astype(str).value_counts(dropna=False).head(10)
        topcats[c] = [{"value": str(i), "count": int(vc[i])} for i in vc.index]
    summary["top_categories"] = topcats
    return summary

# -----------------------------
# 4) LLM: 구조화 인사이트(JSON Schema)
# -----------------------------
INSIGHT_SCHEMA = {
  "name": "analysis_report",
  "schema": {
    "type": "object",
    "additionalProperties": False,
    "properties": {
      "one_line_summary": {"type": "string"},
      "key_insights": {
        "type": "array",
        "items": {
          "type": "object",
          "additionalProperties": False,
          "properties": {
            "title": {"type": "string"},
            "evidence": {"type": "string"},
            "why_it_matters": {"type": "string"},
            "recommended_next_step": {"type": "string"}
          },
          "required": ["title", "evidence", "why_it_matters", "recommended_next_step"]
        }
      },
      "data_quality_risks": {
        "type": "array",
        "items": {"type": "string"}
      },
      "statistical_notes": {
        "type": "array",
        "items": {"type": "string"}
      },
      "suggested_additional_analyses": {
        "type": "array",
        "items": {
          "type": "object",
          "additionalProperties": False,
          "properties": {
            "analysis": {"type": "string"},
            "how_to_do": {"type": "string"},
            "expected_output": {"type": "string"}
          },
          "required": ["analysis", "how_to_do", "expected_output"]
        }
      }
    },
    "required": ["one_line_summary", "key_insights", "data_quality_risks", "statistical_notes", "suggested_additional_analyses"]
  }
}

def llm_insights(df: pd.DataFrame, eda: dict, domain: str) -> dict:
    # 모델에 보내는 데이터는 "요약 + PII 마스킹된 샘플"만
    safe_df = mask_pii_df(df)
    sample_rows = safe_df.sample(min(30, len(safe_df)), random_state=42).to_dict(orient="records")

    system = (
        "당신은 데이터 분석 책임자(Lead Data Scientist)입니다. "
        "출력은 반드시 주어진 JSON Schema를 만족해야 합니다. "
        "과장 금지, 근거 기반으로만 작성하고, 통계적 한계/표본 편향 가능성을 반드시 포함하세요."
    )
    user = {
        "domain_context": domain,
        "eda_summary": eda,
        "masked_sample_rows": sample_rows
    }

    # Responses API는 OpenAI의 최신 통합 인터페이스입니다. :contentReference[oaicite:5]{index=5}
    resp = client.responses.create(
        model=MODEL,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)}
        ],
        # Structured Outputs 가이드: JSON Schema로 강제 :contentReference[oaicite:6]{index=6}
        text={
            "format": {
                "type": "json_schema",
                "json_schema": INSIGHT_SCHEMA
            }
        }
    )

    # SDK 버전에 따라 접근 방식이 다를 수 있어 안전 파싱
    # resp.output_text가 있으면 그걸 쓰고, 아니면 output에서 텍스트를 찾아봄
    out_text = getattr(resp, "output_text", None)
    if not out_text:
        out_text = json.dumps(resp.model_dump(), ensure_ascii=False)

    # out_text는 JSON이어야 함
    return json.loads(out_text)

# -----------------------------
# 5) UI
# -----------------------------
st.title("📊 AI 기반 설문/데이터 분석 대시보드")

with st.sidebar:
    st.header("설정")
    domain = st.text_input("분석 도메인 컨텍스트", "고등학생 사교육비/설문 데이터 분석")
    show_raw = st.checkbox("원본 데이터 미리보기", True)
    run_ai = st.button("AI 인사이트 생성")

file = st.file_uploader("CSV 또는 Excel 업로드", type=["csv", "xlsx", "xls"])

if file:
    df = load_data(file)
    st.success(f"로드 완료: {df.shape[0]}행 × {df.shape[1]}열")

    if show_raw:
        st.subheader("데이터 미리보기")
        st.dataframe(df.head(30), use_container_width=True)

    st.subheader("기본 EDA")
    eda = make_eda_summary(df)

    c1, c2 = st.columns(2)
    with c1:
        st.metric("행 수", eda["shape"][0])
        st.metric("열 수", eda["shape"][1])
    with c2:
        miss_total = sum(x["missing"] for x in eda["columns"])
        st.metric("결측치 총합", miss_total)

    # 결측 상위 열
    miss_df = pd.DataFrame(eda["columns"]).sort_values("missing", ascending=False).head(15)
    st.write("결측치 상위 15개 열")
    st.dataframe(miss_df, use_container_width=True)

    # 수치형 요약 표
    if eda["numeric_describe"]:
        st.write("수치형 요약(Describe)")
        desc = pd.DataFrame(eda["numeric_describe"])
        st.dataframe(desc, use_container_width=True)

    # AI 인사이트
    if run_ai:
        with st.spinner("AI 인사이트 생성 중..."):
            report = llm_insights(df, eda, domain)

        st.subheader("AI 요약")
        st.write(report["one_line_summary"])

        st.subheader("핵심 인사이트")
        for i, ins in enumerate(report["key_insights"], 1):
            with st.expander(f"{i}. {ins['title']}"):
                st.markdown(f"**근거**: {ins['evidence']}")
                st.markdown(f"**의미**: {ins['why_it_matters']}")
                st.markdown(f"**다음 단계**: {ins['recommended_next_step']}")

        st.subheader("데이터 품질/해석 리스크")
        st.write(report["data_quality_risks"])

        st.subheader("통계적 유의사항")
        st.write(report["statistical_notes"])

        st.subheader("추가 분석 제안")
        for s in report["suggested_additional_analyses"]:
            st.markdown(f"- **{s['analysis']}**  \n  방법: {s['how_to_do']}  \n  산출물: {s['expected_output']}")

        # 다운로드(보고서 JSON)
        st.download_button(
            "📥 인사이트 JSON 다운로드",
            data=json.dumps(report, ensure_ascii=False, indent=2),
            file_name="ai_analysis_report.json",
            mime="application/json"
        )
else:
    st.info("파일을 업로드하면 분석을 시작합니다.")
