
import os
import json
import re
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

st.set_page_config(page_title="환경 데이터 분석 (AI 포함)", layout="wide")
st.title("🌍 환경 데이터 분석 대시보드 (전문가용)")

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(\+?\d{1,3}[-\s]?)?(\d{2,4}[-\s]?\d{3,4}[-\s]?\d{4})")

def mask_pii_text(s):
    s = EMAIL_RE.sub("[EMAIL]", s)
    s = PHONE_RE.sub("[PHONE]", s)
    return s

def mask_pii_df(df):
    out = df.copy()
    for c in out.select_dtypes(include="object").columns:
        out[c] = out[c].astype(str).map(mask_pii_text)
    return out.head(100)

def get_client(api_key):
    if OpenAI is None:
        raise RuntimeError("openai 패키지가 필요합니다.")
    return OpenAI(api_key=api_key)

def make_eda(df):
    return {
        "shape": df.shape,
        "columns": [
            {"name": c, "dtype": str(df[c].dtype), "missing": int(df[c].isna().sum())}
            for c in df.columns
        ]
    }

INSIGHT_SCHEMA = {
    "name": "env_report",
    "schema": {
        "type": "object",
        "properties": {
            "one_line_summary": {"type": "string"},
            "key_findings": {"type": "array", "items": {"type": "string"}},
            "limitations": {"type": "array", "items": {"type": "string"}},
            "next_actions": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["one_line_summary", "key_findings", "limitations", "next_actions"]
    }
}

def run_ai(df, eda, requirements, api_key, model):
    masked = mask_pii_df(df)
    payload = {
        "eda": eda,
        "requirements": requirements,
        "sample": masked.to_dict(orient="records")
    }

    system = (
        "당신은 환경 데이터 분석 전문가입니다. "
        "사용자 요구 사항(requirements)을 최우선으로 반영하세요."
    )

    client = get_client(api_key)
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

with st.sidebar:
    st.header("설정")
    api_key = st.text_input("OPENAI_API_KEY", type="password")
    model = st.text_input("MODEL", value="gpt-4.1-mini")
    requirements = st.text_area(
        "요구 사항 입력",
        value="- 정책 제언 중심\n- 한계 명시\n- 다음 행동 제시",
        height=150
    )

uploaded = st.file_uploader("CSV 업로드", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
    st.dataframe(df.head())

    eda = make_eda(df)
    st.write("EDA 요약", eda)

    if api_key and st.button("AI 분석 실행"):
        with st.spinner("AI 분석 중..."):
            report = run_ai(df, eda, requirements, api_key, model)

        st.subheader("한 줄 요약")
        st.write(report["one_line_summary"])

        st.subheader("핵심 발견")
        st.write(report["key_findings"])

        st.subheader("한계")
        st.write(report["limitations"])

        st.subheader("다음 행동")
        st.write(report["next_actions"])
