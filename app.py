import streamlit as st
from utils.analyzer import analyze_resume_and_jd
from PyPDF2 import PdfReader
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="AI Resume Analyzer", page_icon="📄")
st.title("📄 AI Resume Analyzer with Job Match Scoring")

st.markdown("Upload your **Resume** and a **Job Description** to analyze how well they match.")

col1, col2 = st.columns(2)
with col1:
    resume_pdf = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

with col2:
    jd_pdf = st.file_uploader("Upload Job Description (PDF)", type=["pdf"])

if st.button("Analyze Match"):
    if resume_pdf and jd_pdf:
        result = analyze_resume_and_jd(resume_pdf, jd_pdf)

        st.metric("Match Score", f"{result['score']}%")
        st.subheader("Missing Keywords")
        st.write(", ".join(result['missing_keywords']) or "None 🎉")

        st.subheader("AI Suggestions")
        st.write(result['suggestions'])
    else:
        st.warning("Please upload both Resume and Job Description.")
