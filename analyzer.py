from PyPDF2 import PdfReader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import google.generativeai as genai
import os

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

embedder = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

def extract_text(file):
    reader = PdfReader(file)
    return " ".join([page.extract_text() for page in reader.pages if page.extract_text()])

def get_embedding(text):
    return embedder.embed_query(text)

def get_similarity_score(resume_text, jd_text):
    resume_vec = get_embedding(resume_text)
    jd_vec = get_embedding(jd_text)
    score = cosine_similarity([resume_vec], [jd_vec])[0][0]
    return round(score * 100)

def extract_keywords(jd_text):
    prompt = f"""
    Extract the important skills, tools, and technologies from the following job description:
    {jd_text}
    Return them as a comma-separated list.
    """
    response = genai.GenerativeModel("gemini-pro").generate_content(prompt)
    keywords = [kw.strip().lower() for kw in response.text.split(",") if kw.strip()]
    return list(set(keywords))

def get_ai_suggestions(resume_text, jd_text):
    prompt = f"""
    You are an AI resume coach. Given the following resume and job description, suggest three improvements the user should make to better match the job.

    Resume:
    {resume_text}

    Job Description:
    {jd_text}
    """
    response = genai.GenerativeModel("gemini-pro").generate_content(prompt)
    return response.text.strip()

def analyze_resume_and_jd(resume_file, jd_file):
    resume_text = extract_text(resume_file)
    jd_text = extract_text(jd_file)

    score = get_similarity_score(resume_text, jd_text)
    jd_keywords = extract_keywords(jd_text)
    missing = [kw for kw in jd_keywords if kw not in resume_text.lower()]
    suggestions = get_ai_suggestions(resume_text, jd_text)

    return {
        "score": score,
        "missing_keywords": missing,
        "suggestions": suggestions
    }
