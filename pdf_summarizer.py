from transformers import pipeline
import streamlit as st
from PyPDF2 import PdfReader

# Load lightweight model from Hugging Face
text_generator = pipeline("text-generation", model="EleutherAI/gpt-neo-125M")

# Function to summarize PDFs
def summarize_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
    summary = text_generator(text[:1000], max_length=300, do_sample=True)[0]['generated_text']
    return summary

# Stylish UI
st.set_page_config(page_title="Research Tool", page_icon="🔎", layout="wide")
st.markdown("<h1 style='text-align: center; color: darkblue;'>Research Tool</h1>", unsafe_allow_html=True)

# Select paper
paper_input = st.selectbox("📄 Select Research Paper Name", [
    "Attention Is All You Need",
    "BERT: Pre-training of Deep Bidirectional Transformers",
    "GPT-3: Language Models are Few-Shot Learners",
    "Diffusion Models Beat GANs on Image Synthesis"
])

# Select explanation style
style_input = st.selectbox("🎨 Select Explanation Style", [
    "Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"
])

# Select explanation length
length_input = st.selectbox("📏 Select Explanation Length", [
    "Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"
])

# File upload for PDF summarization
uploaded_file = st.file_uploader("📎 Upload a Research Paper (PDF)", type="pdf")
if uploaded_file:
    st.subheader("Summary of the Uploaded PDF:")
    st.write(summarize_pdf(uploaded_file))