import streamlit as st
from pypdf import PdfReader
from transformers import BartForConditionalGeneration, BartTokenizer
import re

# 1. Page Configuration
st.set_page_config(
    page_title="AI Resume Summarizer & Extractor", 
    page_icon="📄", 
    layout="centered"
)

st.title("📄 AI Resume Summarizer")
st.markdown("Upload any candidate's resume (PDF or plain text) to extract key profile data and generate a 2-3 line executive summary.")

# 2. Cache & Load Model & Tokenizer
@st.cache_resource(show_spinner=False)
def load_model_and_tokenizer():
    model_name = "sshleifer/distilbart-cnn-12-6"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

with st.spinner("Loading AI Model weights..."):
    tokenizer, model = load_model_and_tokenizer()

# 3. Information Extraction Functions
def clean_extracted_text(text: str) -> str:
    """Standardizes whitespace and removes irregular unicode artifacts."""
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

def extract_candidate_name(raw_text: str) -> str:
    """Extracts candidate full name from header or initial lines."""
    lines = [line.strip() for line in raw_text.split('\n') if line.strip()]
    for line in lines[:5]:
        cleaned = re.sub(r'[^a-zA-Z\s]', '', line).strip()
        words = cleaned.split()
        # Look for a line containing 2-4 clean alphabetic words
        if 2 <= len(words) <= 4:
            if not any(k in cleaned.lower() for k in [
                "resume", "curriculum", "vitae", "profile", "contact", "summary", 
                "education", "skills", "experience", "phone", "email", "address"
            ]):
                return " ".join([w.title() for w in words])
    return "Candidate"

def extract_key_achievements(raw_text: str) -> list:
    """Extracts bullet points, percentages, or metric-based achievements."""
    achievements = []
    lines = raw_text.split('\n')
    
    # Check for bullet points, performance metrics (%), and key action verbs
    for line in lines:
        cleaned_line = line.strip().lstrip('•-*—–> ').strip()
        if not cleaned_line:
            continue
            
        is_metric_driven = bool(re.search(r'\d+%', cleaned_line) or re.search(r'\b(increased|reduced|improved|boosted|optimized|led|developed|created|built)\b', cleaned_line, re.IGNORECASE))
        is_valid_length = 25 <= len(cleaned_line) <= 200
        not_header = not any(h in cleaned_line.lower() for h in ["achievements", "education", "experience", "skills", "contacts", "summary"])

        if (line.strip().startswith(('•', '-', '*', '–')) or is_metric_driven) and is_valid_length and not_header:
            if cleaned_line not in achievements:
                achievements.append(cleaned_line)

    return achievements[:4]  # Return top 3-4 highlights

def format_summary_output(summary: str, detected_name: str) -> str:
    """Cleans punctuation and aligns proper names."""
    summary = re.sub(r'\s+([.,!?;:])', r'\1', summary)
    summary = '. '.join(s.strip().capitalize() for s in summary.split('.') if s.strip()) + '.'
    
    if detected_name and detected_name != "Candidate":
        words = summary.split()
        if len(words) > 0 and words[0].lower() not in ["the", "a", "an", "this", "experienced"]:
            summary = re.sub(re.escape(words[0]), detected_name, summary, count=1)
            
    return summary

# 4. UI Input Controls
uploaded_file = st.file_uploader("Upload Resume (PDF format)", type=["pdf"])
text_input = st.text_area("Or paste raw resume text:", height=180, placeholder="Paste resume contents here...")

raw_text = ""

if uploaded_file is not None:
    try:
        reader = PdfReader(uploaded_file)
        pages_text = [page.extract_text() or "" for page in reader.pages]
        raw_text = "\n".join(pages_text)
    except Exception as e:
        st.error(f"Error reading PDF file: {str(e)}")
elif text_input.strip():
    raw_text = text_input

# 5. Pipeline Execution
if st.button("Generate Summary & Highlights", type="primary"):
    cleaned_input = clean_extracted_text(raw_text)
    
    if not cleaned_input:
        st.warning("Please upload a valid PDF or enter resume text first.")
    else:
        with st.spinner("Analyzing candidate profile and generating summary..."):
            try:
                candidate_name = extract_candidate_name(raw_text)
                achievements = extract_key_achievements(raw_text)
                
                # Tokenize for transformer summarization
                inputs = tokenizer(
                    cleaned_input,
                    max_length=1024,
                    truncation=True,
                    return_tensors="pt"
                )
                
                # Generate 2-3 line summary
                summary_ids = model.generate(
                    inputs["input_ids"],
                    num_beams=4,
                    length_penalty=2.0,
                    max_length=80,
                    min_length=35,
                    no_repeat_ngram_size=3,
                    early_stopping=True
                )
                
                raw_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                final_summary = format_summary_output(raw_summary, candidate_name)
                
                st.success("Analysis Complete!")
                
                # Candidate Header
                st.subheader(f"👤 Candidate: {candidate_name}")
                
                # Executive Summary Box
                st.markdown("### 📌 Executive Summary")
                st.info(final_summary)
                
                # Key Achievements Box
                if achievements:
                    st.markdown("### 🏆 Key Achievements & Highlights")
                    for item in achievements:
                        st.markdown(f"- {item}")
                
            except Exception as e:
                st.error(f"Inference error: {str(e)}")