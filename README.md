# 📄 AI Resume Summarizer & Extractor

An automated GenAI tool built with Streamlit and Hugging Face Transformers that parses candidate resumes (PDF or raw text), extracts profile highlights, and generates concise 2-3 line executive summaries.

---

## 🎯 Objective
To accelerate candidate screening and recruitment workflows by automatically distilling extensive resume data into crisp, actionable 2-3 line summaries paired with metric-driven achievements.

---

## 🛠️ Tech Stack
* **Language:** Python
* **NLP / Model:** Hugging Face `transformers` (`sshleifer/distilbart-cnn-12-6`)
* **Deep Learning Engine:** PyTorch (`torch`)
* **Document Parsing:** `pypdf`
* **Interface / Framework:** Streamlit
* **Version Control:** Git & GitHub

---

## ⚙️ Implementation Details
1. **Document Ingestion:** Extracts raw text from uploaded multi-page PDF documents via `pypdf` or captures pasted text directly.
2. **Text Normalization & Entity Extraction:** Cleans text formatting, extracts candidate names, and filters metric-driven action bullets for key achievements.
3. **Transformer Summarization:** Utilizes DistilBART via beam search decoding (`num_beams=4`, constrained between 35 and 80 tokens) to generate abstractive 2-3 sentence executive summaries.
4. **Interactive Dashboard:** Built using Streamlit for testing and document analysis.

---

## 📸 Screenshots
*(Save your screenshot image as `screenshot.png` in this folder)*
![Resume Summarizer Demo](screenshot.png)

---

## 🚀 How to Run Locally

1. **Clone repository:**
   ```bash
   git clone [https://github.com/himanshhuu973/resume-summarizer-app.git](https://github.com/himanshhuu973/resume-summarizer-app.git)
   cd resume-summarizer-app