# 🎬 ASK-THE-VIDEO

Ask any question about a YouTube video and get intelligent answers based on its transcript using Google Gemini and LangChain!

![ASK-THE-VIDEO Banner](https://img.shields.io/badge/Streamlit-Ask%20the%20Video-red?style=for-the-badge&logo=streamlit)

---

## 🚀 Overview

**ASK-THE-VIDEO** is an interactive Streamlit app that:
- Accepts a YouTube URL.
- Extracts and processes the video transcript.
- Converts the transcript into semantic vector embeddings.
- Uses a vector database (`FAISS`) for retrieval-augmented generation.
- Responds intelligently to user queries using Google's **Gemini 1.5 Flash** LLM.

---

## 🧠 Powered By

- [LangChain](https://www.langchain.com/)
- [Google Generative AI (Gemini)](https://ai.google.dev/)
- [HuggingFace Transformers](https://huggingface.co/)
- [FAISS (Facebook AI Similarity Search)](https://github.com/facebookresearch/faiss)
- [YouTube Transcript API](https://github.com/jdepoix/youtube-transcript-api)
- [Streamlit](https://streamlit.io/)

---

## 📦 Installation

1. **Clone the repository**

git clone https://github.com/yourusername/ask-the-video.git
cd ask-the-video
