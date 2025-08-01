import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
import os
from dotenv import load_dotenv

from langchain_core.output_parsers import StrOutputParser
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

st.set_page_config(page_title="Ask-The-Video", layout="wide")
st.title("🎬 ASK-THE-VIDEO")
def extract_video_id(url):
    try:
        if "youtube.com/watch" in url:
            query = urlparse(url).query
            return parse_qs(query)['v'][0]
        elif "youtu.be/" in url:
            return url.split("/")[-1].split("?")[0]
        elif "youtube.com/live/" in url:
            return url.split("/")[-1].split("?")[0]
        else:
            return None
    except Exception:
        return None
col1, col2 = st.columns([1, 2])

with col1:
    url = st.text_input("🔗 Enter YouTube video URL")
    if url:
        video_id = extract_video_id(url)
        if not video_id:
            st.error("❌ Invalid YouTube URL.")
        else:
            st.video(f"https://www.youtube.com/watch?v={video_id}")

with col2:
    if url and video_id:
        try:
            proxies = {
    "http": "http://98.126.232.10:80",
    "https": "http://98.126.232.10:80"
}

            
            transcript_list = YouTubeTranscriptApi().fetch(video_id,  proxies=proxies)
            full_text = " ".join(chunk.text for chunk in transcript_list)

            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            docs = text_splitter.create_documents([full_text])

            embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vectorstore = FAISS.from_documents(docs, embedding_model)

            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

            query = st.chat_input("Ask anything about the video...")
            if query:
                retrieved_docs = vectorstore.similarity_search(query, k=3)
                context = "\n\n".join(doc.page_content for doc in retrieved_docs)
                raw_response = llm.invoke(f"{context}\n\nQ: {query}")
                parser = StrOutputParser()
                response = parser.invoke(raw_response)

                
                st.write(response)

        except Exception as e:
            st.error(f"⚠️ Error: {e}")
