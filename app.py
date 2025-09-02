import os
from urllib.parse import urlparse, parse_qs
import streamlit as st
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from youtube_transcript_api.formatters import TextFormatter
from langchain_openai import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="VideoGyaan - YouTube Summary and Q&A", layout="wide")
st.title("VideoGyaan - YouTube Summary and Q&A")

if not openai_api_key:
    st.error("OpenAI API key not found. Please set OPENAI_API_KEY in your .env file.")
    st.stop()

def extract_video_id(url: str) -> str:
    p = urlparse(url)
    host = (p.hostname or "").lower()
    if host in {"youtu.be"}:
        return p.path.lstrip("/")
    if host in {"www.youtube.com", "youtube.com", "m.youtube.com"}:
        q = parse_qs(p.query)
        if "v" in q and len(q["v"]) > 0:
            return q["v"][0]
        parts = [x for x in p.path.split("/") if x]
        if "embed" in parts:
            i = parts.index("embed")
            if i + 1 < len(parts):
                return parts[i + 1]
        if "shorts" in parts:
            i = parts.index("shorts")
            if i + 1 < len(parts):
                return parts[i + 1]
    return url

video_url = st.text_input("Enter YouTube Video URL")
language_choice = st.selectbox("Choose transcript language", ["en", "hi", "es", "fr", "de"])

if video_url:
    try:
        video_id = extract_video_id(video_url).strip()
        if not video_id:
            st.error("Could not parse a valid YouTube video ID from the provided URL.")
            st.stop()

        ytt = YouTubeTranscriptApi()
        try:
            fetched = ytt.fetch(video_id, languages=[language_choice, "en"])
        except NoTranscriptFound:
            fetched = ytt.fetch(video_id, languages=["en"])

        formatter = TextFormatter()
        full_transcript = formatter.format_transcript(fetched)

        if not full_transcript or not full_transcript.strip():
            st.warning("Transcript is empty or could not be fetched.")
            st.stop()

        st.subheader("Transcript")
        st.text_area("Transcript", value=full_transcript, height=300)

        chunk_size = 2000
        chunks = [full_transcript[i:i + chunk_size] for i in range(0, len(full_transcript), chunk_size)]
        docs = [Document(page_content=c) for c in chunks]

        llm = ChatOpenAI(api_key=openai_api_key, temperature=0, model="gpt-4o-mini")
        chain = load_summarize_chain(llm, chain_type="map_reduce")

        st.subheader("Summary")
        summary = chain.run(docs)
        st.write(summary)

        st.subheader("Ask a Question")
        question_input = st.text_input("Ask a question about the video or topic")
        if question_input:
            with st.spinner("Generating answer..."):
                context = "\n\n".join(chunks[-3:]) if chunks else full_transcript
                prompt = f"""You are a helpful assistant. Based on the following transcript, answer the user's question clearly and concisely.

---Transcript---
{context}
----------------

Question: {question_input}
Answer:"""
                try:
                    resp = llm.invoke(prompt)
                    st.markdown("**Answer:**")
                    st.write(getattr(resp, "content", str(resp)))
                except Exception as e:
                    st.error(f"An error occurred while generating the answer: {e}")

    except TranscriptsDisabled:
        st.error("Transcripts are disabled for this video.")
    except NoTranscriptFound:
        st.error("No transcripts available for this video.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
