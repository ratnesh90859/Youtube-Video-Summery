import os
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from youtube_transcript_api.formatters import TextFormatter
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    st.error("OpenAI API key not found. Please set OPENAI_API_KEY in your .env file.")
    st.stop()

# Streamlit UI
st.title("📹 VideoGyaan")

video_url = st.text_input("Enter YouTube video URL")

# Language selector
language_choice = st.selectbox("Choose transcript language", ["en", "hi", "es", "fr", "de"])

if video_url:
    try:
        # Extract video ID
        video_id = video_url.split("v=")[-1].split("&")[0]

        # Fetch transcript
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        try:
            # Try fetching the requested language
            transcript = transcript_list.find_transcript([language_choice])
        except NoTranscriptFound:
            st.warning(f"No transcript found in {language_choice}. Trying English...")

            # Fallback to English if the selected language transcript is not available
            transcript = transcript_list.find_transcript(['en'])

        # Get the transcript text
        transcript_text = transcript.fetch()
        formatter = TextFormatter()
        full_transcript = formatter.format_transcript(transcript_text)

        if not full_transcript:
            st.warning("Transcript is empty or could not be fetched.")
            st.stop()

        # Show transcript
        st.subheader("📜 Transcript")
        st.text_area("Transcript", value=full_transcript, height=300)

        # Split the transcript into chunks to fit within the token limit
        chunk_size = 2000  # Adjust chunk size as needed
        chunks = [full_transcript[i:i+chunk_size] for i in range(0, len(full_transcript), chunk_size)]

        # Summarize each chunk
        st.subheader("📝 Summary")
        llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)
        chain = load_summarize_chain(llm, chain_type="map_reduce")
        
        # Create documents for each chunk
        docs = [Document(page_content=chunk) for chunk in chunks]
        summary = chain.run(docs)
        st.write(summary)

        # Ask questions
        st.subheader("❓ Ask a Question")
        question_input = st.text_input("Ask a question about the video or topic")

        if question_input:
            with st.spinner("Thinking..."):
                # Limit the context to the most recent chunk or a relevant portion
                relevant_chunk = chunks[-1]  # Using the most recent chunk
                prompt = f"""
                You are a helpful assistant. Based on the following video transcript, answer the user's question:

                ---Transcript---
                {relevant_chunk}
                ----------------

                Question: {question_input}
                Answer:"""

                try:
                    response = llm.predict(prompt)
                    st.markdown("**Answer:**")
                    st.write(response)
                except Exception as e:
                    st.error(f"An error occurred while generating the answer: {e}")

    except TranscriptsDisabled:
        st.error("Transcripts are disabled for this video.")
    except NoTranscriptFound:
        st.error("No transcripts available for this video.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
