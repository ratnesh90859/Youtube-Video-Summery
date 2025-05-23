import os
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from youtube_transcript_api.formatters import TextFormatter
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document

# Load environment variables
load_dotenv()

# Fetch OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

# Check if OpenAI API key is present
if not openai_api_key:
    st.error("OpenAI API key not found. Please set OPENAI_API_KEY in your .env file.")
    st.stop()

# Set title for the Streamlit app
st.title("📹 VideoGyaan - YouTube Summary and Q&A")

# Input for YouTube video URL
video_url = st.text_input("Enter YouTube Video URL")

# Selectbox for choosing the transcript language
language_choice = st.selectbox("Choose transcript language", ["en", "hi", "es", "fr", "de"])

# Proceed if a valid video URL is provided
if video_url:
    try:
        # Extract video ID from the URL
        video_id = video_url.split("v=")[-1].split("&")[0]

        # Get the available transcripts for the video
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        try:
            # Try fetching the transcript in the selected language
            transcript = transcript_list.find_transcript([language_choice])
        except NoTranscriptFound:
            # If no transcript found in selected language, try English
            st.warning(f"No transcript found in {language_choice}. Trying English...")
            transcript = transcript_list.find_transcript(['en'])

        # Fetch the full transcript text
        transcript_text = transcript.fetch()
        formatter = TextFormatter()
        full_transcript = formatter.format_transcript(transcript_text)

        # Check if the transcript is empty
        if not full_transcript:
            st.warning("Transcript is empty or could not be fetched.")
            st.stop()

        # Display the transcript
        st.subheader("📜 Transcript")
        st.text_area("Transcript", value=full_transcript, height=300)

        # Split the transcript into chunks for summarization
        chunk_size = 2000
        chunks = [full_transcript[i:i+chunk_size] for i in range(0, len(full_transcript), chunk_size)]

        # Initialize OpenAI model for summarization
        llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0)
        chain = load_summarize_chain(llm, chain_type="map_reduce")

        # Create document objects from chunks
        docs = [Document(page_content=chunk) for chunk in chunks]

        # Summarize the transcript
        st.subheader("📝 Summary")
        summary = chain.run(docs)
        st.write(summary)

        # Input for asking a question
        st.subheader("❓ Ask a Question")
        question_input = st.text_input("Ask a question about the video or topic")

        # If the user asks a question, generate an answer based on the transcript
        if question_input:
            with st.spinner("Thinking..."):
                context = "\n\n".join(chunks[-3:])  # Use last few chunks for more context
                prompt = f"""
You are a helpful assistant. Based on the following transcript, answer the user's question:

---Transcript---
{context}
----------------

Question: {question_input}
Answer:"""

                try:
                    # Get the response from the model
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
