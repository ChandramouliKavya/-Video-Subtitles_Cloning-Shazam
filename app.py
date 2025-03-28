import streamlit as st
import soundfile as sf
import numpy as np
from transformers import pipeline
import chromadb
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Load AssemblyAI API key from environment
assembly_api_key = os.environ.get("ASSEMBLYAI_API_KEY")

if not assembly_api_key:
    st.error("❌ AssemblyAI API key not found. Please provide it in your .env file.")

# Load the Hugging Face speech-to-text model
@st.cache_resource
def load_speech_to_text_model():
    return pipeline("automatic-speech-recognition", model="openai/whisper-small")

speech_to_text = load_speech_to_text_model()

# Ensure collection exists
def ensure_collection():
    try:
        chroma_client = chromadb.PersistentClient(path="./dataset")
        try:
            return chroma_client.get_collection("subtitles")
        except Exception:
            chroma_client.create_collection(name="subtitles")
            return chroma_client.get_collection("subtitles")
    except Exception as e:
        st.error(f"Error while ensuring collection: {e}")
        return None

collection = ensure_collection()

# Function to convert audio to text
def convert_audio_to_text(audio_file):
    try:
        audio_data, samplerate = sf.read(audio_file)
        
        # Ensure single-channel (mono) audio
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        result = speech_to_text(audio_data, return_timestamps=True)
        transcription = " ".join([segment['text'] for segment in result['chunks']])
        return transcription
    except Exception as e:
        st.error(f"Audio conversion failed: {e}")
        return None

# Function to search for relevant subtitles
def search_subtitles(query):
    try:
        if collection is None:
            return ["Subtitle collection not found."]
        results = collection.query(query_texts=[query], n_results=5)
        return results["documents"][0] if results["documents"] else ["No matching subtitles found."]
    except Exception as e:
        st.error(f"Subtitle search failed: {e}")
        return ["Error during search."]

# Streamlit UI
st.title("🎬 Video Subtitle Search Engine")
st.write("Upload an audio file to search for matching subtitles.")

# File uploader for audio
uploaded_file = st.file_uploader("Upload Audio File", type=["wav", "mp3", "ogg"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    
    with st.spinner("Converting speech to text... 🎙️"):
        query_text = convert_audio_to_text(uploaded_file)
        if query_text:
            st.write("Detected Text:", query_text)
        else:
            st.stop()

    with st.spinner("Searching subtitles... 🔎"):
        subtitles = search_subtitles(query_text)
        st.write("Matching Subtitles:")
        for subtitle in subtitles:
            st.write("-", subtitle)
