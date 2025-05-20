import streamlit as st
from audiorecorder import audiorecorder
from faster_whisper import WhisperModel
import os

# Configurar página
st.set_page_config(page_title="Transcriptor de Audio en Español", layout="centered")
st.title("🎤 Transcribe tu voz (Español)")

# Cargar modelo Whisper local
@st.cache_resource
def load_whisper_model():
    model = WhisperModel("small", device="cpu", compute_type="int8")
    return model

model = load_whisper_model()

# Grabación de audio
st.write("Haz clic en el micrófono para grabar tu voz:")
audio = audiorecorder("🎙️ Grabar", "Grabando...")

if audio:
    st.audio(audio, format="audio/wav")

    # Guardar audio temporal
    temp_file = "temp_audio.wav"
    audio.export(temp_file, format="wav")

    # Transcribir
    with st.spinner("Transcribiendo..."):
        try:
            segments, info = model.transcribe(temp_file, beam_size=5, language="es")
            transcription = "".join([seg.text for seg in segments])
            st.success("Transcripción:")
            st.write(transcription)
        except Exception as e:
            st.error(f"Error al transcribir: {e}")
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)
