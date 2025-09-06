import streamlit as st
import whisper
from deep_translator import GoogleTranslator
import sounddevice as sd
import numpy as np
import queue
import threading
import tempfile
import scipy.io.wavfile as wavfile
import time

# --------------------
# Configuración inicial
# --------------------
st.set_page_config(page_title="🎤 English ↔ Spanish Live", layout="wide")
st.title("🎤 Live Transcription & Translation (GRATIS)")
st.write("Habla al micrófono y mira la transcripción y traducción en tiempo real.")

col1, col2 = st.columns(2)
col1.subheader("📌 Original")
col2.subheader("🌎 Traducción")

# Cargar modelo Whisper
model = whisper.load_model("small")

# --------------------
# Cola de audio para procesar
# --------------------
audio_queue = queue.Queue()
stop_recording = threading.Event()

# --------------------
# Función para grabar audio en segundo plano
# --------------------
def record_audio(fs=16000, block_duration=2):
    def callback(indata, frames, time_info, status):
        audio_queue.put(indata.copy())

    with sd.InputStream(samplerate=fs, channels=1, callback=callback):
        while not stop_recording.is_set():
            sd.sleep(int(block_duration * 1000))

# --------------------
# Función para procesar audio
# --------------------
def process_audio():
    accumulated_text = ""
    while not stop_recording.is_set() or not audio_queue.empty():
        try:
            audio_block = audio_queue.get(timeout=1)
        except queue.Empty:
            continue

        # Guardar temporalmente
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        wavfile.write(temp_file.name, 16000, np.int16(audio_block * 32767))

        # Transcribir con Whisper
        result = model.transcribe(temp_file.name)
        new_text = result["text"].strip()
        if new_text:
            accumulated_text += " " + new_text
            col1.empty()
            col1.write(accumulated_text)

            # Traducir con Google Translator
            try:
                translated_text = GoogleTranslator(source='auto', target='es').translate(accumulated_text)
            except Exception as e:
                translated_text = f"Error en traducción: {e}"

            col2.empty()
            col2.write(translated_text)

        time.sleep(0.1)  # pequeño delay para no saturar

# --------------------
# Botón para iniciar/detener
# --------------------
if "recording" not in st.session_state:
    st.session_state.recording = False

def toggle_recording():
    st.session_state.recording = not st.session_state.recording
    if st.session_state.recording:
        stop_recording.clear()
        threading.Thread(target=record_audio, daemon=True).start()
        threading.Thread(target=process_audio, daemon=True).start()
    else:
        stop_recording.set()

st.button("🎙️ Iniciar / Detener", on_click=toggle_recording)
