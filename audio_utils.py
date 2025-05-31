import os, uuid, asyncio, io, wave, tempfile, numpy as np, sounddevice as sd
import noisereduce as nr
import whisper, edge_tts
from pydub import AudioSegment
import webrtcvad
from collections import deque
import sys

# --- Configuración VAD ---
VAD_SAMPLE_RATE = 16000
VAD_FRAME_MS = 30
VAD_CHUNK_SIZE = (VAD_SAMPLE_RATE * VAD_FRAME_MS) // 1000

# Permitir ajustar fácilmente vía variables de entorno
#   VAD_LEVEL         (0-3)    → agresividad del detector, más estricto
#   VAD_SILENCE       (seg)    → duración del silencio para cortar
#   VAD_MIN_SPEECH    (seg)    → habla mínima a aceptar

VAD_AGGRESSIVENESS = int(os.getenv("VAD_LEVEL", "2"))
SILENCE_DURATION_S  = float(os.getenv("VAD_SILENCE", "0.6"))
MIN_SPEECH_DURATION_S = float(os.getenv("VAD_MIN_SPEECH", "0.4"))

# Un padding pequeño acelera el ciclo
PADDING_S = 0.15  # Cantidad de silencio a añadir antes/después

# --- Constantes ---
# Permite elegir el modelo vía variable de entorno (por defecto, 'small' es más preciso que 'base')
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "small")
print(f"[audio_utils] Cargando modelo Whisper: {WHISPER_MODEL_NAME}")
WHISPER_MODEL = whisper.load_model(WHISPER_MODEL_NAME)
TTS_VOICE = "es-ES-ElviraNeural"
TMP_DIR = tempfile.gettempdir()

# --- Variables Globales para VAD ---
audio_buffer = deque()
recording_complete_event = asyncio.Event()
is_speaking_global = False
speech_frames_count_global = 0
silence_frames_count_global = 0
vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)

def _audio_callback(indata, frames, time_info, status):
    """Callback de sounddevice, procesa audio en tiempo real con VAD."""
    global audio_buffer, is_speaking_global, speech_frames_count_global, silence_frames_count_global, recording_complete_event
    
    if status:
        print(f"Estado del stream de audio: {status}", file=sys.stderr)
    
    # Convertir float32 a int16 para VAD
    clipped_indata = np.clip(indata, -1.0, 1.0)
    indata_int16 = (clipped_indata * 32767).astype(np.int16)
    audio_chunk_bytes = indata_int16.tobytes()
    
    bytes_per_sample = 2  # int16 = 2 bytes
    samples_per_vad_chunk = VAD_CHUNK_SIZE
    bytes_per_vad_chunk = samples_per_vad_chunk * bytes_per_sample
    
    # Procesar en chunks que VAD puede manejar
    for i in range(0, len(audio_chunk_bytes), bytes_per_vad_chunk):
        chunk_vad = audio_chunk_bytes[i : i + bytes_per_vad_chunk]
        
        # Si el chunk no es completo, saltar
        if len(chunk_vad) < bytes_per_vad_chunk:
            continue
        
        # Determinar si es habla o silencio
        try:
            is_speech = vad.is_speech(chunk_vad, VAD_SAMPLE_RATE)
        except Exception:
            is_speech = False
        
        if is_speech:
            # Es habla, resetear contador de silencio
            silence_frames_count_global = 0
            
            if not is_speaking_global:
                # Empezar a grabar
                is_speaking_global = True
                speech_frames_count_global = 1
                
                # Añadir silencio de padding al principio
                padding_frames = int((PADDING_S * VAD_SAMPLE_RATE) / VAD_CHUNK_SIZE)
                for _ in range(padding_frames):
                    audio_buffer.append(b'\x00' * bytes_per_vad_chunk)
            
            # Añadir frame al buffer
            audio_buffer.append(chunk_vad)
            speech_frames_count_global += 1
            
        else:
            # Es silencio, pero ya estábamos hablando
            if is_speaking_global:
                # Seguir grabando el silencio por un tiempo
                silence_frames_count_global += 1
                audio_buffer.append(chunk_vad)
                
                # Verificar si el silencio es suficientemente largo
                silence_threshold_frames = int((SILENCE_DURATION_S * VAD_SAMPLE_RATE) / VAD_CHUNK_SIZE)
                if silence_frames_count_global >= silence_threshold_frames:
                    # Terminamos la grabación
                    is_speaking_global = False
                    
                    # Añadir silencio de padding al final
                    padding_frames = int((PADDING_S * VAD_SAMPLE_RATE) / VAD_CHUNK_SIZE)
                    for _ in range(padding_frames):
                        audio_buffer.append(b'\x00' * bytes_per_vad_chunk)
                    
                    # Verificar duración mínima para considerar válida la grabación
                    min_speech_frames = int((MIN_SPEECH_DURATION_S * VAD_SAMPLE_RATE) / VAD_CHUNK_SIZE)
                    if speech_frames_count_global > min_speech_frames:
                        # Señalar que la grabación está completa
                        try:
                            loop = asyncio.get_event_loop()
                            loop.call_soon_threadsafe(recording_complete_event.set)
                        except RuntimeError:
                            recording_complete_event.set()
                    else:
                        # Descartar por ser muy corta
                        audio_buffer.clear()
                        speech_frames_count_global = 0
                    
                    silence_frames_count_global = 0

async def listen_for_speech(status_callback=None) -> bytes | None:
    """Graba hasta detectar silencio tras habla y devuelve los bytes WAV."""
    # Variables locales como en la solución anterior
    audio_buffer = deque()
    recording_complete_event = asyncio.Event()
    is_speaking = False
    speech_frames = 0
    silence_frames = 0

    def _audio_cb(indata, frames, time_info, status):
        nonlocal audio_buffer, is_speaking, speech_frames, silence_frames
        if status:
            print(status, file=sys.stderr)

        samples = (np.clip(indata, -1, 1) * 32767).astype(np.int16).tobytes()
        chunk = VAD_CHUNK_SIZE * 2  # bytes
        for i in range(0, len(samples), chunk):
            frame = samples[i:i+chunk]
            if len(frame) < chunk:               # incompleto
                continue
            speech = vad.is_speech(frame, VAD_SAMPLE_RATE)
            if speech:
                silence_frames = 0
                if not is_speaking:
                    is_speaking = True
                    speech_frames = 0
                    pad = int(PADDING_S * VAD_SAMPLE_RATE / VAD_CHUNK_SIZE)
                    audio_buffer.extend(b'\x00'*chunk for _ in range(pad))
                audio_buffer.append(frame)
                speech_frames += 1
            else:
                if is_speaking:
                    silence_frames += 1
                    audio_buffer.append(frame)
                    thr = int(SILENCE_DURATION_S * VAD_SAMPLE_RATE / VAD_CHUNK_SIZE)
                    if silence_frames >= thr:
                        is_speaking = False
                        pad = int(PADDING_S * VAD_SAMPLE_RATE / VAD_CHUNK_SIZE)
                        audio_buffer.extend(b'\x00'*chunk for _ in range(pad))
                        min_ok = int(MIN_SPEECH_DURATION_S * VAD_SAMPLE_RATE / VAD_CHUNK_SIZE)
                        if speech_frames > min_ok:
                            recording_complete_event.set()
                        else:
                            audio_buffer.clear()
                        silence_frames = 0

    if status_callback:
        status_callback("🎤 Escuchando…")
    
    loop = asyncio.get_event_loop()
    try:
        with sd.InputStream(samplerate=VAD_SAMPLE_RATE,
                            channels=1,
                            dtype='float32',
                            blocksize=VAD_CHUNK_SIZE,
                            callback=lambda *a: loop.call_soon_threadsafe(_audio_cb,*a)):
            await recording_complete_event.wait()
            
        # Verificar que tenemos audio antes de continuar
        if not audio_buffer:
            if status_callback:
                status_callback("No se detectó audio. Intenta de nuevo.")
            return None
            
        # AQUÍ ESTÁ EL CAMBIO: en lugar de devolver los bytes sin procesar,
        # los convertimos a un formato WAV válido antes de devolverlos
        return save_audio_to_wav_bytes(b''.join(audio_buffer), VAD_SAMPLE_RATE)
        
    except Exception as e:
        print(f"Error al grabar audio: {e}", file=sys.stderr)
        return None

def _wav_bytes_to_np(wav_bytes: bytes):
    with io.BytesIO(wav_bytes) as buf:
        wav_file = wave.open(buf, 'rb')
        sr = wav_file.getframerate()
        data = np.frombuffer(wav_file.readframes(wav_file.getnframes()), dtype=np.int16)
    return sr, data

def save_audio_to_wav_bytes(audio_data: bytes, sample_rate: int) -> bytes:
    """Guarda los datos de audio (bytes, int16) en formato WAV en memoria."""
    bytes_io = io.BytesIO()
    with wave.open(bytes_io, 'wb') as wf:
        wf.setnchannels(1)         # Mono
        wf.setsampwidth(2)         # 16 bits por muestra
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data)
    bytes_io.seek(0)               # Volver al principio del buffer
    return bytes_io.getvalue()

# -------------------- PARÁMETROS DE FILTRO -------------------- #
# Permite al usuario ajustar la tolerancia vía variables de entorno.
# Valores menos estrictos aumentan la probabilidad de aceptar transcripciones
# cortas o con menor confianza.
MIN_TRANSCRIPTION_LENGTH = int(os.getenv("TRANS_MIN_LEN", "2"))
MIN_AVG_LOGPROB_ALLOWED  = float(os.getenv("TRANS_MIN_LOGPROB", "-3.0"))

async def transcribe(wav_bytes: bytes) -> str | None:
    """Transcribe WAV bytes usando Whisper (idioma español por defecto).

    – Aplica reducción de ruido más suave.
    – Filtra resultados poco fiables, pero con umbrales menos estrictos que antes.
    """
    if not wav_bytes:
        return None

    # Convertir bytes WAV a numpy
    with io.BytesIO(wav_bytes) as buf:
        with wave.open(buf, "rb") as w:
            sr = w.getframerate()
            data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)

    data_f32 = data.astype(np.float32) / 32768.0

    # Reducción de ruido (menos agresiva para conservar calidad de voz)
    clean = nr.reduce_noise(y=data_f32, sr=sr, prop_decrease=0.4)

    # --- PARÁMETROS WHISPER -------------------------------------------------
    # Permite cambiar de modelo vía env var WHISPER_MODEL.
    # Se usa beam search para mayor precisión.
    result = WHISPER_MODEL.transcribe(
        clean, language="es", task="transcribe", fp16=False,
        condition_on_previous_text=False, beam_size=5, best_of=5, temperature=0.0
    )

    text_out: str = result["text"].strip()

    # -------------------- FILTRO DE CALIDAD -------------------- #
    # 1) Longitud mínima muy corta (configurable)
    if len(text_out) < MIN_TRANSCRIPTION_LENGTH:
        return None

    # 2) Confianza media de los segmentos
    try:
        import numpy as _np
        if result.get("segments"):
            avg_lp = _np.mean([seg["avg_logprob"] for seg in result["segments"]])
            if avg_lp < MIN_AVG_LOGPROB_ALLOWED:
                return None
    except Exception:
        pass

    return text_out

async def tts_play(text: str):
    """Genera y reproduce audio TTS."""
    if not text:
        return
    
    mp3_path = os.path.join(TMP_DIR, f"jarvis_{uuid.uuid4()}.mp3")
    await edge_tts.Communicate(text, TTS_VOICE).save(mp3_path)
    
    audio = AudioSegment.from_mp3(mp3_path)
    sd.play(np.array(audio.get_array_of_samples()), audio.frame_rate, blocking=True)
    
    os.remove(mp3_path)