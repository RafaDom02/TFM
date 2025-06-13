import streamlit as st
from audiorecorder import audiorecorder
from faster_whisper import WhisperModel
import os
import io
import uuid
import asyncio
import cv2
from PIL import Image
import numpy as np
from scipy.spatial import distance as dist
from collections import OrderedDict
import time
import requests
import base64
import math  # Para cálculos de ángulos

# Importar componentes de Jarvis
import config
from graph_llm import jarvis_graph_app
import audio_utils

# Configuración del Robot API
ROBOT_API_BASE_URL = "http://192.168.53.200:5000"

# -------------------- TRACKER DE PERSONAS -------------------- #
class CentroidTracker:
    def __init__(self, maxDisappeared=50, 
                 lost_track_timeout_frames=150,
                 hist_comparison_threshold=0.4):
        
        self.nextObjectID = 0
        self.objects = OrderedDict() 
        self.disappeared = OrderedDict()
        self.lost_tracks = OrderedDict() 

        self.maxDisappeared = maxDisappeared 
        self.lost_track_timeout = lost_track_timeout_frames
        self.hist_threshold = hist_comparison_threshold

    def _calculate_histogram(self, roi):
        if roi is None or roi.size == 0:
            return None 
        try:
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
            cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            return hist.flatten()
        except cv2.error:
            return None

    def _purge_lost_tracks(self, current_frame_count):
        ids_to_purge = []
        for objectID, data in self.lost_tracks.items():
            if current_frame_count - data['deregistered_frame'] > self.lost_track_timeout:
                ids_to_purge.append(objectID)
        
        for objectID in ids_to_purge:
            del self.lost_tracks[objectID]

    def register(self, centroid, bbox, histogram):
        self.objects[self.nextObjectID] = {
            'centroid': centroid, 
            'bbox': bbox, 
            'histogram': histogram
        }
        self.disappeared[self.nextObjectID] = 0
        current_id = self.nextObjectID
        self.nextObjectID += 1
        return current_id

    def deregister(self, objectID, current_frame_count):
        if objectID in self.objects:
            self.lost_tracks[objectID] = {
                'centroid': self.objects[objectID]['centroid'],
                'bbox': self.objects[objectID]['bbox'],
                'histogram': self.objects[objectID]['histogram'],
                'deregistered_frame': current_frame_count 
            }
            del self.objects[objectID]
            if objectID in self.disappeared:
                del self.disappeared[objectID]

    def update(self, rects, rois, current_frame_count):
        self._purge_lost_tracks(current_frame_count)

        if len(rects) == 0:
            ids_a_eliminar = []
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    ids_a_eliminar.append(objectID)
            for objectID in ids_a_eliminar:
                 self.deregister(objectID, current_frame_count)
            return self.objects

        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        inputBBoxes = [None] * len(rects)
        inputHistograms = [None] * len(rects)
        valid_indices = []
        
        for i in range(len(rects)):
             startX, startY, endX, endY = rects[i]
             if i < len(rois) and rois[i] is not None and rois[i].size > 0:
                hist = self._calculate_histogram(rois[i])
                if hist is not None:
                     inputHistograms[i] = hist
                     cX = int((startX + endX) / 2.0)
                     cY = int((startY + endY) / 2.0)
                     inputCentroids[i] = (cX, cY)
                     inputBBoxes[i] = rects[i]
                     valid_indices.append(i)

        if len(valid_indices) != len(rects):
             inputCentroids = inputCentroids[valid_indices]
             inputBBoxes = [inputBBoxes[i] for i in valid_indices]
             inputHistograms = [inputHistograms[i] for i in valid_indices]
             if not valid_indices:
                 return self.update([], [], current_frame_count)

        if len(self.objects) == 0:
            for i in range(len(inputCentroids)):
                reidentified_id = self._attempt_reid(inputHistograms[i])
                if reidentified_id != -1:
                     self.objects[reidentified_id] = {
                        'centroid': inputCentroids[i], 
                        'bbox': inputBBoxes[i], 
                        'histogram': inputHistograms[i]
                     }
                     self.disappeared[reidentified_id] = 0
                     del self.lost_tracks[reidentified_id]
                else:
                     self.register(inputCentroids[i], inputBBoxes[i], inputHistograms[i])
            return self.objects

        objectIDs = list(self.objects.keys())
        objectCentroids = [data['centroid'] for data in self.objects.values()]
        
        D = dist.cdist(np.array(objectCentroids), inputCentroids)
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        usedRows = set()
        usedCols = set()

        for (row, col) in zip(rows, cols):
            if row in usedRows or col in usedCols: continue

            objectID = objectIDs[row]
            self.objects[objectID]['centroid'] = inputCentroids[col]
            self.objects[objectID]['bbox'] = inputBBoxes[col]
            self.objects[objectID]['histogram'] = inputHistograms[col]
            self.disappeared[objectID] = 0
            usedRows.add(row)
            usedCols.add(col)

        unusedRows = set(range(len(objectCentroids))).difference(usedRows)
        for row in unusedRows:
            objectID = objectIDs[row]
            self.disappeared[objectID] += 1
            if self.disappeared[objectID] > self.maxDisappeared:
                self.deregister(objectID, current_frame_count)

        unusedCols = set(range(len(inputCentroids))).difference(usedCols)
        
        for col in unusedCols:
             reidentified_id = self._attempt_reid(inputHistograms[col])
             if reidentified_id != -1:
                 self.objects[reidentified_id] = {
                    'centroid': inputCentroids[col], 
                    'bbox': inputBBoxes[col], 
                    'histogram': inputHistograms[col]
                 }
                 self.disappeared[reidentified_id] = 0
                 del self.lost_tracks[reidentified_id]
             else:
                 self.register(inputCentroids[col], inputBBoxes[col], inputHistograms[col])

        return self.objects

    def _attempt_reid(self, new_histogram):
        best_match_id = -1
        min_dist = self.hist_threshold 

        if new_histogram is None:
             return -1

        for lost_id, lost_data in self.lost_tracks.items():
            lost_hist = lost_data.get('histogram')
            if lost_hist is None: continue

            try:
                dist = cv2.compareHist(new_histogram, lost_hist, cv2.HISTCMP_BHATTACHARYYA)
                if dist < min_dist:
                    min_dist = dist
                    best_match_id = lost_id
            except cv2.error:
                continue

        return best_match_id

# Configurar página
st.set_page_config(page_title="Jarvis - Asistente Robótico", layout="centered")
st.title("🤖 Jarvis")
st.caption("Tu asistente robótico para análisis y conversación con grabación de audio.")

# Cargar modelo Whisper local (usando CPU para evitar problemas con CUDA)
@st.cache_resource
def load_whisper_model():
    # Cambiar a CPU para evitar errores de CUDA/cuDNN
    model = WhisperModel("small", device="cpu", compute_type="int8")
    return model

model = load_whisper_model()

# --- Helper Functions ---
def convert_streamlit_history_to_graph_history(st_history):
    """Converts Streamlit message history to graph-compatible history."""
    graph_h = []
    user_msg_content = None
    for msg in list(st_history): 
        if msg["role"] == "user":
            user_msg_content = msg["content"]
        elif msg["role"] == "assistant" and user_msg_content is not None:
            if isinstance(user_msg_content, str):
                 graph_h.append((user_msg_content, msg["content"]))
            user_msg_content = None
        elif msg["role"] == "user_image":
            user_msg_content = None
    return graph_h

def send_tts_to_robot(text: str):
    """Envía texto para TTS al robot API y reproduce localmente como fallback."""
    try:
        # Verificar conexión del robot
        robot_response = requests.get(f"{ROBOT_API_BASE_URL}/robot/audio/status", timeout=5)
        if robot_response.status_code == 200:
            robot_status = robot_response.json()
            if (robot_status.get('speakers') == 'available' and 
                robot_status.get('tts_available', False)):
                
                # Enviar TTS al robot
                tts_response = requests.post(
                    f"{ROBOT_API_BASE_URL}/robot/audio/play",
                    json={"text": text, "voice": "es"},
                    timeout=30
                )
                
                if tts_response.status_code == 200:
                    data = tts_response.json()
                    if data.get('status') == 'success':
                        print(f"[DEBUG] TTS enviado al robot exitosamente")
                        st.info("🔊 Respuesta reproducida en el robot")
                        return True
                    else:
                        print(f"[DEBUG] Error TTS robot: {data.get('error')}")
                else:
                    print(f"[DEBUG] Error código TTS robot: {tts_response.status_code}")
        else:
            print(f"[DEBUG] Robot no disponible para TTS: {robot_response.status_code}")
    except Exception as e:
        print(f"[DEBUG] Excepción en TTS robot: {e}")
    
    # Fallback: reproducir localmente
    try:
        asyncio.run(audio_utils.tts_play(text))
        print(f"[DEBUG] TTS local ejecutado como fallback")
    except Exception as e:
        print(f"[DEBUG] Error en TTS local: {e}")
    
    return False

def capture_image_auto() -> tuple[str, Image.Image] | tuple[None, None]:
    """Captures image from robot API and returns (file_path, image_object)."""
    try:
        # Hacer llamada al robot API para capturar imagen
        response = requests.post(f"{ROBOT_API_BASE_URL}/robot/camera/capture", timeout=10)
        
        if response.status_code != 200:
            print(f"Error: Robot API respondió con código {response.status_code}")
            st.warning(f"Jarvis no pudo obtener imagen del robot. Código: {response.status_code}")
            return None, None
        
        data = response.json()
        
        if data.get('status') != 'success':
            print(f"Error: {data.get('error', 'Error desconocido')}")
            st.warning(f"Error del robot: {data.get('error', 'Error desconocido')}")
            return None, None
        
        # Decodificar imagen base64
        img_base64 = data.get('image_data')
        if not img_base64:
            print("Error: No se recibió imagen del robot")
            st.warning("No se recibió imagen del robot")
            return None, None
        
        # Convertir base64 a imagen
        img_bytes = base64.b64decode(img_base64)
        pil_image = Image.open(io.BytesIO(img_bytes))
        
        # Guardar imagen temporal
        os.makedirs("temp_images", exist_ok=True)
        path = f"temp_images/robot_capture_{uuid.uuid4()}.jpg"
        pil_image.save(path)
        
        print(f"Imagen del robot guardada en: {path}")
        return path, pil_image
        
    except requests.exceptions.ConnectionError:
        print("Error: No se pudo conectar con el robot API")
        st.error("❌ No se pudo conectar con el robot. Verifica que el robot API esté ejecutándose en el puerto 5000.")
        return None, None
    except requests.exceptions.Timeout:
        print("Error: Timeout al conectar con el robot API")
        st.warning("⏱️ El robot tardó demasiado en responder.")
        return None, None
    except Exception as e:
        print(f"Error en capture_image_auto: {e}")
        st.error(f"Error inesperado al obtener imagen del robot: {e}")
        return None, None

def cleanup_video_state():
    """Limpia el estado relacionado con el modo video."""
    # Eliminar todas las variables de estado del video
    video_keys = [
        "video_mode_active", "video_frame_count", "video_tracker", 
        "robot_api_connected", "vision_client", "last_tracked_people",
        "video_should_run", "frame_width", "frame_height"
    ]
    for key in video_keys:
        if key in st.session_state:
            del st.session_state[key]

def get_direction_emoji(centroid, frame_width, frame_height):
    """Calcula el emoji de dirección basado en la posición del centroide.
    Además, envía un comando al robot para girar y apuntar hacia la persona
    usando un control proporcional sencillo del error horizontal.
    """
    if centroid is None:
        return "🎯"  # Centro por defecto si no hay centroide
    
    x, _ = centroid
    
    # Desviación horizontal normalizada (-1 a 1)
    dx_norm = (x - frame_width / 2) / (frame_width / 2)
    
    # Control proporcional simple → ángulo máximo de 30°
    max_angle_deg = 30.0
    angle_deg = dx_norm * max_angle_deg
    
    # Umbral para evitar micro‐ajustes
    angle_threshold_deg = 3.0  # No girar si desviación < 3° aprox.
    
    if abs(angle_deg) > angle_threshold_deg:
        # Calcular velocidad angular (rad/s) para un giro suave de ~0.5 s
        duration = 0.5  # segundos
        angular_velocity = math.radians(angle_deg) / duration  # rad/s
        # Limitar velocidad angular máxima
        max_w = 1.0  # rad/s
        angular_velocity = max(-max_w, min(max_w, angular_velocity))

        # Rate-limit para no saturar al robot (máx. una orden cada 0.3 s)
        now = time.time()
        last_turn = st.session_state.get("_last_robot_turn_ts", 0)
        if now - last_turn > 0.3:
            try:
                requests.post(
                    f"{ROBOT_API_BASE_URL}/robot/move",
                    json={
                        "movement": "custom",
                        "velocity": 0.0,
                        "angular_velocity": -angular_velocity,
                        "duration": duration,
                    },
                    timeout=0.2,
                )
                st.session_state._last_robot_turn_ts = now
            except Exception as e:
                print(f"[WARN] No se pudo enviar giro al robot: {e}")
    
    # --- Mapear a emoji de dirección (visual) ---
    third_width = frame_width // 3
    hard_width = frame_width // 6
    if x < hard_width:
        horizontal = "hard_left"
    elif x > frame_width - hard_width:
        horizontal = "hard_right"
    elif x < third_width:
        horizontal = "left"
    elif x < 2 * third_width:
        horizontal = "center"
    else:
        horizontal = "right"
    
    direction_map = {
        ("hard_left"): "↙️​",
        ("left"): "↖️",
        ("center"): "⬆️",
        ("right"): "↗️",
        ("hard_right"): "↘️",
    }
    return direction_map.get((horizontal), "🎯")

def run_video_detection_mode():
    """Ejecuta el modo de detección de personas en video usando el robot API."""
    # Inicializar estado del video si no existe
    if "video_mode_active" not in st.session_state:
        st.session_state.video_mode_active = False
        st.session_state.video_frame_count = 0
        st.session_state.video_tracker = None
        st.session_state.robot_api_connected = False
        st.session_state.video_should_run = False
        st.session_state.frame_width = 640
        st.session_state.frame_height = 480
        st.session_state.last_tracked_people = {}
    
    # Importar Google Cloud Vision
    try:
        from google.cloud import vision
        if "vision_client" not in st.session_state:
            st.session_state.vision_client = vision.ImageAnnotatorClient()
    except Exception as e:
        st.error(f"Error al inicializar Google Cloud Vision: {e}")
        st.info("Asegúrate de que las credenciales estén configuradas correctamente.")
        # Volver al chat en caso de error
        st.session_state.in_video_mode = False
        cleanup_video_state()
        st.rerun()
        return

    # Verificar conexión con robot API
    try:
        response = requests.get(f"{ROBOT_API_BASE_URL}/robot/camera/status", timeout=5)
        if response.status_code == 200:
            st.session_state.robot_api_connected = True
        else:
            st.session_state.robot_api_connected = False
    except:
        st.session_state.robot_api_connected = False

    # Inicializar tracker si no existe
    if st.session_state.video_tracker is None:
        st.session_state.video_tracker = CentroidTracker(
            maxDisappeared=50,
            lost_track_timeout_frames=150,
            hist_comparison_threshold=0.4
        )

    # Crear contenedores para el video y controles
    st.markdown("## 🎥 Modo de detección de personas (Robot API)")
    
    # Mostrar estado de conexión del robot
    if st.session_state.robot_api_connected:
        st.success("🟢 Conectado al robot API")
    else:
        st.error("🔴 Robot API no disponible")
        st.info("💡 Asegúrate de que el robot API esté ejecutándose en el puerto 5000")
    
    # Controles superiores
    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
    
    with col1:
        start_disabled = not st.session_state.robot_api_connected
        if st.button("▶️ Iniciar", key="start_video_button", disabled=start_disabled):
            st.session_state.video_should_run = True
            st.session_state.video_mode_active = True
    
    with col2:
        if st.button("⏸️ Pausar", key="pause_video_button"):
            st.session_state.video_should_run = False
            st.session_state.video_mode_active = False
    
    with col3:
        if st.button("⏹ Detener", key="stop_video_button"):
            st.session_state.video_should_run = False
            st.session_state.video_mode_active = False
            st.success("Video detenido")
    
    with col4:
        if st.button("💬 Volver al Chat", key="back_to_chat_button", type="primary"):
            # Volver al modo chat normal
            st.session_state.in_video_mode = False
            cleanup_video_state()
            st.rerun()
            return
    
    with col5:
        # Control de intervalo de API
        api_interval = st.slider("Frames/API", min_value=1, max_value=50, value=1, key="api_interval")
    
    # Información del modo
    if st.checkbox("ℹ️ Ayuda", key="show_help"):
        st.info("""
        **Controles:**
        - ▶️ **Iniciar**: Activa la detección (requiere robot API conectado)
        - ⏸️ **Pausar**: Pausa la detección  
        - ⏹ **Detener**: Para completamente el video
        - 💬 **Volver al Chat**: Regresa al chat
        - **Frames/API**: Frecuencia de llamadas (mayor = más eficiente)
        
        **Fuente de video:**
        - 📡 Las imágenes provienen del **robot API** (puerto 5000)
        - 🎥 Cámara del robot remoto en tiempo real
        
        **Visualización:**
        - 🟢 **Cajas verdes**: Personas detectadas
        - 🟢 **Puntos verdes**: Centroides
        - 🔴 **IDs rojos**: Identificadores únicos
        - 🟡 **Línea amarilla**: Dirección hacia persona seguida (ID más bajo)
        - 🟡 **Punto amarillo**: Centro de la cámara
        
        **Seguimiento direccional:**
        - Sigue automáticamente a la persona con ID más bajo
        - Muestra emoji de dirección: ↖️⬆️↗️⬅️🎯➡️↙️⬇️↘️
        - Línea amarilla conecta centro con persona seguida
        """)
    
    # Mostrar estado actual
    status_col1, status_col2, status_col3, status_col4 = st.columns(4)
    with status_col1:
        status_placeholder = st.empty()
    with status_col2:
        frames_placeholder = st.empty()
    with status_col3:
        people_placeholder = st.empty()
    with status_col4:
        direction_placeholder = st.empty()

    # Inicializar métricas
    status_placeholder.metric("Estado", "⏸️ Pausado")
    frames_placeholder.metric("Frames", 0)
    people_placeholder.metric("Personas", 0)
    direction_placeholder.metric("Dirección", "🎯")
    
    # Contenedor para el video - FIJO, no se recarga
    video_placeholder = st.empty()
    info_placeholder = st.empty()
    
    # Solo iniciar el procesamiento si se debe ejecutar
    if st.session_state.video_should_run:
        # Verificar que tengamos conexión con el robot API
        if not st.session_state.robot_api_connected:
            st.error("❌ No se puede iniciar sin conexión al robot API")
            st.session_state.video_should_run = False
            return

        # Configuración NMS
        nms_confidence_threshold = 0.4
        nms_overlap_threshold = 0.3
        
        # BUCLE DE VIDEO SIN ST.RERUN
        max_frames = 100  # Procesar máximo 100 frames antes de pausar
        frames_processed = 0
        
        while st.session_state.video_should_run and frames_processed < max_frames:
            try:
                # Obtener frame del robot API
                response = requests.post(
                    f"{ROBOT_API_BASE_URL}/robot/camera/frame", 
                    json={"return_image": True},
                    timeout=5
                )
                
                if response.status_code != 200:
                    info_placeholder.error(f"Error del robot API: {response.status_code}")
                    break
                
                data = response.json()
                if data.get('status') != 'success':
                    info_placeholder.error(f"Error: {data.get('error', 'Error desconocido')}")
                    break
                
                # Decodificar imagen del robot
                img_base64 = data.get('image_data')
                if not img_base64:
                    info_placeholder.error("No se recibió imagen del robot")
                    break
                
                # Convertir base64 a frame OpenCV
                img_bytes = base64.b64decode(img_base64)
                img_array = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                
                if frame is None:
                    info_placeholder.error("Error al decodificar frame del robot")
                    break
                
                st.session_state.video_frame_count += 1
                frames_processed += 1
                (h, w) = frame.shape[:2]
                
                # Guardar dimensiones del frame en session state
                st.session_state.frame_width = w
                st.session_state.frame_height = h
                
                # Solo llamar a la API cada N frames para reducir costos
                if st.session_state.video_frame_count % api_interval == 0:
                    boxes_for_nms = []
                    confidences_for_nms = []
                    original_rects_pre_nms = []
                    
                    # Codificar frame para API
                    ret2, buf = cv2.imencode('.jpg', frame)
                    if ret2:
                        content = buf.tobytes()
                        image = vision.Image(content=content)
                        
                        try:
                            response = st.session_state.vision_client.object_localization(image=image)
                            objects_detected = response.localized_object_annotations
                            if response.error.message:
                                objects_detected = []
                        except Exception:
                            objects_detected = []
                        
                        # Procesar detecciones de personas
                        for obj in objects_detected:
                            if obj.name.lower() == 'person' and obj.score >= nms_confidence_threshold:
                                verts = obj.bounding_poly.normalized_vertices
                                pts_norm = [(max(0.0, min(1.0, v.x)), max(0.0, min(1.0, v.y))) for v in verts]
                                pts = [(int(v[0] * w), int(v[1] * h)) for v in pts_norm]
                                x_coords = [p[0] for p in pts]
                                y_coords = [p[1] for p in pts]
                                x_min, x_max = min(x_coords), max(x_coords)
                                y_min, y_max = min(y_coords), max(y_coords)
                                x_min, y_min = max(0, x_min), max(0, y_min)
                                x_max, y_max = min(w - 1, x_max), min(h - 1, y_max)
                                
                                if x_max > x_min and y_max > y_min:
                                    boxes_for_nms.append([x_min, y_min, x_max - x_min, y_max - y_min])
                                    confidences_for_nms.append(float(obj.score))
                                    original_rects_pre_nms.append((x_min, y_min, x_max, y_max))
                        
                        # Aplicar NMS solo si hay detecciones
                        if boxes_for_nms:
                            indices_to_keep = cv2.dnn.NMSBoxes(
                                boxes_for_nms, confidences_for_nms,
                                nms_confidence_threshold, nms_overlap_threshold
                            )
                            
                            # Preparar rects y ROIs para el tracker
                            rects_after_nms = []
                            rois_after_nms = []
                            
                            if len(indices_to_keep) > 0:
                                if isinstance(indices_to_keep, np.ndarray) and indices_to_keep.ndim > 1:
                                    indices_to_keep = indices_to_keep.flatten()
                                
                                for i in indices_to_keep:
                                    if 0 <= i < len(original_rects_pre_nms):
                                        (x_min, y_min, x_max, y_max) = original_rects_pre_nms[i]
                                        roi_y_min, roi_y_max = max(0, y_min), min(h, y_max)
                                        roi_x_min, roi_x_max = max(0, x_min), min(w, x_max)
                                        
                                        if roi_y_max > roi_y_min and roi_x_max > roi_x_min:
                                            roi = frame[roi_y_min:roi_y_max, roi_x_min:roi_x_max]
                                            if roi.size > 0:
                                                rects_after_nms.append(original_rects_pre_nms[i])
                                                rois_after_nms.append(roi)
                            
                            # Actualizar tracker
                            st.session_state.last_tracked_people = st.session_state.video_tracker.update(
                                rects_after_nms, rois_after_nms, st.session_state.video_frame_count
                            )
                        else:
                            # No hay detecciones, actualizar tracker con listas vacías
                            st.session_state.last_tracked_people = st.session_state.video_tracker.update(
                                [], [], st.session_state.video_frame_count
                            )
                
                # Dibujar resultados del último tracking
                tracked_people = getattr(st.session_state, 'last_tracked_people', {})
                
                # Crear una copia del frame para dibujar
                display_frame = frame.copy()
                
                # Dibujar bounding boxes y IDs
                for objectID, data in tracked_people.items():
                    centroid = data['centroid']
                    (startX, startY, endX, endY) = data['bbox']
                    
                    # Dibujar bounding box
                    cv2.rectangle(display_frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    
                    # Dibujar centroide
                    cv2.circle(display_frame, centroid, 4, (0, 255, 0), -1)
                    
                    # Dibujar ID
                    text = f"ID {objectID}"
                    text_y = startY - 10 if startY - 10 > 10 else startY + 15
                    cv2.putText(display_frame, text, (startX, text_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # Agregar información en el frame
                info_text = f"Robot API | Personas: {len(tracked_people)} | Frame: {st.session_state.video_frame_count}"
                cv2.putText(display_frame, info_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Añadir indicador de dirección en el frame
                if tracked_people:
                    min_id = min(tracked_people.keys())
                    centroid = tracked_people[min_id]['centroid']
                    direction = get_direction_emoji(centroid, w, h)
                    
                    # Dibujar dirección en la esquina superior derecha
                    direction_text = f"Siguiendo ID {min_id}: {direction}"
                    cv2.putText(display_frame, direction_text, (w - 300, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
                    # Dibujar una línea desde el centro hacia el centroide de la persona seguida
                    center_x, center_y = w // 2, h // 2
                    cv2.line(display_frame, (center_x, center_y), centroid, (0, 255, 255), 2)
                    cv2.circle(display_frame, (center_x, center_y), 5, (0, 255, 255), -1)
                
                # Convertir BGR a RGB para mostrar en Streamlit
                frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                
                # Actualizar métricas en tiempo real
                status = "🟢 Activo" if st.session_state.video_mode_active else "⏸️ Pausado"
                status_placeholder.metric("Estado", status)
                frames_placeholder.metric("Frames", st.session_state.video_frame_count)
                people_placeholder.metric("Personas", len(tracked_people))
                
                # Actualizar dirección en tiempo real
                if tracked_people:
                    min_id = min(tracked_people.keys())
                    centroid = tracked_people[min_id]['centroid']
                    direction = get_direction_emoji(centroid, w, h)
                    direction_placeholder.metric(f"Siguiendo ID {min_id}", direction)
                else:
                    direction_placeholder.metric("Dirección", "🎯")
                
                # Actualizar SOLO el placeholder del video - SIN RECARGAR LA PÁGINA
                video_placeholder.image(frame_rgb, channels="RGB")
                
                # Pequeña pausa entre frames
                time.sleep(0.05)  # 20 FPS aproximadamente
                    
            except Exception as e:
                info_placeholder.error(f"Error: {e}")
                st.session_state.video_should_run = False
                break
        
        # Después del bucle, si aún debe correr, programar reinicio suave
        if st.session_state.video_should_run:
            time.sleep(0.1)
            st.rerun()  # Solo aquí, después de procesar varios frames
    
    else:
        # Mostrar estado cuando no está corriendo
        # Actualizar métricas cuando no está corriendo
        status = "🟢 Activo" if st.session_state.video_mode_active else "⏸️ Pausado"
        status_placeholder.metric("Estado", status)
        frames_placeholder.metric("Frames", getattr(st.session_state, 'video_frame_count', 0))
        
        tracked_people = getattr(st.session_state, 'last_tracked_people', {})
        people_placeholder.metric("Personas", len(tracked_people))
        
        if tracked_people:
            min_id = min(tracked_people.keys())
            centroid = tracked_people[min_id]['centroid']
            frame_width = getattr(st.session_state, 'frame_width', 640)
            frame_height = getattr(st.session_state, 'frame_height', 480)
            direction = get_direction_emoji(centroid, frame_width, frame_height)
            direction_placeholder.metric(f"Siguiendo ID {min_id}", direction)
        else:
            direction_placeholder.metric("Dirección", "🎯")
        
        if st.session_state.video_mode_active:
            video_placeholder.info("🎥 Video pausado. Presiona ▶️ Iniciar para reanudar.")
        else:
            if st.session_state.robot_api_connected:
                video_placeholder.info("🎬 Presiona ▶️ Iniciar para comenzar la detección de personas desde el robot.")
                # Auto-iniciar la primera vez si hay conexión
                if st.session_state.video_frame_count == 0:
                    st.session_state.video_should_run = True
                    st.session_state.video_mode_active = True
                    st.rerun()
            else:
                video_placeholder.warning("🤖 Esperando conexión con el robot API para iniciar detección...")
                video_placeholder.info("💡 Asegúrate de que `robot_flask_api.py` esté ejecutándose en el puerto 5000")

def process_prompt(user_text_prompt: str):
    """Maneja la lógica completa dado un prompt de texto del usuario."""
    if not user_text_prompt:
        return

    # Protección contra re-ejecución automática
    if "processing_in_progress" in st.session_state and st.session_state.processing_in_progress:
        print("[DEBUG] Procesamiento en progreso, evitando re-ejecución automática")
        return
    
    # Marcar que el procesamiento está en progreso
    st.session_state.processing_in_progress = True
    
    # DEBUG: Logging detallado
    print(f"[DEBUG] ==> INICIANDO process_prompt con entrada: '{user_text_prompt}'")
    print(f"[DEBUG] ==> Historial actual tiene {len(st.session_state.history)} mensajes")

    try:
        # Añadir a historial como mensaje de usuario
        st.session_state.history.append({"role": "user", "content": user_text_prompt})
        with st.chat_message("user"):
            st.markdown(user_text_prompt)

        # Crear estado inicial solo para clasificación
        initial_graph_state = {
            "user_input": user_text_prompt,
            "history": convert_streamlit_history_to_graph_history(st.session_state.history[:-1]),
            "image_path": None,
            "classification": None,
            "vision_analysis": None,
            "final_response": None,
            "error": None,
        }

        # Solo obtener la clasificación primero
        with st.spinner("Jarvis está analizando tu solicitud..."):
            try:
                classification_state = st.session_state.graph_app.invoke(initial_graph_state)
            except Exception as e:
                st.error(f"Error al procesar la solicitud: {e}")
                return

        # DEBUG: Verificar qué clasificación estamos recibiendo
        classification = classification_state.get("classification")
        print(f"[DEBUG] Clasificación recibida: '{classification}' (tipo: {type(classification)})")
        
        # Limpiar espacios en blanco y convertir a minúsculas para comparación segura
        if classification:
            classification = str(classification).strip().lower()
            print(f"[DEBUG] Clasificación normalizada: '{classification}'")

        # VERIFICAR MODO VIDEO PRIMERO - activar directamente sin más procesamiento
        if classification == "video":
            print("[DEBUG] ¡Activando modo video directamente!")
            
            # Marcar que estamos en modo video en session state
            st.session_state.in_video_mode = True
            
            # Añadir mensaje al historial
            video_message = "He activado el modo de detección de personas. Estoy analizando el video en tiempo real."
            st.session_state.history.append({"role": "assistant", "content": video_message})
            with st.chat_message("assistant"):
                st.markdown(video_message)
            
            # Limpiar el estado del video anterior si existe
            cleanup_video_state()
            
            # Forzar re-run para mostrar la interfaz de video
            st.rerun()
            return

        elif classification == "describe_image":
            print("[DEBUG] Activando modo describe_image")
            st.info("Jarvis necesita ver. Capturando imagen...")
            img_path, img_object = capture_image_auto()

            if img_path and img_object:
                st.session_state.history.append({"role": "user_image", "content": img_object})
                with st.chat_message("user"):
                    st.image(img_object, caption="Imagen capturada")

                # Crear un historial limpio solo para el análisis de imagen actual
                # No incluir conversaciones anteriores que podrían interferir
                clean_history = []  # Historial vacío para evitar interferencias de conversaciones previas
                
                second_state = {
                    "user_input": user_text_prompt,
                    "history": clean_history,  # Usar historial limpio en lugar del historial completo
                    "image_path": img_path,
                    "classification": "describe_image",
                    "vision_analysis": None,
                    "final_response": None,
                    "error": None,
                }

                with st.spinner("Jarvis está analizando la imagen..."):
                    second_final = st.session_state.graph_app.invoke(second_state)

                ai_img_resp = second_final.get("final_response", "No pude describir la imagen.")
                st.session_state.history.append({"role": "assistant", "content": ai_img_resp})
                with st.chat_message("assistant"):
                    st.markdown(ai_img_resp)

                # Limpieza de archivo temporal
                if os.path.exists(img_path):
                    try:
                        os.remove(img_path)
                    except Exception as e:
                        print(f"Error al eliminar imagen temporal: {e}")

                send_tts_to_robot(ai_img_resp)
        else:
            # Para casos que no son video ni describe_image - continuar con el procesamiento normal
            print(f"[DEBUG] Procesando como respuesta normal. Clasificación: '{classification}'")
            
            # Solo ahora obtener la respuesta completa para otros casos
            if classification_state.get("final_response") is None:
                with st.spinner("Jarvis está generando la respuesta..."):
                    final_state = st.session_state.graph_app.invoke(classification_state)
            else:
                final_state = classification_state
                
            ai_resp = final_state.get("final_response", "Lo siento, no pude procesar tu solicitud.")
            st.session_state.history.append({"role": "assistant", "content": ai_resp})
            with st.chat_message("assistant"):
                st.markdown(ai_resp)
            send_tts_to_robot(ai_resp)
    finally:
        # Desmarcar que el procesamiento está en progreso
        st.session_state.processing_in_progress = False

# Initialize session state variables
if "history" not in st.session_state:
    st.session_state.history = []
if "graph_app" not in st.session_state:
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS") and hasattr(config, 'CREDENTIALS_FILE_PATH'):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = config.CREDENTIALS_FILE_PATH
    st.session_state.graph_app = jarvis_graph_app
if "in_video_mode" not in st.session_state:
    st.session_state.in_video_mode = False

# Si estamos en modo video, mostrar interfaz de video en lugar del chat
if st.session_state.get("in_video_mode", False):
    run_video_detection_mode()
    st.stop()  # Detener la ejecución del resto de la aplicación

# Display chat messages from history
for message in st.session_state.history:
    with st.chat_message(message["role"] if message["role"] != "user_image" else "user"):
        if message["role"] == "user_image":
            if isinstance(message["content"], str):
                if os.path.exists(message["content"]):
                    st.image(message["content"], caption="Imagen")
                else:
                    st.warning("Imagen ya no disponible")
            else:
                st.image(message["content"], caption="Imagen")
        else:
            st.markdown(message["content"])

# Chat input manual
user_text_prompt = st.chat_input("Escribe tu mensaje a Jarvis...")
if user_text_prompt:
    print(f"[DEBUG] ==> process_prompt llamado desde CHAT INPUT con: '{user_text_prompt}'")
    process_prompt(user_text_prompt)

# Separador
st.divider()

# Grabación de audio desde robot API
st.subheader("🎤 Entrada por voz desde Robot")

# Verificar estado del audio del robot
try:
    audio_status_response = requests.get(f"{ROBOT_API_BASE_URL}/robot/audio/status", timeout=5)
    if audio_status_response.status_code == 200:
        audio_status = audio_status_response.json()
        robot_audio_available = (
            audio_status.get('microphone') == 'available' and
            audio_status.get('speakers') == 'available'
        )
        if robot_audio_available:
            st.success("🟢 Sistema de audio del robot disponible")
        else:
            st.warning("🟡 Sistema de audio del robot limitado")
            st.caption(f"Micrófono: {audio_status.get('microphone', 'unknown')}, Altavoces: {audio_status.get('speakers', 'unknown')}")
    else:
        robot_audio_available = False
        st.error("🔴 No se pudo verificar el sistema de audio del robot")
except:
    robot_audio_available = False
    st.error("🔴 Robot API no disponible para audio")

# Controles de audio
col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    duration = st.slider("Duración (segundos)", min_value=1, max_value=10, value=5)

with col2:
    if st.button("🎙️ Grabar desde Robot", disabled=not robot_audio_available):
        if robot_audio_available:
            with st.spinner(f"Grabando audio del robot por {duration} segundos..."):
                try:
                    # Hacer llamada al robot API para grabar audio
                    response = requests.post(
                        f"{ROBOT_API_BASE_URL}/robot/audio/capture",
                        json={"duration": duration, "sample_rate": 16000},
                        timeout=duration + 10
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        if data.get('status') == 'success':
                            # Decodificar audio del robot
                            audio_base64 = data.get('audio_data')
                            if audio_base64:
                                audio_bytes = base64.b64decode(audio_base64)
                                
                                # Mostrar reproductor de audio
                                st.audio(audio_bytes, format="audio/wav")
                                st.success(f"✅ Audio grabado desde el robot ({data.get('size_bytes', 0)} bytes)")
                                
                                # Verificar si ya hemos procesado este audio para evitar re-ejecuciones
                                audio_hash = hash(audio_bytes)
                                if "processed_robot_audio_hashes" not in st.session_state:
                                    st.session_state.processed_robot_audio_hashes = set()
                                
                                if audio_hash not in st.session_state.processed_robot_audio_hashes:
                                    # Marcar este audio como procesado
                                    st.session_state.processed_robot_audio_hashes.add(audio_hash)
                                    
                                    # Guardar audio temporal para transcripción
                                    temp_file = "temp_robot_audio.wav"
                                    with open(temp_file, 'wb') as f:
                                        f.write(audio_bytes)
                                    
                                    # Transcribir
                                    with st.spinner("Transcribiendo audio del robot..."):
                                        try:
                                            segments, info = model.transcribe(temp_file, beam_size=5, language="es")
                                            transcription = "".join([seg.text for seg in segments]).strip()
                                            
                                            if transcription:
                                                st.success(f"Transcripción: {transcription}")
                                                # Procesar la transcripción como si fuera un prompt normal
                                                print(f"[DEBUG] ==> process_prompt llamado desde ROBOT AUDIO con: '{transcription}'")
                                                process_prompt(transcription)
                                            else:
                                                st.warning("No se detectó voz en la grabación del robot.")
                                                
                                        except Exception as e:
                                            st.error(f"Error al transcribir audio del robot: {e}")
                                        finally:
                                            if os.path.exists(temp_file):
                                                os.remove(temp_file)
                            else:
                                st.error("No se recibió audio del robot")
                        else:
                            st.error(f"Error del robot: {data.get('error', 'Error desconocido')}")
                    else:
                        st.error(f"Error del robot API: {response.status_code}")
                        
                except requests.exceptions.Timeout:
                    st.error("⏱️ Timeout: El robot tardó demasiado en grabar")
                except requests.exceptions.ConnectionError:
                    st.error("❌ No se pudo conectar con el robot para grabar audio")
                except Exception as e:
                    st.error(f"Error inesperado: {e}")
        else:
            st.error("Sistema de audio del robot no disponible")

with col3:
    if st.button("🔄", help="Actualizar estado"):
        st.rerun()

# Separador para audio local (fallback)
st.divider()
st.subheader("🎤 Entrada por voz local (fallback)")
st.caption("Usar solo si el robot no está disponible")

audio = audiorecorder("🎙️ Grabar Local", "🔴 Grabando...")

if audio:
    # Convertir AudioSegment a bytes para st.audio()
    audio_bytes = io.BytesIO()
    audio.export(audio_bytes, format="wav")
    audio_bytes.seek(0)
    st.audio(audio_bytes.getvalue(), format="audio/wav")

    # Verificar si ya hemos procesado este audio para evitar re-ejecuciones
    audio_hash = hash(audio_bytes.getvalue())
    if "processed_audio_hashes" not in st.session_state:
        st.session_state.processed_audio_hashes = set()
    
    if audio_hash in st.session_state.processed_audio_hashes:
        print(f"[DEBUG] Audio local ya procesado (hash: {audio_hash}), saltando...")
    else:
        # Marcar este audio como procesado
        st.session_state.processed_audio_hashes.add(audio_hash)
        
        # Guardar audio temporal
        temp_file = "temp_audio.wav"
        audio.export(temp_file, format="wav")

        # Transcribir
        with st.spinner("Transcribiendo..."):
            try:
                segments, info = model.transcribe(temp_file, beam_size=5, language="es")
                transcription = "".join([seg.text for seg in segments]).strip()
                
                if transcription:
                    st.success(f"Transcripción: {transcription}")
                    # Procesar la transcripción como si fuera un prompt normal
                    print(f"[DEBUG] ==> process_prompt llamado desde AUDIO LOCAL con: '{transcription}'")
                    process_prompt(transcription)
                else:
                    st.warning("No se detectó voz en la grabación.")
                    
            except Exception as e:
                st.error(f"Error al transcribir: {e}")
            finally:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
