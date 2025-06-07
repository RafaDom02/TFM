#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema de Monitorización Externa del Robot
===========================================

Este sistema recibe audio del robot, lo procesa con Whisper, 
usa graph_llm.py para tomar decisiones y envía comandos al robot_flask_api.py

Funcionalidades:
- Recepción de audio vía endpoint HTTP
- Transcripción con Whisper
- Análisis y toma de decisiones con LLM
- Envío de comandos de movimiento al robot
- Síntesis de voz para respuestas del robot
- Monitorización en tiempo real
"""

from flask import Flask, request, jsonify, render_template_string
from faster_whisper import WhisperModel
import os
import io
import uuid
import requests
import json
import time
import threading
import queue
import logging
from datetime import datetime
from typing import Dict, Any, Optional
import tempfile

# Importar componentes del sistema Jarvis
import config
from graph_llm import jarvis_graph_app
import audio_utils

# Nuevas importaciones para manejo de imágenes y detección
import cv2
from PIL import Image
import numpy as np
from scipy.spatial import distance as dist
from collections import OrderedDict

# Importar Google Cloud Vision para detección de personas
try:
    from google.cloud import vision
    VISION_AVAILABLE = True
except ImportError:
    VISION_AVAILABLE = False
    logger.warning("Google Cloud Vision no disponible")

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('robot_monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuración
ROBOT_API_BASE_URL = "http://localhost:5000"  # URL del robot_flask_api.py
WHISPER_MODEL_SIZE = "small"  # Modelo Whisper a usar
MAX_AUDIO_QUEUE_SIZE = 10  # Máximo de audios en cola
PROCESSING_TIMEOUT = 30  # Timeout para procesamiento en segundos
AUDIO_POLL_DURATION = 5  # Segundos de grabación por captura
AUDIO_POLL_INTERVAL = 1  # Tiempo de espera entre capturas

# Flags y manejadores para captura de audio automática
audio_capture_active = False
audio_capture_thread = None
audio_capture_stop_event = threading.Event()

# Inicializar Flask
app = Flask(__name__)

# Cola para procesar audios de forma asíncrona
audio_queue = queue.Queue(maxsize=MAX_AUDIO_QUEUE_SIZE)
processing_stats = {
    "total_audios_received": 0,
    "total_audios_processed": 0,
    "total_commands_sent": 0,
    "last_audio_time": None,
    "last_command_time": None,
    "system_status": "waiting",
    "current_processing": None
}

# Historial de conversación
conversation_history = []

# Cargar modelo Whisper
@app.before_first_request
def initialize_services():
    """Inicializar servicios al arrancar la aplicación."""
    global whisper_model
    logger.info("Inicializando modelo Whisper...")
    whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
    logger.info("Modelo Whisper cargado correctamente")
    
    # Configurar credenciales de Google Cloud si es necesario
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS") and hasattr(config, 'CREDENTIALS_FILE_PATH'):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = config.CREDENTIALS_FILE_PATH
        logger.info("Credenciales de Google Cloud configuradas")
    
    logger.info("Sistema de monitorización iniciado y listo")

def convert_history_to_graph_format(history):
    """Convierte el historial de conversación al formato esperado por graph_llm."""
    graph_history = []
    for entry in history:
        if entry.get("user_input") and entry.get("ai_response"):
            graph_history.append((entry["user_input"], entry["ai_response"]))
    return graph_history

def send_robot_command(command_type: str, **kwargs) -> bool:
    """
    Envía un comando al robot a través de la API Flask.
    
    Args:
        command_type: Tipo de comando ('move', 'goal', 'stop')
        **kwargs: Parámetros adicionales para el comando
    
    Returns:
        bool: True si el comando se envió correctamente
    """
    try:
        if command_type == "move":
            url = f"{ROBOT_API_BASE_URL}/robot/move"
            data = kwargs
        elif command_type == "goal":
            url = f"{ROBOT_API_BASE_URL}/robot/goal"
            data = kwargs
        elif command_type == "stop":
            url = f"{ROBOT_API_BASE_URL}/robot/stop"
            data = {}
        else:
            logger.error(f"Tipo de comando desconocido: {command_type}")
            return False
        
        headers = {'Content-Type': 'application/json'}
        response = requests.post(url, json=data, headers=headers, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"Comando enviado correctamente: {command_type} - {result.get('message', 'Sin mensaje')}")
            processing_stats["total_commands_sent"] += 1
            processing_stats["last_command_time"] = datetime.now().isoformat()
            return True
        else:
            logger.error(f"Error al enviar comando: {response.status_code} - {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Error de conexión al enviar comando al robot: {e}")
        return False
    except Exception as e:
        logger.error(f"Error inesperado al enviar comando: {e}")
        return False

def check_robot_api_status() -> bool:
    """Verifica si la API del robot está disponible."""
    try:
        response = requests.get(f"{ROBOT_API_BASE_URL}/robot/status", timeout=5)
        return response.status_code == 200
    except:
        return False

def capture_image_auto() -> tuple[str, Image.Image] | tuple[None, None]:
    """Captura imagen con webcam del robot y devuelve (ruta_archivo, objeto_imagen)."""
    try:
        # Solicitar captura al robot remoto
        logger.info("Solicitando captura de imagen al robot...")
        response = requests.post(
            f"{ROBOT_API_BASE_URL}/robot/camera/capture",
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        
        if response.status_code != 200:
            logger.error(f"Error al solicitar captura al robot: {response.status_code} - {response.text}")
            return None, None
        
        result = response.json()
        if result.get('status') != 'success':
            logger.error(f"Robot reportó error en captura: {result.get('error', 'Error desconocido')}")
            return None, None
        
        # Decodificar imagen base64 del robot
        import base64
        img_base64 = result.get('image_data')
        if not img_base64:
            logger.error("Robot no devolvió datos de imagen")
            return None, None
        
        img_data = base64.b64decode(img_base64)
        
        # Crear imagen PIL desde datos
        pil_image = Image.open(io.BytesIO(img_data))
        
        # Guardar archivo local para procesamiento
        os.makedirs("temp_images", exist_ok=True)
        path = f"temp_images/robot_capture_{uuid.uuid4()}.jpg"
        pil_image.save(path, 'JPEG')
        
        logger.info(f"Imagen del robot recibida y guardada en: {path}")
        logger.info(f"Dimensiones: {result.get('width')}x{result.get('height')}")
        
        return path, pil_image
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Error de conexión al solicitar captura al robot: {e}")
        return None, None
    except Exception as e:
        logger.error(f"Error en capture_image_auto: {e}")
        return None, None

def get_direction_emoji(centroid, frame_width, frame_height):
    """Calcula el emoji de dirección basado en la posición del centroide."""
    if centroid is None:
        return "🎯"  # Centro por defecto si no hay centroide
    
    x, _ = centroid
    
    # Dividir la pantalla en secciones para direccionamiento
    third_width = frame_width // 3
    hard_width = frame_width // 6
    
    # Determinar posición horizontal
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
    
    # Mapear combinaciones a emojis
    direction_map = {
        ("hard_left"): "↙️​",
        ("left"): "↖️",
        ("center"): "⬆️", 
        ("right"): "↗️",
        ("hard_right"): "↘️"
    }
    
    return direction_map.get((horizontal), "🎯")

def convert_direction_to_movement_command(centroid, frame_width, frame_height):
    """Convierte la posición del centroide en comandos de movimiento del robot."""
    if centroid is None:
        return None
    
    x, _ = centroid
    
    # Dividir la pantalla en zonas
    third_width = frame_width // 3
    hard_width = frame_width // 6
    
    # Determinar comando de movimiento
    if x < hard_width:
        # Muy a la izquierda - giro fuerte izquierda
        return {"command_type": "move", "movement": "spin_left", "angular_velocity": 0.8, "duration": 1.0}
    elif x > frame_width - hard_width:
        # Muy a la derecha - giro fuerte derecha
        return {"command_type": "move", "movement": "spin_right", "angular_velocity": 0.8, "duration": 1.0}
    elif x < third_width:
        # Izquierda - avance diagonal izquierda
        return {"command_type": "move", "movement": "forward_left", "velocity": 0.3, "angle": 30, "duration": 1.5}
    elif x < 2 * third_width:
        # Centro - avance directo
        return {"command_type": "move", "movement": "forward", "velocity": 0.3, "duration": 1.5}
    else:
        # Derecha - avance diagonal derecha
        return {"command_type": "move", "movement": "forward_right", "velocity": 0.3, "angle": 30, "duration": 1.5}

def detect_people_and_get_direction():
    """
    Detecta personas usando Google Cloud Vision con imagen de la cámara del robot.
    El robot solo envía la imagen, el monitor hace el procesamiento pesado.
    """
    if not VISION_AVAILABLE:
        logger.error("Google Cloud Vision no está disponible para detección de personas")
        return None, "Vision API no disponible"
    
    try:
        logger.info("🚨 Solicitando imagen al robot para detección de personas...")
        
        # Solicitar imagen al robot (SIN procesamiento)
        payload = {
            "return_image": True  # Necesitamos la imagen para procesarla aquí
        }
        
        response = requests.post(
            f"{ROBOT_API_BASE_URL}/robot/camera/detect_people",
            json=payload,
            headers={'Content-Type': 'application/json'},
            timeout=15
        )
        
        if response.status_code != 200:
            error_msg = f"Error al solicitar imagen al robot: {response.status_code} - {response.text}"
            logger.error(error_msg)
            return None, error_msg
        
        result = response.json()
        
        if result.get('status') != 'success':
            error_msg = f"Robot reportó error en captura: {result.get('error', 'Error desconocido')}"
            logger.error(error_msg)
            return None, error_msg
        
        # Obtener imagen del robot
        img_base64 = result.get('image_data')
        if not img_base64:
            error_msg = "Robot no devolvió datos de imagen"
            logger.error(error_msg)
            return None, error_msg
        
        frame_width = result.get('frame_width', 640)
        frame_height = result.get('frame_height', 480)
        
        logger.info(f"📷 Imagen recibida del robot: {frame_width}x{frame_height}")
        
        # Decodificar imagen para procesamiento con Google Cloud Vision
        import base64
        img_data = base64.b64decode(img_base64)
        
        # Inicializar cliente de Vision API (procesamiento pesado en el monitor)
        client = vision.ImageAnnotatorClient()
        image = vision.Image(content=img_data)
        
        logger.info("🔍 Procesando con Google Cloud Vision (detección pesada)...")
        
        # Detectar objetos con Google Cloud Vision
        response_vision = client.object_localization(image=image)
        objects_detected = response_vision.localized_object_annotations
        
        if response_vision.error.message:
            error_msg = f"Error en Vision API: {response_vision.error.message}"
            logger.error(error_msg)
            return None, error_msg
        
        # Filtrar solo personas
        people_detected = []
        for obj in objects_detected:
            if obj.name.lower() == 'person' and obj.score >= 0.4:
                verts = obj.bounding_poly.normalized_vertices
                pts_norm = [(max(0.0, min(1.0, v.x)), max(0.0, min(1.0, v.y))) for v in verts]
                pts = [(int(v[0] * frame_width), int(v[1] * frame_height)) for v in pts_norm]
                x_coords = [p[0] for p in pts]
                y_coords = [p[1] for p in pts]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                
                # Calcular centroide
                centroid = (int((x_min + x_max) / 2), int((y_min + y_max) / 2))
                people_detected.append({
                    'centroid': centroid,
                    'bbox': (x_min, y_min, x_max, y_max),
                    'score': obj.score
                })
        
        logger.info(f"📊 Detección completada - {len(people_detected)} personas encontradas")
        
        if not people_detected:
            logger.info("❌ No se detectaron supervivientes")
            return None, "No hay personas detectadas"
        
        # Seleccionar la persona más cercana al centro (superviviente prioritario)
        center_x = frame_width // 2
        best_person = min(people_detected, key=lambda p: abs(p['centroid'][0] - center_x))
        
        # Generar comando de movimiento hacia la persona (procesamiento en monitor)
        movement_command = convert_direction_to_movement_command(
            best_person['centroid'], frame_width, frame_height
        )
        
        # Información adicional
        direction_emoji = get_direction_emoji(best_person['centroid'], frame_width, frame_height)
        
        info_msg = f"Superviviente detectado en posición {best_person['centroid']} - Dirigiéndose {direction_emoji} | Confianza: {best_person['score']:.2f}"
        logger.info(f"✅ {info_msg}")
        
        return movement_command, info_msg
        
    except requests.exceptions.RequestException as e:
        error_msg = f"Error de conexión al solicitar imagen al robot: {e}"
        logger.error(error_msg)
        return None, error_msg
    except Exception as e:
        error_msg = f"Error en detect_people_and_get_direction: {e}"
        logger.error(error_msg)
        return None, error_msg

def extract_movement_commands(llm_response: str) -> list:
    """
    Extrae comandos de movimiento de la respuesta del LLM.
    
    Args:
        llm_response: Respuesta del LLM
        
    Returns:
        list: Lista de comandos de movimiento extraídos
    """
    commands = []
    response_lower = llm_response.lower()
    
    # Mapeo de frases a comandos
    movement_mappings = {
        # Movimientos básicos
        "muévete hacia adelante": {"command_type": "move", "movement": "forward"},
        "ve hacia adelante": {"command_type": "move", "movement": "forward"},
        "avanza": {"command_type": "move", "movement": "forward"},
        "muévete hacia atrás": {"command_type": "move", "movement": "backward"},
        "retrocede": {"command_type": "move", "movement": "backward"},
        "ve hacia atrás": {"command_type": "move", "movement": "backward"},
        
        # Giros
        "gira a la derecha": {"command_type": "move", "movement": "spin_right"},
        "gira hacia la derecha": {"command_type": "move", "movement": "spin_right"},
        "gira a la izquierda": {"command_type": "move", "movement": "spin_left"},
        "gira hacia la izquierda": {"command_type": "move", "movement": "spin_left"},
        
        # Movimientos diagonales
        "muévete en diagonal derecha": {"command_type": "move", "movement": "forward_right"},
        "muévete en diagonal izquierda": {"command_type": "move", "movement": "forward_left"},
        
        # Control
        "detente": {"command_type": "stop"},
        "para": {"command_type": "stop"},
        "stop": {"command_type": "stop"},
        
        # Velocidades específicas
        "muévete lento": {"command_type": "move", "movement": "forward", "velocity": 0.2, "duration": 3.0},
        "muévete rápido": {"command_type": "move", "movement": "forward", "velocity": 0.6, "duration": 2.0},
    }
    
    # Buscar comandos en la respuesta
    for phrase, command in movement_mappings.items():
        if phrase in response_lower:
            commands.append(command)
            logger.info(f"Comando detectado: {phrase} -> {command}")
    
    return commands

def process_audio_with_llm(transcription: str) -> Dict[str, Any]:
    """
    Procesa la transcripción con el LLM y determina las acciones a tomar.
    Maneja tres tipos de clasificación: normal, describe_image, y video.
    
    Args:
        transcription: Texto transcrito del audio
        
    Returns:
        dict: Resultado del procesamiento con respuesta y comandos
    """
    try:
        logger.info(f"Procesando transcripción con LLM: '{transcription}'")
        
        # Preparar estado inicial para clasificación
        initial_graph_state = {
            "user_input": transcription,
            "history": convert_history_to_graph_format(conversation_history),
            "image_path": None,
            "classification": None,
            "vision_analysis": None,
            "final_response": None,
            "error": None,
        }
        
        # Obtener clasificación del LLM
        classification_state = jarvis_graph_app.invoke(initial_graph_state)
        classification = classification_state.get("classification", "normal")
        
        # Limpiar y normalizar clasificación
        if classification:
            classification = str(classification).strip().lower()
        
        logger.info(f"Clasificación del LLM: '{classification}'")
        
        # Variables para el resultado
        ai_response = ""
        movement_commands = []
        additional_info = ""
        
        # MANEJAR CLASIFICACIÓN "VIDEO" - Detección de supervivientes
        if classification == "video":
            logger.info("🚨 MODO RESCATE ACTIVADO - Buscando supervivientes")
            ai_response = "He activado el modo de rescate. Buscando supervivientes en el área."
            
            # Detectar personas y obtener comandos de movimiento
            movement_command, detection_info = detect_people_and_get_direction()
            
            if movement_command:
                movement_commands = [movement_command]
                ai_response += f" {detection_info}"
                additional_info = detection_info
            else:
                ai_response += f" {detection_info if detection_info else 'No se detectaron supervivientes.'}"
                # Comando para girar y buscar
                movement_commands = [{"command_type": "move", "movement": "spin_right", "angular_velocity": 0.5, "duration": 2.0}]
                additional_info = "Rotando para buscar supervivientes en otras direcciones"
        
        # MANEJAR CLASIFICACIÓN "DESCRIBE_IMAGE" - Captura y análisis de imagen
        elif classification == "describe_image":
            logger.info("📷 MODO ANÁLISIS VISUAL ACTIVADO - Capturando imagen")
            ai_response = "Voy a analizar lo que veo. Capturando imagen..."
            
            # Capturar imagen automáticamente
            img_path, img_object = capture_image_auto()
            
            if img_path and img_object:
                # Procesar imagen con el grafo LLM
                image_state = {
                    "user_input": transcription,
                    "history": convert_history_to_graph_format(conversation_history),
                    "image_path": img_path,
                    "classification": "describe_image",
                    "vision_analysis": None,
                    "final_response": None,
                    "error": None,
                }
                
                # Obtener descripción de la imagen
                final_state = jarvis_graph_app.invoke(image_state)
                ai_response = final_state.get("final_response", "No pude analizar la imagen correctamente.")
                
                # Limpiar archivo temporal
                if os.path.exists(img_path):
                    try:
                        os.remove(img_path)
                    except Exception as e:
                        logger.error(f"Error al eliminar imagen temporal: {e}")
                
                additional_info = f"Imagen analizada correctamente desde {img_path}"
            else:
                ai_response = "No pude acceder a la cámara para capturar una imagen."
                additional_info = "Error al capturar imagen"
        
        # MANEJAR CLASIFICACIÓN "NORMAL" - Respuesta conversacional estándar
        else:
            logger.info("💬 MODO CONVERSACIÓN NORMAL")
            
            # Obtener respuesta completa si no se ha generado aún
            if classification_state.get("final_response") is None:
                final_state = jarvis_graph_app.invoke(classification_state)
            else:
                final_state = classification_state
            
            ai_response = final_state.get("final_response", "Lo siento, no pude procesar tu solicitud.")
            
            # Extraer comandos de movimiento de la respuesta (funcionalidad existente)
            movement_commands = extract_movement_commands(ai_response)
            additional_info = f"Respuesta conversacional normal con {len(movement_commands)} comandos"
        
        # Agregar al historial
        conversation_entry = {
            "timestamp": datetime.now().isoformat(),
            "user_input": transcription,
            "ai_response": ai_response,
            "classification": classification,
            "commands_extracted": movement_commands,
            "additional_info": additional_info
        }
        conversation_history.append(conversation_entry)
        
        # Mantener solo los últimos 20 intercambios
        if len(conversation_history) > 20:
            conversation_history.pop(0)
        
        logger.info(f"Procesamiento completado - Clasificación: {classification}, Comandos: {len(movement_commands)}")
        
        return {
            "success": True,
            "transcription": transcription,
            "ai_response": ai_response,
            "classification": classification,
            "movement_commands": movement_commands,
            "conversation_entry": conversation_entry,
            "additional_info": additional_info
        }
        
    except Exception as e:
        logger.error(f"Error al procesar con LLM: {e}")
        return {
            "success": False,
            "error": str(e),
            "transcription": transcription,
            "classification": "error",
            "movement_commands": [],
            "additional_info": f"Error en procesamiento: {str(e)}"
        }

def audio_processor_worker():
    """Worker thread que procesa audios de la cola de forma asíncrona."""
    logger.info("Worker de procesamiento de audio iniciado")
    
    while True:
        try:
            # Obtener audio de la cola (bloquea hasta que haya uno)
            audio_data, timestamp = audio_queue.get(timeout=1)
            
            processing_stats["system_status"] = "processing"
            processing_stats["current_processing"] = timestamp
            
            logger.info(f"Procesando audio recibido en: {timestamp}")
            
            # Guardar audio temporal
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_audio_path = temp_file.name
            
            try:
                # Transcribir con Whisper
                logger.info("Transcribiendo audio...")
                segments, info = whisper_model.transcribe(
                    temp_audio_path, 
                    beam_size=5, 
                    language="es"
                ) 
                
                transcription = "".join([seg.text for seg in segments]).strip()
                
                if transcription:
                    logger.info(f"Transcripción obtenida: '{transcription}'")
                    
                    # Procesar con LLM
                    llm_result = process_audio_with_llm(transcription)
                    
                    if llm_result["success"]:
                        ai_response = llm_result["ai_response"]
                        movement_commands = llm_result["movement_commands"]
                        classification = llm_result.get("classification", "normal")
                        additional_info = llm_result.get("additional_info", "")
                        
                        logger.info(f"🤖 Resultado del procesamiento:")
                        logger.info(f"   Clasificación: {classification}")
                        logger.info(f"   Respuesta: {ai_response}")
                        logger.info(f"   Comandos: {len(movement_commands)}")
                        logger.info(f"   Info adicional: {additional_info}")
                        
                        # Ejecutar comandos de movimiento según la clasificación
                        if movement_commands:
                            if classification == "video":
                                logger.info(f"🚨 EJECUTANDO COMANDOS DE RESCATE: {len(movement_commands)} comandos")
                            elif classification == "describe_image":
                                logger.info(f"📷 EJECUTANDO COMANDOS DE ANÁLISIS VISUAL: {len(movement_commands)} comandos")
                            else:
                                logger.info(f"💬 EJECUTANDO COMANDOS CONVERSACIONALES: {len(movement_commands)} comandos")
                            
                            for i, cmd in enumerate(movement_commands):
                                command_type = cmd.pop("command_type")
                                success = send_robot_command(command_type, **cmd)
                                if success:
                                    logger.info(f"✅ Comando {i+1}/{len(movement_commands)} ejecutado: {command_type}")
                                else:
                                    logger.error(f"❌ Falló comando {i+1}/{len(movement_commands)}: {command_type}")
                                
                                # Pausa entre comandos (más larga para comandos de rescate)
                                if classification == "video":
                                    time.sleep(1.0)  # Pausa más larga para rescate
                                else:
                                    time.sleep(0.5)  # Pausa normal
                        else:
                            logger.info("ℹ️  No se generaron comandos de movimiento")
                        
                        # Generar respuesta de voz para el robot (TTS)
                        if ai_response and len(ai_response.strip()) > 0:
                            try:
                                logger.info(f"🔊 Generando respuesta de audio: '{ai_response[:50]}...'")
                                
                                # Intentar usar audio_utils para TTS
                                try:
                                    import asyncio
                                    asyncio.run(audio_utils.tts_play(ai_response))
                                    logger.info("✅ Respuesta de audio reproducida correctamente")
                                except Exception as tts_error:
                                    logger.warning(f"⚠️  Error en TTS: {tts_error}")
                                    # TODO: Implementar envío de audio TTS al robot físico
                                    logger.info(f"📢 Robot debería decir: '{ai_response}'")
                                
                            except Exception as e:
                                logger.error(f"❌ Error generando respuesta de audio: {e}")
                        else:
                            logger.info("🔇 No hay respuesta de audio que generar")
                        
                        # Log especial para modo rescate
                        if classification == "video":
                            logger.info("🚨 MODO RESCATE COMPLETADO - Robot dirigiéndose hacia superviviente")
                        elif classification == "describe_image":
                            logger.info("📷 ANÁLISIS VISUAL COMPLETADO - Descripción generada")
                    
                    else:
                        logger.error(f"❌ Error en procesamiento LLM: {llm_result.get('error')}")
                        # En caso de error, intentar respuesta básica
                        try:
                            error_response = "Lo siento, tuve un problema al procesar tu solicitud."
                            logger.info(f"🔊 Enviando respuesta de error: '{error_response}'")
                            import asyncio
                            asyncio.run(audio_utils.tts_play(error_response))
                        except:
                            pass
                
                else:
                    logger.warning("No se detectó voz en el audio recibido")
                
            finally:
                # Limpiar archivo temporal
                if os.path.exists(temp_audio_path):
                    os.remove(temp_audio_path)
            
            processing_stats["total_audios_processed"] += 1
            processing_stats["system_status"] = "waiting"
            processing_stats["current_processing"] = None
            
            # Marcar tarea como completada
            audio_queue.task_done()
            
        except queue.Empty:
            # No hay audio en cola, continuar
            processing_stats["system_status"] = "waiting"
            continue
        except Exception as e:
            logger.error(f"Error en worker de procesamiento: {e}")
            processing_stats["system_status"] = "error"
            # Marcar tarea como completada aunque haya error
            try:
                audio_queue.task_done()
            except:
                pass

# Iniciar worker thread
worker_thread = threading.Thread(target=audio_processor_worker, daemon=True)
worker_thread.start()

# ================================
# ENDPOINTS HTTP
# ================================

@app.route('/robot/audio', methods=['POST'])
def receive_audio():
    """
    Endpoint para recibir audio del robot.
    
    Expects:
        - Multipart form with 'audio' file
        - Or raw audio data in request body
    
    Returns:
        JSON response with status
    """
    try:
        timestamp = datetime.now().isoformat()
        
        # Verificar si hay datos de audio
        audio_data = None
        
        if 'audio' in request.files:
            # Audio enviado como archivo multipart
            audio_file = request.files['audio']
            audio_data = audio_file.read()
            logger.info(f"Audio recibido como archivo: {len(audio_data)} bytes")
        elif request.data:
            # Audio enviado como datos raw
            audio_data = request.data
            logger.info(f"Audio recibido como datos raw: {len(audio_data)} bytes")
        else:
            return jsonify({
                "error": "No se encontraron datos de audio",
                "timestamp": timestamp
            }), 400
        
        # Verificar que no esté vacío
        if not audio_data or len(audio_data) == 0:
            return jsonify({
                "error": "Datos de audio vacíos",
                "timestamp": timestamp
            }), 400
        
        # Verificar que la cola no esté llena
        if audio_queue.full():
            logger.warning("Cola de audio llena, descartando audio más antiguo")
            try:
                audio_queue.get_nowait()  # Remover audio más antiguo
            except queue.Empty:
                pass
        
        # Añadir a la cola para procesamiento asíncrono
        try:
            audio_queue.put_nowait((audio_data, timestamp))
            processing_stats["total_audios_received"] += 1
            processing_stats["last_audio_time"] = timestamp
            
            logger.info(f"Audio añadido a cola de procesamiento")
            
            return jsonify({
                "status": "success",
                "message": "Audio recibido y añadido a cola de procesamiento",
                "timestamp": timestamp,
                "queue_size": audio_queue.qsize()
            })
            
        except queue.Full:
            logger.error("Cola de audio llena, no se pudo añadir audio")
            return jsonify({
                "error": "Cola de procesamiento llena, intente más tarde",
                "timestamp": timestamp
            }), 503
        
    except Exception as e:
        logger.error(f"Error al recibir audio: {e}")
        return jsonify({
            "error": f"Error interno: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/monitor/status', methods=['GET'])
def get_monitor_status():
    """Obtiene el estado del sistema de monitorización."""
    robot_status = check_robot_api_status()
    
    return jsonify({
        "monitor_status": "active",
        "robot_api_status": "connected" if robot_status else "disconnected",
        "robot_api_url": ROBOT_API_BASE_URL,
        "processing_stats": processing_stats,
        "audio_queue_size": audio_queue.qsize(),
        "conversation_history_length": len(conversation_history),
        "timestamp": datetime.now().isoformat()
    })

@app.route('/monitor/history', methods=['GET'])
def get_conversation_history():
    """Obtiene el historial de conversación."""
    limit = request.args.get('limit', 10, type=int)
    limit = min(limit, 50)  # Máximo 50 entradas
    
    recent_history = conversation_history[-limit:] if conversation_history else []
    
    return jsonify({
        "history": recent_history,
        "total_entries": len(conversation_history),
        "returned_entries": len(recent_history),
        "timestamp": datetime.now().isoformat()
    })

@app.route('/monitor/test_robot', methods=['POST'])
def test_robot_connection():
    """Prueba la conexión con el robot enviando un comando de prueba."""
    try:
        # Enviar comando de estado
        response = requests.get(f"{ROBOT_API_BASE_URL}/robot/status", timeout=5)
        
        if response.status_code == 200:
            robot_data = response.json()
            
            # Enviar comando de movimiento corto como prueba
            test_move = send_robot_command("move", movement="forward", duration=0.5)
            
            return jsonify({
                "status": "success",
                "robot_status": robot_data,
                "test_command_sent": test_move,
                "timestamp": datetime.now().isoformat()
            })
        else:
            return jsonify({
                "error": f"Robot API respondió con código: {response.status_code}",
                "timestamp": datetime.now().isoformat()
            }), 503
            
    except Exception as e:
        return jsonify({
            "error": f"No se pudo conectar con el robot: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }), 503

@app.route('/monitor/clear_history', methods=['POST'])
def clear_conversation_history():
    """Limpia el historial de conversación."""
    global conversation_history
    conversation_history.clear()
    
    logger.info("Historial de conversación limpiado")
    
    return jsonify({
        "status": "success",
        "message": "Historial de conversación limpiado",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/', methods=['GET'])
def dashboard():
    """Dashboard web para monitorizar el sistema."""
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Monitor del Robot - Dashboard</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; }
            .card { background: white; padding: 20px; margin: 10px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .status-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; }
            .metric { text-align: center; padding: 15px; }
            .metric-value { font-size: 2em; font-weight: bold; color: #2196F3; }
            .metric-label { color: #666; margin-top: 5px; }
            .status-ok { color: #4CAF50; }
            .status-error { color: #f44336; }
            .status-warning { color: #ff9800; }
            h1, h2 { color: #333; }
            .refresh-btn { background: #2196F3; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; }
            .refresh-btn:hover { background: #1976D2; }
            .history-item { border-left: 4px solid #2196F3; padding: 10px; margin: 10px 0; background: #f9f9f9; }
            .history-item.video { border-left-color: #f44336; background: #fff3f3; }
            .history-item.describe_image { border-left-color: #ff9800; background: #fff8f0; }
            .history-item.normal { border-left-color: #4CAF50; background: #f8fff8; }
            .timestamp { color: #666; font-size: 0.9em; }
            .command { background: #e3f2fd; padding: 5px 10px; border-radius: 4px; margin: 5px 0; }
            .classification-badge { 
                display: inline-block; 
                padding: 4px 8px; 
                border-radius: 12px; 
                font-size: 0.8em; 
                font-weight: bold; 
                color: white; 
            }
            .classification-video { background-color: #f44336; }
            .classification-describe_image { background-color: #ff9800; }
            .classification-normal { background-color: #4CAF50; }
            .classification-error { background-color: #9e9e9e; }
            .info-section { 
                margin-top: 10px; 
                padding: 10px; 
                background: rgba(33, 150, 243, 0.1); 
                border-radius: 4px; 
                border-left: 3px solid #2196F3; 
            }
        </style>
        <script>
            function refreshData() {
                fetch('/monitor/status')
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('status-data').innerHTML = JSON.stringify(data, null, 2);
                        updateMetrics(data);
                    });
                
                fetch('/monitor/history?limit=5')
                    .then(response => response.json())
                    .then(data => {
                        updateHistory(data.history);
                    });
            }
            
            function updateMetrics(data) {
                document.getElementById('audios-received').textContent = data.processing_stats.total_audios_received;
                document.getElementById('audios-processed').textContent = data.processing_stats.total_audios_processed;
                document.getElementById('commands-sent').textContent = data.processing_stats.total_commands_sent;
                document.getElementById('queue-size').textContent = data.audio_queue_size;
                
                const robotStatus = document.getElementById('robot-status');
                robotStatus.textContent = data.robot_api_status;
                robotStatus.className = data.robot_api_status === 'connected' ? 'status-ok' : 'status-error';
                
                const systemStatus = document.getElementById('system-status');
                systemStatus.textContent = data.processing_stats.system_status;
                systemStatus.className = data.processing_stats.system_status === 'waiting' ? 'status-ok' : 'status-warning';
            }
            
            function updateHistory(history) {
                const container = document.getElementById('history-container');
                container.innerHTML = '';
                
                history.reverse().forEach(item => {
                    const div = document.createElement('div');
                    div.className = `history-item ${item.classification || 'normal'}`;
                    
                    // Determinar emoji y clase de clasificación
                    let classificationEmoji = '';
                    let classificationClass = '';
                    if (item.classification === 'video') {
                        classificationEmoji = '🚨';
                        classificationClass = 'classification-video';
                    } else if (item.classification === 'describe_image') {
                        classificationEmoji = '📷';
                        classificationClass = 'classification-describe_image';
                    } else if (item.classification === 'error') {
                        classificationEmoji = '❌';
                        classificationClass = 'classification-error';
                    } else {
                        classificationEmoji = '💬';
                        classificationClass = 'classification-normal';
                    }
                    
                    div.innerHTML = `
                        <div class="timestamp">${item.timestamp}</div>
                        <div><strong>Usuario:</strong> ${item.user_input}</div>
                        <div><strong>IA:</strong> ${item.ai_response}</div>
                        <div><strong>Clasificación:</strong> 
                            <span class="classification-badge ${classificationClass}">
                                ${classificationEmoji} ${item.classification}
                            </span>
                        </div>
                        ${item.additional_info ? 
                            '<div class="info-section"><strong>📋 Info adicional:</strong> ' + item.additional_info + '</div>' : ''}
                        ${item.commands_extracted && item.commands_extracted.length > 0 ? 
                            '<div><strong>🤖 Comandos ejecutados:</strong> ' + 
                            item.commands_extracted.map(cmd => `<span class="command">${JSON.stringify(cmd)}</span>`).join(' ') + 
                            '</div>' : ''}
                    `;
                    container.appendChild(div);
                });
            }
            
            function testRobot() {
                fetch('/monitor/test_robot', { method: 'POST' })
                    .then(response => response.json())
                    .then(data => {
                        alert('Prueba completada: ' + JSON.stringify(data, null, 2));
                    })
                    .catch(error => {
                        alert('Error en prueba: ' + error);
                    });
            }
            
            function clearHistory() {
                if (confirm('¿Estás seguro de que quieres limpiar el historial?')) {
                    fetch('/monitor/clear_history', { method: 'POST' })
                        .then(response => response.json())
                        .then(data => {
                            alert('Historial limpiado');
                            refreshData();
                        });
                }
            }
            
            // Auto-refresh cada 5 segundos
            setInterval(refreshData, 5000);
            // Refresh inicial
            window.onload = refreshData;
        </script>
    </head>
    <body>
        <div class="container">
            <h1>🤖 Monitor del Robot - Dashboard</h1>
            
            <div class="card">
                <h2>Estado del Sistema</h2>
                <div class="status-grid">
                    <div class="metric">
                        <div class="metric-value" id="audios-received">-</div>
                        <div class="metric-label">Audios Recibidos</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value" id="audios-processed">-</div>
                        <div class="metric-label">Audios Procesados</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value" id="commands-sent">-</div>
                        <div class="metric-label">Comandos Enviados</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value" id="queue-size">-</div>
                        <div class="metric-label">Cola de Audio</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value status-ok" id="robot-status">-</div>
                        <div class="metric-label">Estado Robot</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value status-ok" id="system-status">-</div>
                        <div class="metric-label">Estado Sistema</div>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <h2>Controles</h2>
                <button class="refresh-btn" onclick="refreshData()">🔄 Actualizar Datos</button>
                <button class="refresh-btn" onclick="testRobot()">🧪 Probar Robot</button>
                <button class="refresh-btn" onclick="clearHistory()">🗑️ Limpiar Historial</button>
            </div>
            
            <div class="card">
                <h2>ℹ️ Modos de Operación del Robot</h2>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px;">
                    <div style="padding: 15px; border-left: 4px solid #4CAF50; background: #f8fff8;">
                        <h3>💬 Modo Conversación Normal</h3>
                        <p><strong>Activación:</strong> Comandos conversacionales generales</p>
                        <p><strong>Función:</strong> Respuesta de voz del robot + comandos de movimiento básicos</p>
                        <p><strong>Ejemplos:</strong> "Muévete hacia adelante", "Gira a la derecha"</p>
                        <p><strong>📹 Cámara:</strong> No se usa</p>
                    </div>
                    <div style="padding: 15px; border-left: 4px solid #ff9800; background: #fff8f0;">
                        <h3>📷 Modo Análisis Visual</h3>
                        <p><strong>Activación:</strong> "Describe lo que ves", "Analiza la imagen"</p>
                        <p><strong>Función:</strong> Captura automática de imagen desde cámara del robot</p>
                        <p><strong>Tecnología:</strong> Google Cloud Vision + LLM</p>
                        <p><strong>📹 Cámara:</strong> Cámara del robot vía endpoint remoto</p>
                    </div>
                    <div style="padding: 15px; border-left: 4px solid #f44336; background: #fff3f3;">
                        <h3>🚨 Modo Rescate/Socorro</h3>
                        <p><strong>Activación:</strong> Señales de socorro, "Ayuda", "Emergencia"</p>
                        <p><strong>Función:</strong> Detección de supervivientes + navegación autónoma</p>
                        <p><strong>Comportamiento:</strong> Robot captura → Monitor detecta con Google Vision → Robot navega</p>
                        <p><strong>📹 Cámara:</strong> Robot captura, Monitor procesa con IA</p>
                        <p><strong>🧠 IA:</strong> Google Cloud Vision (procesamiento pesado en monitor)</p>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <h2>Historial Reciente de Conversación</h2>
                <div id="history-container">
                    <p>Cargando historial...</p>
                </div>
            </div>
            
            <div class="card">
                <h2>Datos de Estado (JSON)</h2>
                <pre id="status-data" style="background: #f0f0f0; padding: 15px; border-radius: 4px; overflow-x: auto;">
                    Cargando datos...
                </pre>
            </div>
        </div>
    </body>
    </html>
    """
    return render_template_string(html_template)

# -------------------- TRACKER DE PERSONAS (copiado de streamlit_app.py) -------------------- #
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

def robot_audio_capture_loop(duration=AUDIO_POLL_DURATION, interval=AUDIO_POLL_INTERVAL, sample_rate=16000):
    """Hilo que solicita audio al robot de forma continua hasta que se detenga."""
    global audio_capture_active
    logger.info(f"▶️ Iniciando captura continua de audio del robot (chunk={duration}s, interval={interval}s)...")
    while not audio_capture_stop_event.is_set():
        try:
            response = requests.post(
                f"{ROBOT_API_BASE_URL}/robot/audio/capture",
                json={"duration": duration, "sample_rate": sample_rate},
                timeout=duration + 10
            )
            timestamp = datetime.now().isoformat()
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success':
                    import base64
                    audio_b64 = data.get('audio_data')
                    if audio_b64:
                        audio_bytes = base64.b64decode(audio_b64)
                        # Enqueue for processing
                        try:
                            if audio_queue.full():
                                audio_queue.get_nowait()
                            audio_queue.put_nowait((audio_bytes, timestamp))
                            processing_stats["total_audios_received"] += 1
                            processing_stats["last_audio_time"] = timestamp
                            logger.info(f"Audio ({len(audio_bytes)} bytes) capturado del robot y añadido a cola")
                        except queue.Full:
                            logger.warning("Cola de audio llena, descartando captura")
                else:
                    logger.error(f"Robot devolvió error en captura: {data.get('error')}")
            else:
                logger.error(f"Error HTTP al capturar audio: {response.status_code}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Error de conexión al capturar audio: {e}")
        except Exception as e:
            logger.error(f"Error inesperado en loop de captura: {e}")
        # Esperar intervalo o salir si se detiene
        if audio_capture_stop_event.wait(interval):
            break
    audio_capture_active = False
    logger.info("⏹️ Captura de audio del robot detenida")

@app.route('/monitor/start_audio_capture', methods=['POST'])
def start_audio_capture():
    """Inicia la captura continua de audio desde el robot."""
    global audio_capture_active, audio_capture_thread, audio_capture_stop_event
    if audio_capture_active:
        return jsonify({"status": "already_running"})
    duration = request.json.get('duration', AUDIO_POLL_DURATION) if request.is_json else AUDIO_POLL_DURATION
    interval = request.json.get('interval', AUDIO_POLL_INTERVAL) if request.is_json else AUDIO_POLL_INTERVAL
    audio_capture_stop_event.clear()
    audio_capture_thread = threading.Thread(
        target=robot_audio_capture_loop,
        kwargs={"duration": duration, "interval": interval},
        daemon=True
    )
    audio_capture_thread.start()
    audio_capture_active = True
    logger.info("Solicitud recibida: iniciar captura de audio del robot")
    return jsonify({"status": "started", "duration": duration, "interval": interval})

@app.route('/monitor/stop_audio_capture', methods=['POST'])
def stop_audio_capture():
    """Detiene la captura continua de audio del robot."""
    global audio_capture_active
    if not audio_capture_active:
        return jsonify({"status": "not_running"})
    audio_capture_stop_event.set()
    logger.info("Solicitud recibida: detener captura de audio del robot")
    return jsonify({"status": "stopping"})

if __name__ == '__main__':
    logger.info("Iniciando Sistema de Monitorización del Robot")
    logger.info(f"Robot API URL: {ROBOT_API_BASE_URL}")
    logger.info(f"Modelo Whisper: {WHISPER_MODEL_SIZE}")
    
    # Verificar conexión con robot al inicio
    if check_robot_api_status():
        logger.info("✅ Conexión con robot API verificada")
    else:
        logger.warning("⚠️  No se pudo conectar con robot API al inicio")
    
    # Inicializar servicios manualmente si no se ha hecho
    try:
        initialize_services()
    except:
        pass  # Ya se inicializó
    
    app.run(
        host='0.0.0.0', 
        port=5001, 
        debug=False,  # Desactivar debug en producción
        threaded=True
    )