#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
API Flask para controlar Robotnik Summit XL-HL.
Compatible con ROS 1 y Python 2.7.

Endpoints disponibles:
  POST /robot/move - Ejecutar movimientos del robot
  POST /robot/goal - Enviar objetivo de navegación  
  POST /robot/stop - Detener el robot
  GET /robot/status - Estado del robot
"""

from __future__ import print_function

import math
import json
import rospy
import tf.transformations as tft
from flask import Flask, request, jsonify
from geometry_msgs.msg import Twist, PoseStamped, Quaternion
import subprocess
import tempfile
import os
import base64
import sys
import collections


# -----------------------------------------------------------------------------
# Compatibilidad Python 2.7
# -----------------------------------------------------------------------------
# 1) Back-port mínimo de subprocess.run (disponible desde Python 3.5).
#    Solo se cubren los argumentos utilizados en este archivo: capture_output,
#    text y timeout.
# 2) Mantener la API original para evitar cambios extensivos en el código más
#    abajo.
if not hasattr(subprocess, 'run'):
    def _subprocess_run(cmd, capture_output=False, text=False, timeout=None):
        """Versión simplificada de subprocess.run para Python 2.7."""
        stdout_pipe = subprocess.PIPE if capture_output else None
        stderr_pipe = subprocess.PIPE if capture_output else None

        proc = subprocess.Popen(cmd, stdout=stdout_pipe, stderr=stderr_pipe)

        try:
            # Python 2.7 no soporta el argumento timeout en communicate.
            out, err = proc.communicate(timeout=timeout)  # type: ignore
        except TypeError:  # pragma: no cover – timeout no soportado
            out, err = proc.communicate()

        if text:
            if out is not None:
                out = out.decode('utf-8', 'ignore')
            if err is not None:
                err = err.decode('utf-8', 'ignore')

        CompletedProcess = collections.namedtuple('CompletedProcess',
                                                 ['returncode', 'stdout', 'stderr'])
        return CompletedProcess(proc.returncode, out, err)

    subprocess.run = _subprocess_run  # type: ignore


# ─────────── utilidades de cámara ───────────
def get_available_cameras():
    """
    Encuentra todas las cámaras disponibles en el sistema.
    
    Returns:
        list: Lista de índices de cámaras disponibles
    """
    available_cameras = []
    try:
        import cv2
        for i in range(5):  # Verificar hasta 5 índices de cámara
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        available_cameras.append(i)
                    cap.release()
                else:
                    cap.release()
            except Exception:
                # Silenciar errores de cámaras individuales
                continue
    except ImportError:
        # OpenCV no disponible
        pass
    return available_cameras


# ─────────── utilidades PoseStamped ───────────
def yaw_deg_to_quat(yaw_deg):
    """Convierte yaw en grados a quaternion (x,y,z,w)."""
    q = tft.quaternion_from_euler(0.0, 0.0, math.radians(yaw_deg))
    return Quaternion(*q)


def make_pose_stamped(x, y, yaw_deg, frame_id="map"):
    """Genera un PoseStamped sellado con la hora actual."""
    pose = PoseStamped()
    pose.header.stamp = rospy.Time.now()
    pose.header.frame_id = frame_id
    pose.pose.position.x = x
    pose.pose.position.y = y
    pose.pose.orientation = yaw_deg_to_quat(yaw_deg)
    return pose


# ─────────── clase principal del robot ───────────
class RobotController(object):
    def __init__(self):
        rospy.init_node("summit_flask_api", anonymous=True)

        self.pub_cmd = rospy.Publisher("/robot/cmd_vel", Twist, queue_size=10)
        self.pub_goal = rospy.Publisher("/robot/move_base_simple/goal",
                                        PoseStamped, queue_size=1)

        # Velocidades por defecto
        self.default_v_lin = 0.3   # m/s
        self.default_v_ang = 0.7   # rad/s
        
        self.is_moving = False

        rospy.loginfo("Robot Controller API iniciado")

    # ――― publicaciones Twist ―――
    def _publish_for(self, lin_x, ang_z, duration):
        """Publica comandos de velocidad durante un tiempo determinado."""
        self.is_moving = True
        twist = Twist()
        twist.linear.x = lin_x
        twist.angular.z = ang_z

        rate = rospy.Rate(10)  # 10 Hz
        end_time = rospy.Time.now() + rospy.Duration(duration)
        while rospy.Time.now() < end_time and not rospy.is_shutdown():
            self.pub_cmd.publish(twist)
            rate.sleep()
        self.stop()

    def _publish_instant(self, lin_x, ang_z):
        """Publica un comando de velocidad instantáneo."""
        twist = Twist()
        twist.linear.x = lin_x
        twist.angular.z = ang_z
        self.pub_cmd.publish(twist)
        self.is_moving = (lin_x != 0.0 or ang_z != 0.0)

    def stop(self):
        """Detiene el robot."""
        self.pub_cmd.publish(Twist())
        self.is_moving = False

    # ――― movimientos con parámetros personalizables ―――
    def forward(self, velocity=None, duration=2.0):
        v = velocity if velocity is not None else self.default_v_lin
        self._publish_for(v, 0.0, duration)

    def backward(self, velocity=None, duration=2.0):
        v = velocity if velocity is not None else self.default_v_lin
        self._publish_for(-v, 0.0, duration)

    def forward_right(self, velocity=None, angle_deg=30, duration=2.0):
        v = velocity if velocity is not None else self.default_v_lin
        self._publish_for(v, -math.radians(angle_deg) / duration, duration)

    def forward_left(self, velocity=None, angle_deg=30, duration=2.0):
        v = velocity if velocity is not None else self.default_v_lin
        self._publish_for(v, math.radians(angle_deg) / duration, duration)

    def spin_right(self, angular_velocity=None, duration=2.0):
        w = angular_velocity if angular_velocity is not None else self.default_v_ang
        self._publish_for(0.0, -w, duration)

    def spin_left(self, angular_velocity=None, duration=2.0):
        w = angular_velocity if angular_velocity is not None else self.default_v_ang
        self._publish_for(0.0, w, duration)

    def custom_move(self, linear_velocity, angular_velocity, duration=2.0):
        """Movimiento personalizado con velocidades específicas."""
        self._publish_for(linear_velocity, angular_velocity, duration)

    # ――― envío de goal ―――
    def send_goal(self, x, y, yaw_deg):
        """Envía un objetivo de navegación."""
        goal = make_pose_stamped(x, y, yaw_deg)
        self.pub_goal.publish(goal)
        rospy.loginfo("Goal enviado → (%.2f, %.2f, %.1f°)", x, y, yaw_deg)


# ─────────── configuración Flask ───────────
app = Flask(__name__)
robot = None


def init_robot():
    """Inicializa el controlador del robot."""
    global robot
    if robot is None:
        robot = RobotController()


@app.route('/robot/move', methods=['POST'])
def move_robot():
    """
    Endpoint para mover el robot.
    
    Parámetros JSON:
    - movement: tipo de movimiento ('forward', 'backward', 'forward_right', 'forward_left', 'spin_right', 'spin_left', 'custom')
    - velocity: velocidad lineal (opcional, default: 0.3 m/s)
    - angular_velocity: velocidad angular (opcional, default: 0.7 rad/s, solo para giros)
    - angle: ángulo de giro en grados (opcional, default: 30°, para movimientos forward_right/left)
    - duration: duración del movimiento en segundos (opcional, default: 2.0s)
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No se proporcionaron datos JSON'}), 400

        movement = data.get('movement')
        velocity = data.get('velocity')
        angular_velocity = data.get('angular_velocity')
        angle = data.get('angle', 30)
        duration = data.get('duration', 2.0)

        if not movement:
            return jsonify({'error': 'Se requiere especificar el tipo de movimiento'}), 400

        init_robot()

        # Ejecutar el movimiento correspondiente
        if movement == 'forward':
            robot.forward(velocity, duration)
        elif movement == 'backward':
            robot.backward(velocity, duration)
        elif movement == 'forward_right':
            robot.forward_right(velocity, angle, duration)
        elif movement == 'forward_left':
            robot.forward_left(velocity, angle, duration)
        elif movement == 'spin_right':
            robot.spin_right(angular_velocity, duration)
        elif movement == 'spin_left':
            robot.spin_left(angular_velocity, duration)
        elif movement == 'custom':
            lin_vel = velocity if velocity is not None else 0.0
            ang_vel = angular_velocity if angular_velocity is not None else 0.0
            robot.custom_move(lin_vel, ang_vel, duration)
        else:
            return jsonify({'error': 'Tipo de movimiento no válido'}), 400

        return jsonify({
            'status': 'success',
            'message': 'Movimiento {} ejecutado'.format(movement),
            'parameters': {
                'movement': movement,
                'velocity': velocity,
                'angular_velocity': angular_velocity,
                'angle': angle,
                'duration': duration
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/robot/goal', methods=['POST'])
def send_goal():
    """
    Endpoint para enviar un objetivo de navegación.
    
    Parámetros JSON:
    - x: coordenada x del objetivo
    - y: coordenada y del objetivo  
    - yaw: orientación final en grados
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No se proporcionaron datos JSON'}), 400

        x = data.get('x')
        y = data.get('y')
        yaw = data.get('yaw', 0.0)

        if x is None or y is None:
            return jsonify({'error': 'Se requieren las coordenadas x e y'}), 400

        init_robot()
        robot.send_goal(float(x), float(y), float(yaw))

        return jsonify({
            'status': 'success',
            'message': 'Objetivo enviado',
            'goal': {'x': x, 'y': y, 'yaw': yaw}
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/robot/stop', methods=['POST'])
def stop_robot():
    """Endpoint para detener el robot."""
    try:
        init_robot()
        robot.stop()
        return jsonify({
            'status': 'success',
            'message': 'Robot detenido'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/robot/status', methods=['GET'])
def robot_status():
    """Endpoint para obtener el estado del robot."""
    try:
        init_robot()
        return jsonify({
            'status': 'success',
            'is_moving': robot.is_moving,
            'default_linear_velocity': robot.default_v_lin,
            'default_angular_velocity': robot.default_v_ang
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/robot/movements', methods=['GET'])
def get_available_movements():
    """Endpoint para obtener la lista de movimientos disponibles."""
    movements = {
        'available_movements': [
            'forward',
            'backward', 
            'forward_right',
            'forward_left',
            'spin_right',
            'spin_left',
            'custom'
        ],
        'parameters': {
            'velocity': 'Velocidad lineal en m/s (opcional)',
            'angular_velocity': 'Velocidad angular en rad/s (opcional)',
            'angle': 'Ángulo de giro en grados (opcional, default: 30°)',
            'duration': 'Duración del movimiento en segundos (opcional, default: 2.0s)'
        },
        'examples': {
            'forward': {'movement': 'forward', 'velocity': 0.5, 'duration': 3.0},
            'spin_right': {'movement': 'spin_right', 'angular_velocity': 1.0, 'duration': 1.5},
            'custom': {'movement': 'custom', 'velocity': 0.3, 'angular_velocity': 0.5, 'duration': 2.0}
        }
    }
    return jsonify(movements)


# ================================
# ENDPOINTS DE CÁMARA DEL ROBOT
# ================================

@app.route('/robot/camera/capture', methods=['POST'])
def capture_image():
    """
    Captura una imagen desde la cámara del robot.
    
    Returns:
        JSON con la imagen en base64 o error
    """
    try:
        import cv2
        import base64
        import uuid
        import os
        
        # Verificar disponibilidad de cámaras
        available_cameras = get_available_cameras()
        
        if not available_cameras:
            return jsonify({
                'error': 'No se encontraron cámaras disponibles en el robot',
                'timestamp': rospy.Time.now().to_sec(),
                'available_cameras': available_cameras
            }), 404
        
        # Usar la primera cámara disponible
        camera_index = available_cameras[0]
        rospy.loginfo("Usando cámara en índice: {}".format(camera_index))
        
        # Intentar abrir la cámara
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            return jsonify({
                'error': 'No se pudo acceder a la cámara del robot en índice {}'.format(camera_index),
                'timestamp': rospy.Time.now().to_sec()
            }), 500
        
        # Capturar frame
        ret, frame = cap.read()
        cap.release()
        
        if not ret or frame is None:
            return jsonify({
                'error': 'No se pudo capturar frame de la cámara',
                'timestamp': rospy.Time.now().to_sec()
            }), 500
        
        # Crear carpeta temporal si no existe (Python 2 no soporta exist_ok)
        if not os.path.exists("temp_images"):
            os.makedirs("temp_images")
        temp_filename = "temp_robot_capture_{}.jpg".format(uuid.uuid4())
        temp_path = os.path.join("temp_images", temp_filename)
        
        success = cv2.imwrite(temp_path, frame)
        if not success:
            return jsonify({
                'error': 'Error al guardar imagen capturada',
                'timestamp': rospy.Time.now().to_sec()
            }), 500
        
        # Codificar imagen en base64 para envío
        with open(temp_path, 'rb') as img_file:
            img_data = img_file.read()
            img_base64 = base64.b64encode(img_data).decode('utf-8')
        
        # Limpiar archivo temporal
        try:
            os.remove(temp_path)
        except:
            pass
        
        # Obtener dimensiones de la imagen
        height, width, channels = frame.shape
        
        rospy.loginfo("Imagen capturada desde cámara del robot: {}x{}".format(width, height))
        
        return jsonify({
            'status': 'success',
            'message': 'Imagen capturada correctamente',
            'image_data': img_base64,
            'image_format': 'jpeg',
            'width': int(width),
            'height': int(height),
            'camera_index': camera_index,
            'timestamp': rospy.Time.now().to_sec(),
            'temp_filename': temp_filename
        })
        
    except Exception as e:
        rospy.logerr("Error capturando imagen: {}".format(e))
        return jsonify({
            'error': 'Error interno al capturar imagen: {}'.format(str(e)),
            'timestamp': rospy.Time.now().to_sec()
        }), 500

@app.route('/robot/camera/detect_people', methods=['POST'])
def detect_people_camera():
    """
    Captura imagen desde la cámara del robot para detección de personas.
    NO hace detección pesada - solo captura y envía la imagen raw al monitor.
    
    Returns:
        JSON con imagen en base64 para procesamiento remoto
    """
    try:
        import cv2
        import base64
        import uuid
        import os
        
        # Parámetros opcionales
        data = request.get_json() or {}
        return_image = data.get('return_image', True)  # Por defecto incluir imagen
        
        # Verificar disponibilidad de cámaras
        available_cameras = get_available_cameras()
        
        if not available_cameras:
            return jsonify({
                'error': 'No se encontraron cámaras disponibles en el robot',
                'timestamp': rospy.Time.now().to_sec()
            }), 404
        
        camera_index = available_cameras[0]
        
        # Capturar frame
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            return jsonify({
                'error': 'No se pudo acceder a la cámara del robot',
                'timestamp': rospy.Time.now().to_sec()
            }), 500
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret or frame is None:
            return jsonify({
                'error': 'No se pudo capturar frame',
                'timestamp': rospy.Time.now().to_sec()
            }), 500
        
        (h, w) = frame.shape[:2]
        
        # Información básica de la captura
        result = {
            'status': 'success',
            'frame_width': w,
            'frame_height': h,
            'camera_index': camera_index,
            'timestamp': rospy.Time.now().to_sec(),
            'message': 'Frame capturado para procesamiento remoto'
        }
        
        # Incluir imagen en base64 para envío al monitor
        if return_image:
            ret2, buf = cv2.imencode('.jpg', frame)
            if ret2:
                img_base64 = base64.b64encode(buf.tobytes()).decode('utf-8')
                result['image_data'] = img_base64
                result['image_format'] = 'jpeg'
                rospy.loginfo("Frame capturado y codificado para detección remota: {}x{}".format(w, h))
            else:
                return jsonify({
                    'error': 'Error al codificar imagen para envío',
                    'timestamp': rospy.Time.now().to_sec()
                }), 500
        
        return jsonify(result)
        
    except Exception as e:
        rospy.logerr("Error en detect_people_camera: {}".format(e))
        return jsonify({
            'error': 'Error interno: {}'.format(str(e)),
            'timestamp': rospy.Time.now().to_sec()
        }), 500

@app.route('/robot/camera/status', methods=['GET'])
def camera_status():
    """Verifica el estado de la cámara del robot."""
    try:
        import cv2
        
        # Usar la función utilitaria para obtener cámaras disponibles
        available_cameras = get_available_cameras()
        
        if not available_cameras:
            return jsonify({
                'status': 'unavailable',
                'message': 'No se puede acceder a ninguna cámara del robot',
                'available_cameras': [],
                'camera_details': [],
                'timestamp': rospy.Time.now().to_sec()
            }), 404
        
        # Obtener detalles de cada cámara disponible
        camera_info = []
        for camera_index in available_cameras:
            try:
                cap = cv2.VideoCapture(camera_index)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        height, width, channels = frame.shape
                        camera_info.append({
                            'index': camera_index,
                            'resolution': {'width': int(width), 'height': int(height)},
                            'channels': int(channels)
                        })
                    cap.release()
            except Exception as e:
                rospy.logwarn("Error obteniendo detalles de cámara {}: {}".format(camera_index, e))
                continue
        
        return jsonify({
            'status': 'available',
            'message': 'Cámaras del robot operativas',
            'available_cameras': available_cameras,
            'camera_details': camera_info,
            'default_camera': available_cameras[0],
            'timestamp': rospy.Time.now().to_sec()
        })
            
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': 'Error verificando cámaras: {}'.format(str(e)),
            'timestamp': rospy.Time.now().to_sec()
        }), 500

@app.route('/robot/camera/diagnose', methods=['GET'])
def diagnose_cameras():
    """
    Endpoint de diagnóstico detallado para problemas de cámara.
    
    Returns:
        JSON con información de diagnóstico completa
    """
    try:
        import cv2
        import os
        import subprocess
        
        diagnosis = {
            'timestamp': rospy.Time.now().to_sec(),
            'opencv_version': cv2.__version__,
            'system_info': {},
            'device_check': {},
            'camera_test': {},
            'recommendations': []
        }
        
        # Verificar dispositivos de video del sistema
        try:
            if os.path.exists('/dev'):
                video_devices = [f for f in os.listdir('/dev') if f.startswith('video')]
                diagnosis['device_check']['video_devices'] = video_devices
                diagnosis['device_check']['video_devices_count'] = len(video_devices)
            else:
                diagnosis['device_check']['video_devices'] = []
                diagnosis['device_check']['video_devices_count'] = 0
        except Exception as e:
            diagnosis['device_check']['error'] = str(e)
        
        # Verificar con v4l2-ctl si está disponible
        try:
            result = subprocess.run(['v4l2-ctl', '--list-devices'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                diagnosis['system_info']['v4l2_devices'] = result.stdout
            else:
                diagnosis['system_info']['v4l2_error'] = result.stderr
        except Exception as e:
            diagnosis['system_info']['v4l2_unavailable'] = str(e)
        
        # Test de cámaras con OpenCV
        available_cameras = get_available_cameras()
        diagnosis['camera_test']['available_cameras'] = available_cameras
        diagnosis['camera_test']['total_available'] = len(available_cameras)
        
        # Test detallado de cada índice
        test_results = []
        for i in range(10):  # Test más índices para diagnóstico
            test_result = {
                'index': i,
                'can_open': False,
                'can_read': False,
                'error': None
            }
            
            try:
                cap = cv2.VideoCapture(i)
                test_result['can_open'] = cap.isOpened()
                
                if cap.isOpened():
                    ret, frame = cap.read()
                    test_result['can_read'] = ret and frame is not None
                    if ret and frame is not None:
                        h, w = frame.shape[:2]
                        test_result['resolution'] = {'width': w, 'height': h}
                cap.release()
                
            except Exception as e:
                test_result['error'] = str(e)
            
            test_results.append(test_result)
        
        diagnosis['camera_test']['detailed_tests'] = test_results
        
        # Generar recomendaciones
        if not available_cameras:
            diagnosis['recommendations'].append("No se encontraron cámaras. Verificar conexión física.")
            if diagnosis['device_check'].get('video_devices_count', 0) == 0:
                diagnosis['recommendations'].append("No hay dispositivos /dev/video*. Verificar drivers de cámara.")
            else:
                diagnosis['recommendations'].append("Dispositivos /dev/video* encontrados pero no accesibles via OpenCV.")
        else:
            diagnosis['recommendations'].append("Cámaras funcionando correctamente en índices: {}".format(available_cameras))
        
        if diagnosis['system_info'].get('v4l2_unavailable'):
            diagnosis['recommendations'].append("Instalar v4l-utils para mejor diagnóstico: sudo apt-get install v4l-utils")
        
        return jsonify(diagnosis)
        
    except Exception as e:
        return jsonify({
            'error': 'Error durante diagnóstico: {}'.format(str(e)),
            'timestamp': rospy.Time.now().to_sec()
        }), 500


# ================================
# ENDPOINTS DE AUDIO DEL ROBOT
# ================================

@app.route('/robot/audio/capture', methods=['POST'])
def capture_audio():
    """
    Captura audio desde el micrófono del robot.
    
    Parámetros JSON (opcionales):
    - duration: duración de la grabación en segundos (default: 5)
    - sample_rate: frecuencia de muestreo (default: 16000)
    
    Returns:
        JSON con el audio en base64
    """
    try:
        data = request.get_json() or {}
        duration = data.get('duration', 5)  # 5 segundos por defecto
        sample_rate = data.get('sample_rate', 16000)  # 16kHz por defecto
        
        # Crear archivo temporal para la grabación
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_path = temp_file.name
        
        # Comando para grabar audio con arecord (ALSA)
        cmd = [
            'arecord',
            '-f', 'S16_LE',  # 16-bit little-endian
            '-c', '1',       # mono
            '-r', str(sample_rate),  # sample rate
            '-t', 'wav',     # formato WAV
            '-d', str(duration),  # duración
            temp_path
        ]
        
        rospy.loginfo("Grabando audio por {} segundos...".format(duration))
        
        # Ejecutar comando de grabación
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            # Limpiar archivo temporal
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return jsonify({
                'error': 'Error al grabar audio: {}'.format(result.stderr),
                'timestamp': rospy.Time.now().to_sec()
            }), 500
        
        # Verificar que se creó el archivo
        if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return jsonify({
                'error': 'No se pudo grabar audio o archivo vacío',
                'timestamp': rospy.Time.now().to_sec()
            }), 500
        
        # Leer y codificar audio en base64
        with open(temp_path, 'rb') as audio_file:
            audio_data = audio_file.read()
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        
        # Limpiar archivo temporal
        os.remove(temp_path)
        
        rospy.loginfo("Audio capturado exitosamente: {} bytes".format(len(audio_data)))
        
        return jsonify({
            'status': 'success',
            'message': 'Audio grabado por {} segundos'.format(duration),
            'audio_data': audio_base64,
            'audio_format': 'wav',
            'duration': duration,
            'sample_rate': sample_rate,
            'size_bytes': len(audio_data),
            'timestamp': rospy.Time.now().to_sec()
        })
        
    except Exception as e:
        rospy.logerr("Error capturando audio: {}".format(e))
        return jsonify({
            'error': 'Error interno al capturar audio: {}'.format(str(e)),
            'timestamp': rospy.Time.now().to_sec()
        }), 500

@app.route('/robot/audio/play', methods=['POST'])
def play_audio():
    """
    Reproduce audio en el robot.
    
    Parámetros JSON:
    - audio_data: datos de audio en base64
    - audio_format: formato del audio (wav, mp3, etc.)
    
    O alternativamente:
    - text: texto para convertir a voz (TTS)
    - voice: voz a usar para TTS (opcional)
    
    Returns:
        JSON con resultado de la reproducción
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No se proporcionaron datos JSON'}), 400
        
        temp_path = None
        
        # Modo 1: Reproducir audio desde base64
        if 'audio_data' in data:
            audio_base64 = data.get('audio_data')
            audio_format = data.get('audio_format', 'wav')
            
            if not audio_base64:
                return jsonify({'error': 'audio_data no puede estar vacío'}), 400
            
            # Decodificar audio
            try:
                audio_bytes = base64.b64decode(audio_base64)
            except Exception as e:
                return jsonify({'error': 'Error decodificando audio base64: {}'.format(str(e))}), 400
            
            # Crear archivo temporal
            with tempfile.NamedTemporaryFile(suffix='.' + audio_format, delete=False) as temp_file:
                temp_path = temp_file.name
                temp_file.write(audio_bytes)
            
            rospy.loginfo("Reproduciendo audio desde base64: {} bytes".format(len(audio_bytes)))
        
        # Modo 2: Text-to-Speech
        elif 'text' in data:
            text = data.get('text', '').strip()
            voice = data.get('voice', 'es')  # Español por defecto
            
            if not text:
                return jsonify({'error': 'text no puede estar vacío'}), 400
            
            # Crear archivo temporal para TTS
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Comando espeak para TTS
            cmd = [
                'espeak',
                '-v', voice,
                '-s', '150',  # velocidad
                '-w', temp_path,  # escribir a archivo
                text
            ]
            
            rospy.loginfo("Generando TTS para: '{}...'".format(text[:50]))
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)
                return jsonify({
                    'error': 'Error generando TTS: {}'.format(result.stderr),
                    'timestamp': rospy.Time.now().to_sec()
                }), 500
        
        else:
            return jsonify({'error': 'Se requiere audio_data o text'}), 400
        
        # Reproducir audio con aplay
        if temp_path and os.path.exists(temp_path):
            cmd_play = ['aplay', temp_path]
            result = subprocess.run(cmd_play, capture_output=True, text=True)
            
            # Limpiar archivo temporal
            os.remove(temp_path)
            
            if result.returncode != 0:
                return jsonify({
                    'error': 'Error reproduciendo audio: {}'.format(result.stderr),
                    'timestamp': rospy.Time.now().to_sec()
                }), 500
            
            rospy.loginfo("Audio reproducido exitosamente")
            
            return jsonify({
                'status': 'success',
                'message': 'Audio reproducido correctamente',
                'timestamp': rospy.Time.now().to_sec()
            })
        else:
            return jsonify({
                'error': 'No se pudo crear archivo de audio temporal',
                'timestamp': rospy.Time.now().to_sec()
            }), 500
            
    except Exception as e:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        rospy.logerr("Error reproduciendo audio: {}".format(e))
        return jsonify({
            'error': 'Error interno al reproducir audio: {}'.format(str(e)),
            'timestamp': rospy.Time.now().to_sec()
        }), 500

@app.route('/robot/audio/status', methods=['GET'])
def audio_status():
    """Verifica el estado del sistema de audio del robot."""
    try:
        status_info = {
            'microphone': 'unknown',
            'speakers': 'unknown',
            'tts_available': False,
            'timestamp': rospy.Time.now().to_sec()
        }
        
        # Verificar micrófono (arecord)
        try:
            result = subprocess.run(['arecord', '--list-devices'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0 and 'card' in result.stdout.lower():
                status_info['microphone'] = 'available'
            else:
                status_info['microphone'] = 'unavailable'
        except:
            status_info['microphone'] = 'error'
        
        # Verificar altavoces (aplay)
        try:
            result = subprocess.run(['aplay', '--list-devices'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0 and 'card' in result.stdout.lower():
                status_info['speakers'] = 'available'
            else:
                status_info['speakers'] = 'unavailable'
        except:
            status_info['speakers'] = 'error'
        
        # Verificar espeak para TTS
        try:
            result = subprocess.run(['espeak', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                status_info['tts_available'] = True
        except:
            status_info['tts_available'] = False
        
        return jsonify(status_info)
        
    except Exception as e:
        return jsonify({
            'error': 'Error verificando estado de audio: {}'.format(str(e)),
            'timestamp': rospy.Time.now().to_sec()
        }), 500


if __name__ == '__main__':
    print("Iniciando API Flask para control del robot...")
    print("Endpoints disponibles:")
    print("  POST /robot/move - Mover el robot")
    print("  POST /robot/goal - Enviar objetivo")
    print("  POST /robot/stop - Detener robot")
    print("  GET /robot/status - Estado del robot")
    print("  GET /robot/movements - Movimientos disponibles")
    print("  POST /robot/camera/capture - Capturar imagen desde cámara del robot")
    print("  POST /robot/camera/detect_people - Detectar personas y obtener dirección")
    print("  GET /robot/camera/status - Estado de la cámara del robot")
    print("  GET /robot/camera/diagnose - Diagnóstico detallado de cámaras")
    print("  POST /robot/audio/capture - Capturar audio desde micrófono del robot")
    print("  POST /robot/audio/play - Reproducir audio/TTS en el robot")
    print("  GET /robot/audio/status - Estado del sistema de audio del robot")
    init_robot()
    app.run(host='0.0.0.0', port=5000, debug=True) 