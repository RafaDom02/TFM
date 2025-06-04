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
        
        # Intentar abrir la cámara
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return jsonify({
                'error': 'No se pudo acceder a la cámara del robot',
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
        
        # Guardar imagen temporal en el robot
        os.makedirs("temp_images", exist_ok=True)
        temp_filename = f"temp_robot_capture_{uuid.uuid4()}.jpg"
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
        
        rospy.loginfo(f"Imagen capturada desde cámara del robot: {width}x{height}")
        
        return jsonify({
            'status': 'success',
            'message': 'Imagen capturada correctamente',
            'image_data': img_base64,
            'image_format': 'jpeg',
            'width': int(width),
            'height': int(height),
            'timestamp': rospy.Time.now().to_sec(),
            'temp_filename': temp_filename
        })
        
    except Exception as e:
        rospy.logerr(f"Error capturando imagen: {e}")
        return jsonify({
            'error': f'Error interno al capturar imagen: {str(e)}',
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
        
        # Capturar frame
        cap = cv2.VideoCapture(0)
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
                rospy.loginfo(f"Frame capturado y codificado para detección remota: {w}x{h}")
            else:
                return jsonify({
                    'error': 'Error al codificar imagen para envío',
                    'timestamp': rospy.Time.now().to_sec()
                }), 500
        
        return jsonify(result)
        
    except Exception as e:
        rospy.logerr(f"Error en detect_people_camera: {e}")
        return jsonify({
            'error': f'Error interno: {str(e)}',
            'timestamp': rospy.Time.now().to_sec()
        }), 500

@app.route('/robot/camera/status', methods=['GET'])
def camera_status():
    """Verifica el estado de la cámara del robot."""
    try:
        import cv2
        
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            
            if ret and frame is not None:
                height, width, channels = frame.shape
                return jsonify({
                    'status': 'available',
                    'message': 'Cámara del robot operativa',
                    'resolution': {'width': int(width), 'height': int(height)},
                    'timestamp': rospy.Time.now().to_sec()
                })
            else:
                return jsonify({
                    'status': 'error',
                    'message': 'Cámara accesible pero no puede capturar',
                    'timestamp': rospy.Time.now().to_sec()
                }), 500
        else:
            return jsonify({
                'status': 'unavailable',
                'message': 'No se puede acceder a la cámara del robot',
                'timestamp': rospy.Time.now().to_sec()
            }), 500
            
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error verificando cámara: {str(e)}',
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
    
    app.run(host='0.0.0.0', port=5000, debug=True) 