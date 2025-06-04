#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ejemplo de uso de la API Flask para controlar el robot.
Compatible con Python 3.
"""

import requests
import json
import time

# URL base de la API
BASE_URL = "http://localhost:5000"

def test_robot_api():
    """Prueba todos los endpoints de la API del robot."""
    
    print("=== Prueba de API del Robot ===\n")
    
    # 1. Obtener movimientos disponibles
    print("1. Obteniendo movimientos disponibles...")
    try:
        response = requests.get(BASE_URL + "/robot/movements")
        if response.status_code == 200:
            data = response.json()
            print("Movimientos disponibles:", data['available_movements'])
            print("Parámetros:", data['parameters'])
        else:
            print("Error:", response.text)
    except requests.RequestException as e:
        print("Error de conexión:", e)
        return
    
    print("\n" + "="*50 + "\n")
    
    # 2. Verificar estado del robot
    print("2. Verificando estado del robot...")
    try:
        response = requests.get(BASE_URL + "/robot/status")
        if response.status_code == 200:
            data = response.json()
            print("Estado del robot:", data)
        else:
            print("Error:", response.text)
    except requests.RequestException as e:
        print("Error de conexión:", e)
    
    print("\n" + "="*50 + "\n")
    
    # 3. Ejemplos de movimientos
    movimientos_ejemplo = [
        {
            "nombre": "Adelante con velocidad personalizada",
            "data": {
                "movement": "forward",
                "velocity": 0.5,
                "duration": 3.0
            }
        },
        {
            "nombre": "Giro a la derecha",
            "data": {
                "movement": "spin_right",
                "angular_velocity": 1.0,
                "duration": 2.0
            }
        },
        {
            "nombre": "Adelante-izquierda con ángulo personalizado",
            "data": {
                "movement": "forward_left",
                "velocity": 0.3,
                "angle": 45,
                "duration": 2.5
            }
        },
        {
            "nombre": "Movimiento personalizado",
            "data": {
                "movement": "custom",
                "velocity": 0.2,
                "angular_velocity": 0.3,
                "duration": 2.0
            }
        }
    ]
    
    # Ejecutar cada movimiento de ejemplo
    for i, ejemplo in enumerate(movimientos_ejemplo, 3):
        print("{}. Ejecutando: {}".format(i, ejemplo["nombre"]))
        print("   Parámetros:", ejemplo["data"])
        
        try:
            response = requests.post(
                BASE_URL + "/robot/move",
                json=ejemplo["data"],
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                result = response.json()
                print("   ✓ Éxito:", result["message"])
            else:
                print("   ✗ Error:", response.text)
                
        except requests.RequestException as e:
            print("   ✗ Error de conexión:", e)
        
        print("   Esperando 1 segundo antes del siguiente movimiento...")
        time.sleep(1)
        print("\n" + "="*50 + "\n")
    
    # 4. Ejemplo de envío de objetivo
    print("{}. Enviando objetivo de navegación...".format(len(movimientos_ejemplo) + 3))
    goal_data = {
        "x": 2.0,
        "y": 1.5,
        "yaw": 90
    }
    print("   Objetivo:", goal_data)
    
    try:
        response = requests.post(
            BASE_URL + "/robot/goal",
            json=goal_data,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("   ✓ Éxito:", result["message"])
        else:
            print("   ✗ Error:", response.text)
            
    except requests.RequestException as e:
        print("   ✗ Error de conexión:", e)
    
    print("\n" + "="*50 + "\n")
    
    # 5. Detener el robot
    print("{}. Deteniendo el robot...".format(len(movimientos_ejemplo) + 4))
    try:
        response = requests.post(BASE_URL + "/robot/stop")
        
        if response.status_code == 200:
            result = response.json()
            print("   ✓ Éxito:", result["message"])
        else:
            print("   ✗ Error:", response.text)
            
    except requests.RequestException as e:
        print("   ✗ Error de conexión:", e)
    
    print("\n=== Prueba completada ===")


def movimiento_simple(movement_type, **kwargs):
    """Función auxiliar para enviar un comando de movimiento simple."""
    data = {"movement": movement_type}
    data.update(kwargs)
    
    try:
        response = requests.post(
            BASE_URL + "/robot/move",
            json=data,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("Movimiento ejecutado:", result["message"])
            return True
        else:
            print("Error:", response.text)
            return False
            
    except requests.RequestException as e:
        print("Error de conexión:", e)
        return False


def ejemplos_uso_rapido():
    """Ejemplos rápidos de uso de la API."""
    print("=== Ejemplos de uso rápido ===\n")
    
    # Movimiento hacia adelante por 2 segundos
    print("Adelante por 2 segundos:")
    movimiento_simple("forward", duration=2.0)
    
    time.sleep(3)
    
    # Giro a la derecha por 1.5 segundos
    print("\nGiro derecha por 1.5 segundos:")
    movimiento_simple("spin_right", duration=1.5)
    
    time.sleep(2)
    
    # Movimiento personalizado
    print("\nMovimiento personalizado:")
    movimiento_simple("custom", velocity=0.2, angular_velocity=0.5, duration=2.0)


if __name__ == "__main__":
    import sys
    
    print("Iniciando pruebas de la API del robot...")
    print("Asegúrate de que la API esté ejecutándose en http://localhost:5000")
    print("Puedes ejecutar la API con: python2.7 robot_flask_api.py\n")
    
    # Verificar conexión básica
    try:
        response = requests.get(BASE_URL + "/robot/status")
        if response.status_code != 200:
            print("¡La API no está disponible! Asegúrate de ejecutar robot_flask_api.py primero.")
            sys.exit(1)
    except requests.RequestException:
        print("¡No se puede conectar a la API! Asegúrate de ejecutar robot_flask_api.py primero.")
        sys.exit(1)
    
    # Ejecutar pruebas
    if len(sys.argv) > 1 and sys.argv[1] == "rapido":
        ejemplos_uso_rapido()
    else:
        test_robot_api() 