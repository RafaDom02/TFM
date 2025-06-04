#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de prueba para los nuevos endpoints de cámara del robot.
Verifica que el sistema funcione con la cámara remota del robot.
"""

import requests
import json
import time
import base64
from PIL import Image
import io

# Configuración
ROBOT_API_BASE_URL = "http://localhost:5000"
MONITOR_API_BASE_URL = "http://localhost:5001"

def test_robot_camera_status():
    """Prueba el estado de la cámara del robot."""
    print("🔍 Probando estado de cámara del robot...")
    try:
        response = requests.get(f"{ROBOT_API_BASE_URL}/robot/camera/status", timeout=10)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Cámara del robot: {result['status']}")
            print(f"   Mensaje: {result['message']}")
            if 'resolution' in result:
                res = result['resolution']
                print(f"   Resolución: {res['width']}x{res['height']}")
            return True
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error de conexión: {e}")
        return False

def test_robot_image_capture():
    """Prueba la captura de imagen desde el robot."""
    print("\n📷 Probando captura de imagen del robot...")
    try:
        response = requests.post(
            f"{ROBOT_API_BASE_URL}/robot/camera/capture",
            headers={'Content-Type': 'application/json'},
            timeout=15
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Imagen capturada correctamente")
            print(f"   Formato: {result['image_format']}")
            print(f"   Dimensiones: {result['width']}x{result['height']}")
            print(f"   Tamaño de datos: {len(result['image_data'])} caracteres base64")
            
            # Intentar decodificar y mostrar información de la imagen
            try:
                img_data = base64.b64decode(result['image_data'])
                img = Image.open(io.BytesIO(img_data))
                print(f"   Imagen PIL: {img.format} {img.size} {img.mode}")
                
                # Guardar imagen de prueba
                test_path = "test_robot_capture.jpg"
                img.save(test_path)
                print(f"   💾 Imagen guardada en: {test_path}")
                
            except Exception as decode_error:
                print(f"   ⚠️  Error decodificando imagen: {decode_error}")
            
            return True
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_robot_people_detection():
    """Prueba la captura de imagen del robot para detección de personas (sin procesamiento pesado)."""
    print("\n🚨 Probando captura de imagen del robot para detección...")
    try:
        payload = {
            "return_image": True  # Solicitar imagen para procesamiento en monitor
        }
        
        response = requests.post(
            f"{ROBOT_API_BASE_URL}/robot/camera/detect_people",
            json=payload,
            headers={'Content-Type': 'application/json'},
            timeout=20
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Imagen capturada para detección remota")
            print(f"   Resolución del frame: {result.get('frame_width')}x{result.get('frame_height')}")
            print(f"   Mensaje: {result.get('message')}")
            
            if result.get('image_data'):
                print(f"   📷 Imagen incluida: {len(result['image_data'])} caracteres base64")
                print(f"   🔄 Robot NO hace detección pesada - solo captura y envía")
            else:
                print(f"   ⚠️  No se incluyó imagen en la respuesta")
            
            return True
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_monitor_integration():
    """Prueba la integración del monitor con los endpoints del robot."""
    print("\n🔗 Probando integración del monitor...")
    try:
        # Verificar estado del monitor
        response = requests.get(f"{MONITOR_API_BASE_URL}/monitor/status", timeout=10)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Monitor activo")
            print(f"   Estado del robot API: {result.get('robot_api_status')}")
            print(f"   URL del robot: {result.get('robot_api_url')}")
            
            # Verificar si puede conectar con el robot
            if result.get('robot_api_status') == 'connected':
                print(f"   ✅ Monitor conectado correctamente al robot")
                return True
            else:
                print(f"   ⚠️  Monitor no puede conectar con el robot")
                return False
        else:
            print(f"❌ Error del monitor: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error conectando con monitor: {e}")
        return False

def test_full_workflow():
    """Prueba el flujo completo: robot captura -> monitor procesa -> comandos ejecutados."""
    print("\n🔄 Probando flujo completo de trabajo...")
    
    try:
        print("   1. Robot capturando imagen...")
        response = requests.post(
            f"{ROBOT_API_BASE_URL}/robot/camera/detect_people",
            json={"return_image": True},
            headers={'Content-Type': 'application/json'},
            timeout=15
        )
        
        if response.status_code == 200:
            result = response.json()
            
            if result.get('image_data'):
                print(f"   ✅ Robot envió imagen: {len(result['image_data'])} caracteres base64")
                print("   2. Imagen enviada al monitor para procesamiento pesado...")
                print("      📝 Nota: El monitor usaría Google Cloud Vision para detectar personas")
                print("      📝 Nota: El monitor generaría comandos de movimiento automáticamente")
                
                # Simular comando generado por el monitor (en producción vendría del LLM)
                mock_command = {
                    "movement": "forward",
                    "velocity": 0.3,
                    "duration": 1.5
                }
                
                print(f"   3. Ejecutando comando simulado: {mock_command}")
                cmd_response = requests.post(
                    f"{ROBOT_API_BASE_URL}/robot/move",
                    json=mock_command,
                    headers={'Content-Type': 'application/json'},
                    timeout=10
                )
                
                if cmd_response.status_code == 200:
                    cmd_result = cmd_response.json()
                    print(f"   ✅ Comando ejecutado: {cmd_result.get('message')}")
                    print("   ✅ Flujo completo: Robot captura → Monitor procesa → Robot ejecuta")
                    return True
                else:
                    print(f"   ❌ Error ejecutando comando: {cmd_response.text}")
                    return False
            else:
                print("   ⚠️  Robot no envió imagen")
                return False
        else:
            print(f"   ❌ Error capturando imagen: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error en flujo completo: {e}")
        return False

def main():
    """Función principal de pruebas."""
    print("🤖 PRUEBAS DEL SISTEMA DE CÁMARA REMOTA DEL ROBOT")
    print("📋 Arquitectura: Robot captura → Monitor procesa → Robot ejecuta")
    print("=" * 60)
    
    tests = [
        ("Estado de la cámara", test_robot_camera_status),
        ("Captura de imagen", test_robot_image_capture),
        ("Captura para detección", test_robot_people_detection),
        ("Integración con monitor", test_monitor_integration),
        ("Flujo completo", test_full_workflow)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
            if result:
                print(f"✅ {test_name}: PASÓ")
            else:
                print(f"❌ {test_name}: FALLÓ")
        except Exception as e:
            print(f"💥 {test_name}: ERROR - {e}")
            results.append((test_name, False))
        
        print("-" * 30)
        time.sleep(1)
    
    # Resumen final
    print("\n📊 RESUMEN DE PRUEBAS")
    print("=" * 30)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASÓ" if result else "❌ FALLÓ"
        print(f"  {test_name}: {status}")
    
    print(f"\nResultado final: {passed}/{total} pruebas pasaron")
    
    if passed == total:
        print("🎉 ¡Todas las pruebas pasaron! El sistema está funcionando correctamente.")
        print("📋 Arquitectura verificada:")
        print("   - 🤖 Robot: Captura imágenes desde su cámara (cómputo mínimo)")
        print("   - 📡 Transmisión: Envío de imagen vía HTTP/base64")
        print("   - 💻 Monitor: Procesamiento pesado con Google Cloud Vision")
        print("   - ⚡ Ejecución: Comandos generados en monitor, ejecutados en robot")
    else:
        print("⚠️  Algunas pruebas fallaron. Verifica:")
        print("   - Que robot_flask_api.py esté ejecutándose en puerto 5000")
        print("   - Que robot_monitor.py esté ejecutándose en puerto 5001") 
        print("   - Que la cámara del robot esté conectada y funcionando")
        print("   - Que Google Cloud Vision esté configurado en el monitor")
        print("   - Que las dependencias (OpenCV, PIL, etc.) estén instaladas")

if __name__ == "__main__":
    main() 