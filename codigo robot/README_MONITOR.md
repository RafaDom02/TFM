# Sistema de Monitorización Externa del Robot 🤖

Sistema inteligente que recibe audio del robot, lo procesa con IA y envía comandos de movimiento de vuelta al robot de forma autónoma.

## 🎯 Arquitectura del Sistema

```
[Robot] --audio--> [Monitor] --LLM--> [Comandos] --> [Robot API] --> [Robot]
   ^                   |                                 |
   |                   v                                 |
   |              [Dashboard Web]                       |
   +---------------------------------------------------+
```

## 📋 Componentes

1. **`robot_monitor.py`** - Sistema principal de monitorización
2. **`robot_flask_api.py`** - API del robot (Python 2.7/ROS)
3. **`robot_audio_client.py`** - Cliente simulador de audio
4. **`graph_llm.py`** - Motor de IA para toma de decisiones
5. **Dashboard Web** - Interfaz de monitorización en tiempo real

## 🚀 Características

- ✅ **Recepción de audio** - Endpoint HTTP para recibir audio del robot
- ✅ **Transcripción automática** - Usar Whisper para convertir voz a texto
- ✅ **IA Conversacional** - LLM analiza y decide acciones
- ✅ **Control del robot** - Envío automático de comandos de movimiento
- ✅ **Dashboard en tiempo real** - Monitorización web con métricas
- ✅ **Procesamiento asíncrono** - Cola de audio para alta performance
- ✅ **Logging completo** - Trazabilidad de todas las operaciones
- ✅ **Historial de conversación** - Mantiene contexto de interacciones

## 📦 Instalación

### 1. Instalar dependencias

```bash
# Para el sistema de monitorización (Python 3)
pip3 install -r requirements_monitor.txt

# Para el robot API (Python 2.7)
pip install -r requirements_python27.txt

# Para el cliente de audio (opcional, para pruebas)
pip3 install pyaudio  # Puede requerir librerías del sistema
```

### 2. Configurar credenciales

Asegúrate de que `config.py` esté configurado con:
- Credenciales de Google Cloud (para LLM)
- Claves de API necesarias

### 3. Configurar ROS (para el robot)

```bash
source /opt/ros/melodic/setup.bash  # o tu versión
# Configurar workspace si es necesario
```

## 🎮 Uso del Sistema

### Paso 1: Iniciar API del Robot

```bash
# Terminal 1 - Iniciar API del robot (Python 2.7)
cd "codigo robot"
python2.7 robot_flask_api.py
```

El robot API estará disponible en `http://localhost:5000`

### Paso 2: Iniciar Sistema de Monitorización

```bash
# Terminal 2 - Iniciar monitor (Python 3)
cd "codigo robot"
python3 robot_monitor.py
```

El monitor estará disponible en `http://localhost:5001`

### Paso 3: Acceder al Dashboard

Abrir navegador en `http://localhost:5001` para ver:
- Estado del sistema en tiempo real
- Métricas de procesamiento
- Historial de conversación
- Controles de administración

### Paso 4: Enviar Audio (Simulación)

```bash
# Terminal 3 - Enviar audio de prueba
cd "codigo robot"

# Modo de prueba con comandos predefinidos
python3 robot_audio_client.py --mode test

# Modo de audio en vivo (requiere micrófono)
python3 robot_audio_client.py --mode live

# Enviar archivo de audio específico
python3 robot_audio_client.py --mode file --file mi_audio.wav
```

## 🔄 Flujo de Procesamiento

1. **Recepción**: Audio llega al endpoint `/robot/audio`
2. **Cola**: Audio se añade a cola de procesamiento asíncrono
3. **Transcripción**: Whisper convierte audio a texto
4. **Análisis**: LLM analiza el texto y clasifica la intención
5. **Extracción**: Se extraen comandos de movimiento
6. **Ejecución**: Comandos se envían al robot API
7. **Registro**: Toda la transacción se registra en el historial

## 📊 Endpoints del Monitor

### Audio
- `POST /robot/audio` - Recibir audio del robot

### Monitorización  
- `GET /monitor/status` - Estado del sistema
- `GET /monitor/history` - Historial de conversación
- `POST /monitor/test_robot` - Probar conexión con robot
- `POST /monitor/clear_history` - Limpiar historial

### Dashboard
- `GET /` - Dashboard web interactivo

## 🎤 Formatos de Audio Soportados

- **WAV** (recomendado)
- **MP3** 
- **FLAC**
- **Raw audio data**

### Configuración óptima para Whisper:
- **Sample Rate**: 16kHz
- **Channels**: Mono (1 canal)
- **Bit Depth**: 16-bit
- **Format**: WAV PCM

## 🗣️ Comandos de Voz Reconocidos

El sistema reconoce comandos en español:

### Movimientos Básicos
- "Muévete hacia adelante" / "Ve hacia adelante" / "Avanza"
- "Muévete hacia atrás" / "Retrocede" / "Ve hacia atrás"
- "Gira a la derecha" / "Gira hacia la derecha"
- "Gira a la izquierda" / "Gira hacia la izquierda"

### Movimientos Diagonales
- "Muévete en diagonal derecha"
- "Muévete en diagonal izquierda"

### Control
- "Detente" / "Para" / "Stop"

### Velocidades
- "Muévete lento" (0.2 m/s)
- "Muévete rápido" (0.6 m/s)

## 📈 Dashboard Métricas

El dashboard muestra en tiempo real:

- **Audios Recibidos** - Total de audios recibidos
- **Audios Procesados** - Total procesados exitosamente  
- **Comandos Enviados** - Comandos enviados al robot
- **Cola de Audio** - Audios pendientes de procesar
- **Estado Robot** - Conectado/Desconectado
- **Estado Sistema** - Esperando/Procesando/Error

## 🔧 Configuración Avanzada

### Modificar URLs

En `robot_monitor.py`:
```python
ROBOT_API_BASE_URL = "http://localhost:5000"  # URL del robot API
```

En `robot_audio_client.py`:
```python
MONITOR_URL = "http://localhost:5001"  # URL del monitor
```

### Ajustar Procesamiento de Audio

```python
WHISPER_MODEL_SIZE = "small"  # tiny, base, small, medium, large
MAX_AUDIO_QUEUE_SIZE = 10     # Tamaño máximo de cola
PROCESSING_TIMEOUT = 30       # Timeout en segundos
```

### Personalizar Comandos

Modificar `extract_movement_commands()` en `robot_monitor.py` para añadir nuevos comandos:

```python
movement_mappings = {
    "nueva frase": {"command_type": "move", "movement": "forward", "velocity": 0.3},
    # ... más comandos
}
```

## 🚨 Solución de Problemas

### Monitor no inicia
```bash
# Verificar puerto disponible
netstat -tlnp | grep :5001

# Verificar dependencias
pip3 list | grep -E "(flask|whisper|requests)"
```

### Robot API no disponible
```bash
# Verificar que el robot API esté ejecutándose
curl http://localhost:5000/robot/status

# Verificar ROS
roscore  # Debe estar ejecutándose
```

### Error de transcripción
- Verificar formato de audio (preferir WAV 16kHz)
- Comprobar que el audio no esté vacío
- Revisar logs en `robot_monitor.log`

### Comandos no se ejecutan
- Verificar que los comandos se extraigan correctamente
- Comprobar conexión entre monitor y robot API
- Revisar logs para errores de red

## 📝 Logs

El sistema genera logs detallados en:
- **Consola** - Información en tiempo real
- **`robot_monitor.log`** - Log completo del monitor
- **Dashboard** - Visualización web de eventos

### Nivel de logging
```python
logging.basicConfig(level=logging.INFO)  # INFO, DEBUG, WARNING, ERROR
```

## 🔒 Consideraciones de Seguridad

⚠️ **Importante**: Este sistema está diseñado para desarrollo/demostración.

Para producción considerar:
- Autenticación en endpoints
- Cifrado de comunicaciones (HTTPS)
- Rate limiting
- Validación de entrada más estricta
- Monitorización de seguridad

## 🧪 Testing

### Verificar sistema completo
```bash
# 1. Verificar robot API
curl http://localhost:5000/robot/status

# 2. Verificar monitor
curl http://localhost:5001/monitor/status

# 3. Probar envío de audio
python3 robot_audio_client.py --check-status

# 4. Ejecutar pruebas
python3 robot_audio_client.py --mode test
```

### Comandos de prueba
```bash
# Enviar comando directo al robot
curl -X POST http://localhost:5000/robot/move \
  -H "Content-Type: application/json" \
  -d '{"movement": "forward", "duration": 1.0}'

# Probar conexión del monitor con robot
curl -X POST http://localhost:5001/monitor/test_robot
```

## 📚 Estructura de Archivos

```
codigo robot/
├── robot_monitor.py           # Sistema principal de monitorización
├── robot_flask_api.py         # API del robot (Python 2.7)
├── robot_audio_client.py      # Cliente simulador
├── graph_llm.py              # Motor de IA
├── config.py                 # Configuración
├── audio_utils.py            # Utilidades de audio
├── requirements_monitor.txt   # Dependencias monitor
├── requirements_python27.txt # Dependencias robot
├── ejemplos_curl.txt         # Ejemplos de comandos
└── README_MONITOR.md         # Esta documentación
```

## 🤝 Integración con Robot Real

Para integrar con un robot real:

1. **Captura de Audio**: Implementar captura de audio en el robot
2. **Envío HTTP**: Enviar audio al monitor vía HTTP POST
3. **Configuración de Red**: Ajustar URLs para red local/remota
4. **TTS**: Implementar síntesis de voz para respuestas del robot

### Ejemplo de integración robot:
```python
# En el robot real (Python 2.7 o 3)
import requests

def send_audio_to_monitor(audio_data):
    response = requests.post(
        "http://MONITOR_IP:5001/robot/audio",
        data=audio_data,
        headers={'Content-Type': 'audio/wav'}
    )
    return response.status_code == 200
```

## 📞 Soporte

Para problemas o preguntas:
1. Revisar logs en `robot_monitor.log`
2. Verificar estado en dashboard web
3. Comprobar conectividad entre componentes
4. Revisar configuración en `config.py` 