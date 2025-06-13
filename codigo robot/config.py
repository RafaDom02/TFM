# --- Configuración ---
# Establece la ruta a tu archivo de credenciales de Google Cloud
# ¡ASEGÚRATE DE QUE ESTA RUTA SEA CORRECTA!
CREDENTIALS_FILE_PATH = r"/home/rafadom/2ºCuatrimestre/TFM/nimble-root-457808-r2-b639a6729402.json"

# Configuración del modelo Gemini
# Usaremos gemini-2.0-flash, que es rápido y potente.
GEMINI_MODEL_NAME = "gemini-2.0-flash"

# System Prompt para el LLM
SYSTEM_PROMPT = (
    "Eres Jarvis, una avanzada plataforma robótica móvil diseñada para asistir en operaciones de rescate y ayuda "
    "durante catástrofes naturales. Tu principal objetivo es proporcionar información clara, concisa y útil. "
    "Interactúa con profesionalismo y empatía, recordando siempre tu rol de asistente en situaciones críticas. "
    "Estás equipado con una cámara para analizar el entorno. No respondas con acentos ni signos de puntuación en español como ¿."
)

# Prompt para la clasificación de la entrada del usuario
CLASSIFICATION_PROMPT_TEMPLATE = (
    "Dada la siguiente solicitud de un usuario a Jarvis, una plataforma robótica con cámara: \"{user_input}\"\n"
    "¿La solicitud implica que Jarvis use su cámara para ver, observar, analizar o describir algo visualmente?\n"
    "Además, si el usuario está pidiendo auxilio explícitamente (palabras como 'ayuda', 'socorro', 'auxilio', 'ahhhh', 'aquí'), se debe activar el modo de VÍDEO en tiempo real.\n"
    "Responde con UNA ÚNICA palabra entre: 'video', 'describe_image' o 'normal'.\n"
    "NO incluyas ninguna otra palabra o explicación."
)

# Prompt para la descripción de la imagen
PROMPT_FOR_IMAGE_DESCRIPTION_TEMPLATE = (
    "El usuario te ha pedido que describas lo que ves (su petición original fue: '{user_request}'). "
    "Has capturado una imagen y la has analizado con Google Cloud Vision. Aquí está el resumen del análisis:\n"
    "{vision_analysis}\n\n"
    "Basándote en este análisis, proporciona una descripción natural y conversacional de lo que probablemente hay en la imagen. Indica si es un entorno desordenado o no."
    "Recuerda tu rol como Jarvis."
)
