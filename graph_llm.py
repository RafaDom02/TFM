import os
import cv2
import uuid
from typing import TypedDict, Optional, Any, List

from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from google.cloud import vision
from google.oauth2 import service_account

# --- Configuración ---
# Establece la ruta a tu archivo de credenciales de Google Cloud
# ¡ASEGÚRATE DE QUE ESTA RUTA SEA CORRECTA!
CREDENTIALS_FILE_PATH = r"c:\Users\User\Desktop\Master\TFM\nimble-root-457808-r2-b639a6729402.json"

# Establecer la variable de entorno para las credenciales de Google Cloud
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = CREDENTIALS_FILE_PATH

# Configuración del modelo Gemini
# Usaremos gemini-1.5-flash-latest, que es rápido y potente.
GEMINI_MODEL_NAME = "gemini-2.0-flash"

# Frases para activar la descripción de imagen (en minúsculas)
IMAGE_TRIGGER_PHRASES = [
    "describe lo que ves",
    "qué ves",
    "describe la imagen",
    "mira esto",
    "analiza esta imagen",
    "describe what you see",
    "what do you see"
]

# --- Definición del Estado del Grafo ---
class GraphState(TypedDict):
    user_input: str
    classification: Optional[str]  # "normal" o "describe_image"
    image_path: Optional[str]
    vision_analysis: Optional[str]
    final_response: str
    error: Optional[str]
    history: List[tuple[str, str]] # Para mantener un historial simple de conversación

# --- Nodos del Grafo ---

def classify_input_node(state: GraphState) -> GraphState:
    """Clasifica la entrada del usuario."""
    print("--- CLASIFICANDO ENTRADA ---")
    user_input_lower = state["user_input"].lower()
    if any(phrase in user_input_lower for phrase in IMAGE_TRIGGER_PHRASES):
        print("Clasificación: describe_image")
        return {**state, "classification": "describe_image", "error": None}
    else:
        print("Clasificación: normal")
        return {**state, "classification": "normal", "error": None}

def normal_response_node(state: GraphState) -> GraphState:
    """Genera una respuesta normal usando Gemini."""
    print("--- GENERANDO RESPUESTA NORMAL ---")
    try:
        llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL_NAME, convert_system_message_to_human=True)
        
        # Construir un prompt simple con historial
        prompt_history = "\n".join([f"Humano: {h}\nAI: {a}" for h, a in state.get("history", [])])
        full_prompt = f"{prompt_history}\nHumano: {state['user_input']}\nAI:"
        
        response = llm.invoke(full_prompt)
        final_response = response.content
        print(f"Respuesta normal: {final_response}")
        
        # Actualizar historial
        new_history = state.get("history", []) + [(state["user_input"], final_response)]
        return {**state, "final_response": final_response, "history": new_history, "error": None}
    except Exception as e:
        print(f"Error en normal_response_node: {e}")
        return {**state, "final_response": "Lo siento, tuve un problema al procesar tu solicitud.", "error": str(e)}

def capture_image_node(state: GraphState) -> GraphState:
    """Captura una imagen desde la webcam."""
    print("--- CAPTURANDO IMAGEN ---")
    try:
        cap = cv2.VideoCapture(0) # 0 es usualmente la cámara por defecto
        if not cap.isOpened():
            print("Error: No se pudo abrir la cámara.")
            return {**state, "error": "No se pudo acceder a la cámara web.", "final_response": "No pude acceder a tu cámara web. ¿Está conectada y funcionando?"}

        print("Presiona 'espacio' para capturar la imagen, o 'q' para cancelar.")
        
        temp_image_path = f"temp_webcam_capture_{uuid.uuid4()}.jpg"

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: No se pudo leer el frame de la cámara.")
                cap.release()
                cv2.destroyAllWindows()
                return {**state, "error": "Error al leer de la cámara.", "final_response": "Hubo un problema al leer de tu cámara web."}

            cv2.imshow('Webcam - Presiona ESPACIO para capturar, Q para salir', frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(' '): # Espacio para capturar
                cv2.imwrite(temp_image_path, frame)
                print(f"Imagen guardada en: {temp_image_path}")
                cap.release()
                cv2.destroyAllWindows()
                return {**state, "image_path": temp_image_path, "error": None}
            elif key == ord('q'): # Q para salir/cancelar
                print("Captura cancelada por el usuario.")
                cap.release()
                cv2.destroyAllWindows()
                return {**state, "error": "Captura de imagen cancelada.", "final_response": "De acuerdo, cancelé la captura de imagen."}
        
    except Exception as e:
        print(f"Error en capture_image_node: {e}")
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
        return {**state, "error": str(e), "final_response": "Lo siento, ocurrió un error al intentar usar la cámara web."}

def analyze_image_node(state: GraphState) -> GraphState:
    """Analiza la imagen usando Google Cloud Vision API."""
    print("--- ANALIZANDO IMAGEN CON GOOGLE CLOUD VISION ---")
    if not state.get("image_path"):
        return {**state, "error": "No se proporcionó ruta de imagen para analizar.", "final_response": "No hay imagen para analizar."}

    try:
        # No es necesario pasar las credenciales explícitamente si GOOGLE_APPLICATION_CREDENTIALS está configurado
        client = vision.ImageAnnotatorClient()

        with open(state["image_path"], "rb") as image_file:
            content = image_file.read()
        image = vision.Image(content=content)

        # Especifica las características que quieres detectar
        features = [
            vision.Feature(type_=vision.Feature.Type.LABEL_DETECTION, max_results=10),
            vision.Feature(type_=vision.Feature.Type.OBJECT_LOCALIZATION, max_results=5),
            vision.Feature(type_=vision.Feature.Type.WEB_DETECTION, max_results=5),
            vision.Feature(type_=vision.Feature.Type.FACE_DETECTION, max_results=5), # Opcional
            vision.Feature(type_=vision.Feature.Type.LANDMARK_DETECTION, max_results=5) # Opcional
        ]
        request = vision.AnnotateImageRequest(image=image, features=features)
        response = client.annotate_image(request=request)

        if response.error.message:
            raise Exception(f"Error de Vision API: {response.error.message}")

        analysis_parts = []
        if response.label_annotations:
            labels = [label.description for label in response.label_annotations]
            analysis_parts.append(f"Etiquetas detectadas: {', '.join(labels)}.")
        if response.localized_object_annotations:
            objects = list(set([obj.name for obj in response.localized_object_annotations])) # Nombres únicos
            analysis_parts.append(f"Objetos detectados: {', '.join(objects)}.")
        if response.web_detection and response.web_detection.web_entities:
            web_entities = [entity.description for entity in response.web_detection.web_entities if entity.description]
            if web_entities:
                analysis_parts.append(f"Entidades web relacionadas: {', '.join(web_entities[:3])}.") # Limitar a 3

        vision_analysis_str = "\n".join(analysis_parts) if analysis_parts else "No se detectaron características destacadas en la imagen."
        print(f"Análisis de Vision API: {vision_analysis_str}")
        
        # Limpiar la imagen temporal después del análisis
        if os.path.exists(state["image_path"]):
            os.remove(state["image_path"])
            print(f"Imagen temporal {state['image_path']} eliminada.")
            
        return {**state, "vision_analysis": vision_analysis_str, "error": None}

    except Exception as e:
        print(f"Error en analyze_image_node: {e}")
        if state.get("image_path") and os.path.exists(state["image_path"]): # Asegurarse de limpiar si falla
            os.remove(state["image_path"])
        return {**state, "error": str(e), "final_response": "Lo siento, tuve un problema al analizar la imagen con Google Cloud Vision."}

def describe_image_node(state: GraphState) -> GraphState:
    """Genera una descripción de la imagen usando Gemini y el análisis de Vision."""
    print("--- GENERANDO DESCRIPCIÓN DE IMAGEN CON GEMINI ---")
    if state.get("error") and state["error"] == "Captura de imagen cancelada.": # Si la captura fue cancelada
        return {**state, "final_response": "De acuerdo, no describiré ninguna imagen."}
        
    if not state.get("vision_analysis"):
        return {**state, "final_response": "No pude obtener un análisis de la imagen para describirla."}

    try:
        llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL_NAME, convert_system_message_to_human=True)
        
        prompt = (
            f"El usuario te ha pedido que describas lo que ves. "
            f"Has capturado una imagen y la has analizado con Google Cloud Vision. Aquí está el resumen del análisis:\n"
            f"{state['vision_analysis']}\n\n"
            f"Basándote en este análisis, y considerando que el usuario dijo originalmente: '{state['user_input']}', "
            f"proporciona una descripción natural y conversacional de lo que probablemente hay en la imagen."
        )
        
        response = llm.invoke(prompt)
        final_response = response.content
        print(f"Descripción de imagen: {final_response}")
        
        # Actualizar historial
        new_history = state.get("history", []) + [(state["user_input"], final_response)]
        return {**state, "final_response": final_response, "history": new_history, "error": None}
    except Exception as e:
        print(f"Error en describe_image_node: {e}")
        return {**state, "final_response": "Lo siento, tuve un problema al generar la descripción de la imagen.", "error": str(e)}

# --- Lógica Condicional del Grafo ---

def should_capture_image(state: GraphState) -> str:
    """Decide si se debe capturar una imagen o dar una respuesta normal."""
    print(f"--- DECIDIENDO RUTA: {state['classification']} ---")
    if state.get("error"): # Si hubo un error en la clasificación o antes
        return "handle_error" # Podríamos añadir un nodo de manejo de errores específico
    if state["classification"] == "describe_image":
        return "capture_image"
    else:
        return "normal_response"

def after_capture_image(state: GraphState) -> str:
    """Decide qué hacer después de intentar capturar la imagen."""
    print(f"--- DECIDIENDO DESPUÉS DE CAPTURA: Error: {state.get('error')} ---")
    if state.get("error"): # Si hubo error en la captura (ej. cámara no encontrada, cancelación)
        # Si el error es por cancelación, describe_image_node lo manejará para dar una respuesta adecuada.
        # Si es otro error de captura, el final_response ya podría estar seteado por capture_image_node.
        # O podríamos dirigir a un nodo de error o directamente a END si final_response ya está.
        if state["error"] == "Captura de imagen cancelada.":
             # Aún así, vamos a describe_image_node para que dé el mensaje de cancelación.
            return "describe_image" # O podríamos tener un nodo específico para "cancelado"
        return END # Termina si hay un error grave de cámara y ya hay un mensaje.
    return "analyze_image"

def after_analysis(state: GraphState) -> str:
    """Decide qué hacer después del análisis de imagen."""
    print(f"--- DECIDIENDO DESPUÉS DE ANÁLISIS: Error: {state.get('error')} ---")
    if state.get("error"): # Si hubo error en el análisis
        return END # Termina si hay un error de análisis y ya hay un mensaje.
    return "describe_image"

# --- Construcción del Grafo ---
workflow = StateGraph(GraphState)

# Añadir nodos
workflow.add_node("classify_input", classify_input_node)
workflow.add_node("normal_response", normal_response_node)
workflow.add_node("capture_image", capture_image_node)
workflow.add_node("analyze_image", analyze_image_node)
workflow.add_node("describe_image", describe_image_node)

# Definir el punto de entrada
workflow.set_entry_point("classify_input")

# Añadir ejes condicionales
workflow.add_conditional_edges(
    "classify_input",
    should_capture_image,
    {
        "capture_image": "capture_image",
        "normal_response": "normal_response",
        "handle_error": END # Ejemplo si quisiéramos un nodo de error
    }
)

workflow.add_conditional_edges(
    "capture_image",
    after_capture_image,
    {
        "analyze_image": "analyze_image",
        "describe_image": "describe_image", # Para el caso de cancelación
        END: END
    }
)

workflow.add_conditional_edges(
    "analyze_image",
    after_analysis,
    {
        "describe_image": "describe_image",
        END: END
    }
)

# Añadir ejes normales
workflow.add_edge("normal_response", END)
workflow.add_edge("describe_image", END)

# Compilar el grafo
app = workflow.compile()

# --- Interfaz de Usuario Simple ---
if __name__ == "__main__":
    print("¡Hola! Soy tu asistente de IA. Puedes chatear conmigo.")
    print("Si quieres que describa lo que veo, di algo como 'describe lo que ves'.")
    print(f"Usando el modelo Gemini: {GEMINI_MODEL_NAME}")
    print(f"Credenciales de Google Cloud cargadas desde: {CREDENTIALS_FILE_PATH}")
    
    # Verificar si el archivo de credenciales existe
    if not os.path.exists(CREDENTIALS_FILE_PATH):
        print("\n¡ADVERTENCIA! El archivo de credenciales JSON no se encontró en la ruta especificada.")
        print("La funcionalidad de Google Cloud Vision y Gemini no funcionará.")
        print("Por favor, verifica la variable CREDENTIALS_FILE_PATH en el script.\n")
        # Podríamos salir aquí o dejar que continúe con errores.
        # exit()

    current_history = []
    while True:
        user_input = input("Tú: ")
        if user_input.lower() in ["salir", "exit", "quit"]:
            print("¡Hasta luego!")
            break
        
        initial_state = {"user_input": user_input, "history": current_history}
        final_state = app.invoke(initial_state)
        
        print(f"IA: {final_state.get('final_response', 'No recibí respuesta.')}")
        
        # Actualizar el historial para la siguiente iteración
        current_history = final_state.get("history", current_history)
        
        # Limpiar imagen temporal si por alguna razón no se limpió antes (poco probable con el flujo actual)
        if final_state.get("image_path") and os.path.exists(final_state["image_path"]):
            try:
                os.remove(final_state["image_path"])
                print(f"(Limpieza extra) Imagen temporal {final_state['image_path']} eliminada.")
            except Exception as e:
                print(f"Error al intentar limpieza extra de imagen: {e}")