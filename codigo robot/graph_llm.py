import os
import cv2
import uuid
from typing import TypedDict, Optional, Any, List

from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from google.cloud import vision
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
# Removed: from google.oauth2 import service_account # Not directly used after credential env var is set

# Import configuration variables
import config

# --- Configuración ---
# Establece la variable de entorno para las credenciales de Google Cloud
# Esta línea DEBE estar antes de cualquier inicialización de cliente de Google Cloud
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = config.CREDENTIALS_FILE_PATH

# Configuración del modelo Gemini ya no se define aquí, se usa config.GEMINI_MODEL_NAME

# Frases para activar la descripción de imagen (en minúsculas) - ESTO SE ELIMINARÁ
# IMAGE_TRIGGER_PHRASES = [
# "describe lo que ves",
# "qué ves",
# "describe la imagen",
# "mira esto",
# "analiza esta imagen",
# "describe what you see",
# "what do you see"
# ]

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
    """Clasifica la entrada del usuario usando un LLM para determinar la intención."""
    print("--- CLASIFICANDO ENTRADA CON LLM ---")
    
    try:
        llm = ChatGoogleGenerativeAI(model=config.GEMINI_MODEL_NAME)
        
        classification_prompt_text = config.CLASSIFICATION_PROMPT_TEMPLATE.format(user_input=state['user_input'])
        
        response = llm.invoke(classification_prompt_text)
        # Limpiar un poco más la respuesta: minúsculas, quitar espacios extra y posibles saltos de línea.
        # También quitar markdown si es que lo añade (como **)
        classification_result = response.content.strip().lower().replace("*", "")
        
        print(f"Respuesta del LLM para clasificación (procesada): '{classification_result}'")

        # Ahora buscamos si la palabra clave está contenida en la respuesta,
        # lo que da un poco más de flexibilidad si el LLM no es perfectamente conciso.
        if "describe_image" in classification_result:
            print("Clasificación: describe_image")
            return {**state, "classification": "describe_image", "error": None}
        elif "video" in classification_result:
            print("Clasificación: video")
            return {**state, "classification": "video", "error": None}
        elif "normal" in classification_result: # Priorizar describe_image si ambas estuvieran por error
            print("Clasificación: normal")
            return {**state, "classification": "normal", "error": None}
        else:
            print(f"Clasificación (fallback por respuesta no reconocida del LLM: '{classification_result}'): normal")
            return {**state, "classification": "normal", "error": f"Respuesta de clasificación del LLM no reconocida: {classification_result}"}

    except Exception as e:
        print(f"Error en classify_input_node (LLM): {e}")
        # Fallback a clasificación normal si hay error con el LLM
        return {**state, "classification": "normal", "error": f"Error al clasificar con LLM: {str(e)}"}

def normal_response_node(state: GraphState) -> GraphState:
    """Genera una respuesta normal usando Gemini con el system prompt de Jarvis."""
    print("--- GENERANDO RESPUESTA NORMAL (JARVIS) ---")
    try:
        llm = ChatGoogleGenerativeAI(model=config.GEMINI_MODEL_NAME)
        
        messages = [SystemMessage(content=config.SYSTEM_PROMPT)]
        # Reconstruir el historial con los tipos de mensaje correctos
        for h_text, ai_text in state.get("history", []):
            messages.append(HumanMessage(content=h_text))
            messages.append(AIMessage(content=ai_text))
        messages.append(HumanMessage(content=state['user_input']))
        
        response = llm.invoke(messages)
        final_response = response.content
        print(f"Respuesta normal (Jarvis): {final_response}")
        
        # Actualizar historial
        new_history = state.get("history", []) + [(state["user_input"], final_response)]
        return {**state, "final_response": final_response, "history": new_history, "error": None}
    except Exception as e:
        print(f"Error en normal_response_node: {e}")
        return {**state, "final_response": "Lo siento, tuve un problema al procesar tu solicitud.", "error": str(e)}

def request_image_prompt_node(state: GraphState) -> GraphState:
    """Node to tell the user to provide an image."""
    print("--- SOLICITANDO IMAGEN AL USUARIO ---")
    response_text = "Por favor, utiliza la función de cámara para proporcionar una imagen para que la describa."
    # Mantener el historial actual, no añadir este intercambio como si fuera una respuesta completa de IA aún.
    # Opcionalmente, podríamos añadirlo al historial si quisiéramos que aparezca en el chat.
    # Por ahora, solo establecemos final_response.
    return {**state, "final_response": response_text, "error": "Image requested from user"}

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
        llm = ChatGoogleGenerativeAI(model=config.GEMINI_MODEL_NAME) 
        
        messages = [SystemMessage(content=config.SYSTEM_PROMPT)]
        
        user_request_for_image_description = state['user_input']
        vision_analysis_summary = state['vision_analysis']

        # Formatear el prompt usando la plantilla de config.py
        formatted_prompt_for_description = config.PROMPT_FOR_IMAGE_DESCRIPTION_TEMPLATE.format(
            user_request=user_request_for_image_description,
            vision_analysis=vision_analysis_summary
        )
        
        messages.append(HumanMessage(content=formatted_prompt_for_description))
        
        response = llm.invoke(messages)
        final_response = response.content
        print(f"Descripción de imagen (Jarvis): {final_response}")
        
        new_history = state.get("history", []) + [(state["user_input"], final_response)]
        return {**state, "final_response": final_response, "history": new_history, "error": None}
    except Exception as e:
        print(f"Error en describe_image_node: {e}")
        return {**state, "final_response": "Lo siento, tuve un problema al generar la descripción de la imagen.", "error": str(e)}

# --- Lógica Condicional del Grafo ---

def route_after_classification(state: GraphState) -> str:
    """Decide la siguiente ruta después de la clasificación inicial."""
    print(f"--- RUTEO POST-CLASIFICACIÓN: {state['classification']}, Image_Path: {state.get('image_path')} ---")
    if state.get("error") and "clasificar con LLM" in state["error"]: # Error en clasificación
        return END # Ya debería haber un final_response de error
    
    if state["classification"] == "describe_image":
        if state.get("image_path"): # Si Streamlit proveyó una imagen
            return "analyze_image"
        else: # El usuario quiere describir pero no dio imagen con este input
            return "request_image_prompt"
    elif state["classification"] == "video":
        # Tratamos 'video' como flujo normal (la app manejará el vídeo)
        return "normal_response"
    else: # normal_response
        return "normal_response"

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
workflow.add_node("request_image_prompt", request_image_prompt_node)
workflow.add_node("analyze_image", analyze_image_node)
workflow.add_node("describe_image", describe_image_node)

# Definir el punto de entrada
workflow.set_entry_point("classify_input")

# Añadir ejes condicionales
workflow.add_conditional_edges(
    "classify_input",
    route_after_classification,
    {
        "analyze_image": "analyze_image",
        "request_image_prompt": "request_image_prompt",
        "normal_response": "normal_response",
        END: END # Si hubo error en clasificación
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
workflow.add_edge("request_image_prompt", END)

# Compilar el grafo
app = workflow.compile()

# Nombre de exportación claro y en inglés
jarvis_graph_app = app

# Eliminar el bloque if __name__ == "__main__":
# La ejecución ahora es manejada por streamlit_app.py