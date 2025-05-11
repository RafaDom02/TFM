import streamlit as st
import os
import uuid
from PIL import Image # For handling image data from Streamlit
import cv2                                 # ➊  nuevo

# Import necessary components from your graph_llm and config
import config # Assuming config.py is in the same directory or accessible
from graph_llm import jarvis_graph_app # Corrected import name
from graph_llm import GraphState # For type hinting if needed, though invoke takes a dict

# --- Helper Functions ---
def save_uploaded_image(uploaded_file_obj):
    """Saves an uploaded image to a temporary path and returns the path."""
    if uploaded_file_obj is None:
        return None
    try:
        img = Image.open(uploaded_file_obj)
        temp_filename = f"temp_streamlit_capture_{uuid.uuid4()}.jpg"
        
        # Ensure the directory exists (e.g., if running in a restricted env)
        os.makedirs("temp_images", exist_ok=True)
        temp_image_path = os.path.join("temp_images", temp_filename)
        
        img.save(temp_image_path)
        return temp_image_path
    except Exception as e:
        st.error(f"Error saving image: {e}")
        return None

def convert_streamlit_history_to_graph_history(st_history):
    """Converts Streamlit message history to graph-compatible history."""
    graph_h = []
    user_msg_content = None
    # Iterate through a copy of history to avoid issues if modifying original
    for msg in list(st_history): 
        if msg["role"] == "user":
            user_msg_content = msg["content"]
        elif msg["role"] == "assistant" and user_msg_content is not None:
            # Ensure we are pairing a user text message with an assistant response
            if isinstance(user_msg_content, str): # Only pair if user_msg was text
                 graph_h.append((user_msg_content, msg["content"]))
            user_msg_content = None # Reset for next pair
        elif msg["role"] == "user_image":
            # User images are handled differently, not directly part of (human, ai) text history for the graph's LLM context
            user_msg_content = None # Reset if last user action was an image
    return graph_h

def capture_image_auto() -> tuple[str, Image.Image] | tuple[None, None]:
    """Captures image with webcam and returns (file_path, image_object)."""
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Cámara no accesible")
            st.warning("Jarvis no pudo acceder a la cámara.")
            return None, None
        
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            print("Error: No se pudo leer frame")
            st.warning("Jarvis no pudo capturar una imagen.")
            return None, None
        
        # Convert OpenCV BGR to RGB for PIL
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # Save to file 
        os.makedirs("temp_images", exist_ok=True)
        path = f"temp_images/auto_capture_{uuid.uuid4()}.jpg"
        success = cv2.imwrite(path, frame)
        
        if success:
            print(f"Imagen guardada en: {path}")
            return path, pil_image
        else:
            print("Error al guardar imagen")
            return None, None
            
    except Exception as e:
        print(f"Error en capture_image_auto: {e}")
        return None, None

# --- Streamlit App ---
st.set_page_config(page_title="Jarvis - Asistente Robótico", layout="centered")
st.title("🤖 Jarvis")
st.caption("Tu asistente robótico para análisis y conversación, ahora en Streamlit.")

# Initialize session state variables
if "history" not in st.session_state:
    st.session_state.history = [] # List of dicts: {"role": "user/assistant/system/user_image", "content": "text_or_image_data"}
if "graph_app" not in st.session_state:
    # Ensure GOOGLE_APPLICATION_CREDENTIALS is set before compiling the graph
    # This should ideally be managed by config.py being imported in graph_llm.py
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS") and hasattr(config, 'CREDENTIALS_FILE_PATH'):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = config.CREDENTIALS_FILE_PATH
    st.session_state.graph_app = jarvis_graph_app # Use the corrected graph app name
if "awaiting_image_for_prompt" not in st.session_state:
    st.session_state.awaiting_image_for_prompt = None # Stores the user_input that needs an image
if "image_processed_this_cycle" not in st.session_state:
    st.session_state.image_processed_this_cycle = False

# Display chat messages from history
for message in st.session_state.history:
    with st.chat_message(message["role"] if message["role"] != "user_image" else "user"):
        if message["role"] == "user_image":
            # Content could be either a path (string) or an image object
            if isinstance(message["content"], str):
                # If it's a path that exists
                if os.path.exists(message["content"]):
                    st.image(message["content"], caption="Imagen", use_container_width=True)
                else:
                    st.warning("Imagen ya no disponible")
            else:
                # Assume it's a PIL Image object
                st.image(message["content"], caption="Imagen", use_container_width=True)
        else:
            st.markdown(message["content"])

# User inputs
user_text_prompt = st.chat_input("Habla con Jarvis...")

# --- Main Logic ---

# 1. Handle text input from the user
if user_text_prompt:
    st.session_state.history.append({"role": "user", "content": user_text_prompt})
    with st.chat_message("user"):
        st.markdown(user_text_prompt)
    
    st.session_state.image_processed_this_cycle = False

    initial_graph_state = {
        "user_input": user_text_prompt,
        "history": convert_streamlit_history_to_graph_history(st.session_state.history[:-1]),
        "image_path": None,
        "classification": None, "vision_analysis": None, "final_response": None, "error": None
    }
    
    with st.spinner("Jarvis está pensando..."):
        final_state = st.session_state.graph_app.invoke(initial_graph_state)

    ai_response = final_state.get("final_response", "Lo siento, no pude procesar tu solicitud.")
    st.session_state.history.append({"role": "assistant", "content": ai_response})
    with st.chat_message("assistant"):
        st.markdown(ai_response)

    if final_state.get("classification") == "describe_image":
        st.info("Jarvis necesita ver. Capturando imagen...")
        img_path, img_object = capture_image_auto()

        if img_path and img_object:
            # Store the PIL Image object, not just the path
            st.session_state.history.append({
                "role": "user_image", 
                "content": img_object  # Store the actual image object
            })
            with st.chat_message("user"):
                st.image(img_object, caption="Imagen capturada", use_container_width=True)

            second_state = {
                "user_input": user_text_prompt, # The original prompt that led to image description
                "history": convert_streamlit_history_to_graph_history(st.session_state.history[:-1]), # History before this auto-image
                "image_path": img_path,
                "classification": "describe_image", # We know the intent now
                "vision_analysis": None, "final_response": None, "error": None
            }
            with st.spinner("Jarvis está analizando la imagen..."):
                second_final = st.session_state.graph_app.invoke(second_state)

            ai_img_resp = second_final.get("final_response", "No pude describir la imagen.")
            st.session_state.history.append({"role": "assistant", "content": ai_img_resp})
            with st.chat_message("assistant"):
                st.markdown(ai_img_resp)
            
            # After processing, delete the file but we've already kept the image in memory
            if os.path.exists(img_path):
                try:
                    os.remove(img_path)
                    print(f"Imagen temporal eliminada: {img_path}")
                except Exception as e:
                    print(f"Error al eliminar: {e}")
        else:
            # capture_image_auto returned None, message already sent by it or graph node will handle
            # Potentially add a generic "Could not capture image" to chat if not already handled by capture_image_auto's st.warnings
            if not any("Jarvis no pudo" in m["content"] for m in st.session_state.history[-2:]): # Avoid double error msgs
                st.warning("Jarvis no pudo capturar la imagen para el análisis.")


    # We don't need to rerun or manage camera_key_suffix anymore as there's no user-facing camera widget.
    # However, a rerun might be desirable to refresh state if other complex interactions occur.
    # For now, let's remove the st.experimental_rerun() unless a specific need arises.