import mimetypes
import base64
from langchain_core.messages import HumanMessage
from State.banking_state import BankingState
from Utils.Logger import get_logger
from Config.llm_config import primary_llm  # Make sure you use a vision-capable model (e.g., gpt-4o)

logger = get_logger("VISION_NODE")

def analyze_image_context(image_path: str) -> str:
    """Uses the Vision LLM to analyze images and extract context."""
    try:
        mime_type, _ = mimetypes.guess_type(image_path)
        mime_type = mime_type or "image/jpeg"

        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            
        message = HumanMessage(
            content=[
                {
                    "type": "text", 
                    "text": (
                        # 🌟 NOTE: I uimage_pathpdated your prompt to fit a Banking context instead of IT Support
                        "You are a Banking Vision AI. Analyze this image. "
                        "1. If it's a screenshot of an app error, extract the exact error codes and text. "
                        "2. If it's a receipt or transaction screenshot, extract amounts, dates, and account numbers. "
                        "3. If it's an ID document, list the visible text clearly. "
                        "Focus entirely on actionable text and data. Do not describe aesthetic elements."
                    )
                },
                {
                    "type": "image_url", 
                    "image_url": {"url": f"data:{mime_type};base64,{encoded_string}"}
                }
            ]
        )
        
        response = primary_llm.invoke([message])
        return response.content
        
    except Exception as e:
        logger.error(f"Vision Error: {e}")
        return "System failed to analyze the attached image."

def vision_node(state: BankingState):
    """Checks for an image, analyzes it, and appends the context to the question."""
    logger.info("--- 👁️ RUNNING VISION NODE ---")
    
    question = state.get("question", "")
    image_path = state.get("image_path")

    # If no image was uploaded, just pass the state through unmodified
    if not image_path:
        logger.info("No image detected. Skipping vision analysis.")
        return {}

    logger.info(f"Image detected at {image_path}. Analyzing...")
    image_context = analyze_image_context(image_path)
    
    # 🌟 Append the image text to the user's question so the downstream nodes can read it
    enriched_question = f"{question}\n\n--- ATTACHED IMAGE CONTEXT ---\n{image_context}"
    
    logger.info("✅ Image context successfully appended to question.")
    
    return {"question": enriched_question}