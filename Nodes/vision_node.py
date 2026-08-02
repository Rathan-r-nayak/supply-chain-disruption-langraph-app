import mimetypes
import base64
from langchain_core.messages import HumanMessage
from State.supply_chain_state import SupplyChainState
from Utils.Logger import get_logger
from Config.llm_config import primary_llm

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
                        "You are a Supply Chain Vision AI. Analyze this image. "
                        "1. If it's a map or traffic screenshot, extract the route, bottlenecks, or weather conditions. "
                        "2. If it's a Bill of Lading, weigh station receipt, or cargo manifest, extract the IDs, weights, and locations. "
                        "3. If it's a vehicle dashboard error or mechanical issue, clearly state the warning symbols or text. "
                        "Focus entirely on actionable logistics data."
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

def vision_node(state: SupplyChainState):
    """Checks for an image, analyzes it, and appends the context to the question."""
    logger.info("--- 👁️ RUNNING VISION NODE ---")
    
    question = state.get("question", "")
    image_path = state.get("image_path")

    if not image_path:
        return {}

    image_context = analyze_image_context(image_path)
    enriched_question = f"{question}\n\n--- ATTACHED IMAGE CONTEXT ---\n{image_context}"
    return {"question": enriched_question}