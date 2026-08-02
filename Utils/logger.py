import logging
import warnings

# ==========================================
# 1. GLOBAL LOG & WARNING SUPPRESSION
# ==========================================

warnings.filterwarnings("ignore", message="Accessing `__path__` from .*")
warnings.filterwarnings("ignore", message="Deserializing unregistered type .*")

for noisy_logger in ["transformers", "huggingface_hub", "urllib3", "httpx", "absl"]:
    logging.getLogger(noisy_logger).setLevel(logging.ERROR)

logging.getLogger().setLevel(logging.WARNING)

# ==========================================
# 2. CUSTOM LOGGER
# ==========================================

def get_logger(name):
    logger = logging.getLogger(name)
    
    # Only add the handler if the logger doesn't already have one
    if not logger.handlers:
        handler = logging.StreamHandler()
        
        # --- THE UPDATED FORMATTER ---
        # %(levelname)s = INFO, WARNING, ERROR, etc.
        # %(name)s      = The string you passed into get_logger()
        # %(message)s   = The actual text you are logging
        # %(lineno)d    = The exact line number in the file where logger.info() was called
        formatter = logging.Formatter('%(levelname)s | %(name)s | %(message)s | Line %(lineno)d')
        
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        # Keep our application loggers at INFO level
        logger.setLevel(logging.INFO)
        
        # Prevent these logs from bubbling up to the root logger (avoids duplicate prints)
        logger.propagate = False 
        
    return logger