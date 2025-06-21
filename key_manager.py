import time
import hashlib
import logging

logger = logging.getLogger(__name__)

def is_key_valid(provided_key, chat_id):
    try:
        # Example key validation logic
        # Replace with your actual key validation mechanism
        valid_key = hashlib.sha256(f"{chat_id}{int(time.time() // 86400)}".encode()).hexdigest()[:10]
        if provided_key == valid_key:
            return True, "Key is valid"
        return False, "Invalid key provided"
    except Exception as e:
        logger.error(f"Error validating key for chat {chat_id}: {e}")
        return False, "Error validating key"
