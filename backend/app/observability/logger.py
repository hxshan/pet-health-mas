import logging
from typing import Any, Dict


def get_logger(name: str) -> logging.Logger:
    """Return a standard Python logger configured with a console handler."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def log_event(event_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    return {"event_type": event_type, "payload": payload}
