from typing import Any, Dict


def log_event(event_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    return {"event_type": event_type, "payload": payload}