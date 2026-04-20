"""
Thin HTTP wrapper around the Ollama REST API.

Only the /api/chat endpoint is used so that every agent call follows the
standard chat-completion interface (system + user messages).
"""
import json
import logging
from typing import Any, Dict, List, Optional

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


def chat(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: float = 0.2,
    timeout: float = 120.0,
) -> str:
    """
    Send a list of chat messages to Ollama and return the assistant reply.

    Args:
        messages: List of {"role": ..., "content": ...} dicts.
        model:    Override the default model from settings.
        temperature: Sampling temperature (lower = more deterministic).
        timeout: HTTP request timeout in seconds.

    Returns:
        The raw text content of the assistant reply.

    Raises:
        RuntimeError: If the Ollama server is unreachable or returns an error.
    """
    model = model or settings.OLLAMA_MODEL
    url = f"{settings.OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }

    try:
        response = httpx.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        return data["message"]["content"]
    except httpx.ConnectError as exc:
        raise RuntimeError(
            f"Cannot reach Ollama at {settings.OLLAMA_BASE_URL}. "
            "Make sure Ollama is running (`ollama serve`)."
        ) from exc
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(
            f"Ollama returned HTTP {exc.response.status_code}: {exc.response.text}"
        ) from exc
    except KeyError as exc:
        raise RuntimeError(
            f"Unexpected Ollama response structure: {exc}"
        ) from exc


def chat_json(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: float = 0.1,
    timeout: float = 120.0,
) -> Dict[str, Any]:
    """
    Same as `chat` but parses and returns the response as a JSON dict.

    The prompt should instruct the model to respond ONLY with valid JSON.
    If parsing fails the raw text is returned under the key "raw_response".
    """
    raw = chat(messages, model=model, temperature=temperature, timeout=timeout)
    # Strip markdown code fences if the model wraps its JSON
    clean = raw.strip()
    if clean.startswith("```"):
        lines = clean.splitlines()
        # Remove first and last fence lines
        clean = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        logger.warning("Ollama response is not valid JSON. Raw: %s", raw[:200])
        return {"raw_response": raw}
