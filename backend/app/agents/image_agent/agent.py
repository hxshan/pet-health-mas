"""
Image Assessment Agent
-----------------------
Responsibilities:
  - Guard: skip entirely if image_available is False
  - Validate the image file is accessible
  - Run ImageClassifierTool to get CNN predictions
  - Pass classifier output to the Ollama LLM for bounded reasoning
  - Populate state: image_assessment, uncertainty_flags, possible_out_of_scope
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict

from app.llm import ollama_client
from app.llm.prompts import image_assessment_messages
from app.observability.logger import get_logger
from app.state.schema import PetCaseState
from app.tools.image_tools import ImageClassifierTool

logger = get_logger(__name__)


def run(state: PetCaseState) -> PetCaseState:
    """Execute the Image Assessment Agent pipeline step.

    Checks whether an image was submitted, runs CNN classification via
    ``ImageClassifierTool``, then asks the Ollama LLM to reason over the
    classifier output and produce a structured image assessment.

    Args:
        state: Shared ``PetCaseState`` dictionary populated by earlier agents.

    Returns:
        Updated ``PetCaseState`` with ``image_assessment`` populated.
        If no image is available the state is returned unchanged.
    """
    case_id = state.get("case_id", "unknown")
    logger.info("[ImageAgent][%s] Starting image assessment", case_id)

    # ------------------------------------------------------------------
    # Guard: skip if no image was provided
    # ------------------------------------------------------------------
    if not state.get("image_available", False):
        logger.info("[ImageAgent][%s] No image provided — skipping", case_id)
        return state

    image_path: str | None = state.get("image_path")
    if not image_path or not os.path.isfile(image_path):
        logger.warning("[ImageAgent][%s] Image path invalid or missing: %s", case_id, image_path)
        state["image_assessment"] = {
            "image_prediction": "unknown",
            "confidence": 0.0,
            "alternatives": [],
            "uncertainty_flag": True,
            "possible_out_of_scope": False,
            "image_validity": "unusable",
            "local_interpretation": "Image file could not be found or accessed.",
        }
        return state

    # ------------------------------------------------------------------
    # Build image metadata (format + dimensions via PIL)
    # ------------------------------------------------------------------
    image_meta: Dict[str, Any] = {"path": image_path}
    try:
        from PIL import Image as _PIL
        with _PIL.open(image_path) as img:
            image_meta["format"] = img.format or "unknown"
            image_meta["width"], image_meta["height"] = img.size
            image_meta["mode"] = img.mode
    except Exception as exc:
        logger.warning("[ImageAgent][%s] Could not read image metadata: %s", case_id, exc)
        image_meta["format"] = "unknown"
        image_meta["width"] = None
        image_meta["height"] = None

    # ------------------------------------------------------------------
    # Run CNN classifier tool
    # ------------------------------------------------------------------
    logger.info("[ImageAgent][%s] Running ImageClassifierTool on %s", case_id, image_path)
    tool = ImageClassifierTool()
    raw_tool_output: str = tool._run(json.dumps({"image_path": image_path}))

    try:
        classifier_result: Dict[str, Any] = json.loads(raw_tool_output)
    except json.JSONDecodeError:
        logger.error("[ImageAgent][%s] Tool returned non-JSON: %s", case_id, raw_tool_output)
        classifier_result = {
            "image_prediction": "unknown",
            "confidence": 0.0,
            "alternatives": [],
            "uncertainty_flag": True,
            "possible_out_of_scope": False,
            "image_validity": "unusable",
            "local_interpretation": "Classifier output could not be parsed.",
        }

    logger.info(
        "[ImageAgent][%s] Classifier result: prediction=%s confidence=%.2f uncertainty=%s",
        case_id,
        classifier_result.get("image_prediction"),
        classifier_result.get("confidence", 0.0),
        classifier_result.get("uncertainty_flag"),
    )

    # ------------------------------------------------------------------
    # LLM reasoning over the classifier output
    # ------------------------------------------------------------------
    messages = image_assessment_messages(image_meta, classifier_result)
    try:
        llm_response: str = ollama_client.chat(messages)
        assessment: Dict[str, Any] = json.loads(llm_response)
    except json.JSONDecodeError:
        logger.warning("[ImageAgent][%s] LLM returned non-JSON — using raw classifier result", case_id)
        assessment = classifier_result
    except RuntimeError as exc:
        logger.error("[ImageAgent][%s] Ollama unavailable: %s", case_id, exc)
        assessment = classifier_result

    # ------------------------------------------------------------------
    # Update shared state
    # ------------------------------------------------------------------
    state["image_assessment"] = assessment

    if assessment.get("uncertainty_flag"):
        flags: list = state.get("uncertainty_flags", [])
        flags.append("image_low_confidence")
        state["uncertainty_flags"] = flags

    if assessment.get("possible_out_of_scope"):
        state["possible_out_of_scope"] = True

    logger.info("[ImageAgent][%s] Image assessment complete: validity=%s", case_id, assessment.get("image_validity"))
    return state
