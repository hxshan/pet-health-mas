"""
Image Classifier Tool
=====================
A CrewAI tool that runs a locally-hosted MobileNetV3 CNN to classify dog skin
conditions from an image file. The model was trained to recognise 7 classes:

    Dermatitis | Fungal_infections | Healthy | Hypersensitivity |
    Invalid | demodicosis | ringworm

The "Invalid" class indicates the model believes the submitted image is not a
recognisable pet-skin photograph.

Usage (JSON input):
    {"image_path": "/absolute/or/relative/path/to/image.jpg"}

Returns a JSON string with the following schema:
    {
        "image_prediction":   str,   # top predicted class
        "confidence":         float, # probability of top class (0.0–1.0)
        "alternatives":       [{"condition": str, "confidence": float}],
        "uncertainty_flag":   bool,  # True when confidence is low or top-2 gap is small
        "possible_out_of_scope": bool,
        "image_validity":     str,   # "valid" | "unusable"
        "local_interpretation": str
    }
"""

from __future__ import annotations

import json
import os
from io import BytesIO
from typing import Any, Dict, List, Optional

import numpy as np
from crewai.tools import BaseTool
from PIL import Image, ImageOps

from app.config import settings

# ---------------------------------------------------------------------------
# Class labels — must match the training order of dog_skin_cnn_mobilenet3.keras
# ---------------------------------------------------------------------------
_CLASS_NAMES: List[str] = [
    "Dermatitis",
    "Fungal_infections",
    "Healthy",
    "Hypersensitivity",
    "Invalid",
    "demodicosis",
    "ringworm",
]

# ---------------------------------------------------------------------------
# Lazy model singleton — loaded once on first inference call
# ---------------------------------------------------------------------------
_cnn_model = None
_model_load_error: Optional[str] = None


def _get_model():
    """Load the Keras CNN model lazily and cache it as a module-level singleton.

    Returns:
        The loaded Keras model instance.

    Raises:
        RuntimeError: If the model file cannot be found or loaded.
    """
    global _cnn_model, _model_load_error

    if _cnn_model is not None:
        return _cnn_model

    model_path = settings.CNN_MODEL_PATH
    if not os.path.exists(model_path):
        _model_load_error = f"CNN model file not found at: {model_path}"
        raise RuntimeError(_model_load_error)

    try:
        # Import TensorFlow here to avoid slow startup when tool is not used
        from tensorflow import keras  # type: ignore

        _cnn_model = keras.models.load_model(model_path)
        _model_load_error = None
        return _cnn_model
    except Exception as exc:
        _model_load_error = str(exc)
        raise RuntimeError(f"Failed to load CNN model: {exc}") from exc


def _preprocess_image(image_bytes: bytes, img_size: tuple[int, int] = (224, 224)) -> np.ndarray:
    """Preprocess raw image bytes into the input tensor expected by the CNN.

    Steps:
        1. Decode bytes with PIL and force RGB colour space.
        2. Correct EXIF orientation so the model always sees upright images.
        3. Centre-crop and resize to ``img_size`` using bilinear interpolation.
        4. Cast to float32 and add the batch dimension.

    Args:
        image_bytes: Raw file bytes of any PIL-supported image format.
        img_size: Target (width, height) for the model input. Defaults to (224, 224).

    Returns:
        NumPy array of shape (1, height, width, 3) and dtype float32.
    """
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    img = ImageOps.exif_transpose(img)
    img = ImageOps.fit(img, img_size, method=Image.BILINEAR)
    x = np.array(img, dtype=np.float32)
    x = np.expand_dims(x, axis=0)  # (1, 224, 224, 3)
    return x


def _build_output(
    probs: np.ndarray,
    top_k: int = 3,
) -> Dict[str, Any]:
    """Apply confidence thresholds and build the tool output dictionary.

    Args:
        probs: Softmax probability array of shape (num_classes,).
        top_k: Number of alternative predictions to include. Defaults to 3.

    Returns:
        Dictionary matching the ImageClassifierTool output schema.
    """
    sorted_idx = np.argsort(probs)[::-1]

    top_class: str = _CLASS_NAMES[int(sorted_idx[0])]
    top_conf: float = float(probs[int(sorted_idx[0])])

    alternatives: List[Dict[str, Any]] = [
        {"condition": _CLASS_NAMES[int(i)], "confidence": float(probs[int(i)])}
        for i in sorted_idx[1:top_k]
    ]

    # --- Validity ---------------------------------------------------------
    is_invalid_class = top_class == "Invalid"
    image_validity = "unusable" if is_invalid_class else "valid"

    # --- Uncertainty flags ------------------------------------------------
    low_confidence = top_conf < settings.IMAGE_CONFIDENCE_THRESHOLD

    second_conf = float(probs[int(sorted_idx[1])]) if len(sorted_idx) > 1 else 0.0
    close_gap = (top_conf - second_conf) < settings.UNCERTAINTY_GAP_THRESHOLD

    uncertainty_flag = low_confidence or close_gap or is_invalid_class
    possible_out_of_scope = is_invalid_class

    # --- Human-readable interpretation ------------------------------------
    if is_invalid_class:
        interpretation = (
            "The submitted image does not appear to be a recognisable pet-skin "
            "photograph. Image assessment cannot be completed. Possible out-of-scope case."
        )
    elif uncertainty_flag:
        interpretation = (
            f"Low confidence result. Top prediction '{top_class}' has confidence "
            f"{top_conf:.2f}, which is below the reliability threshold or too close "
            "to alternatives. Veterinary consultation recommended."
        )
    else:
        interpretation = (
            f"Within supported conditions. Top prediction '{top_class}' returned with "
            f"confidence {top_conf:.2f}. Result is considered reliable."
        )

    return {
        "image_prediction": top_class,
        "confidence": round(top_conf, 4),
        "alternatives": alternatives,
        "uncertainty_flag": uncertainty_flag,
        "possible_out_of_scope": possible_out_of_scope,
        "image_validity": image_validity,
        "local_interpretation": interpretation,
    }


# ---------------------------------------------------------------------------
# CrewAI Tool
# ---------------------------------------------------------------------------

class ImageClassifierTool(BaseTool):
    """CrewAI tool that classifies a pet skin image using a local MobileNetV3 CNN.

    The tool accepts a JSON string with an ``image_path`` key, preprocesses the
    image, runs it through the CNN, and returns a structured JSON assessment
    including the top predicted skin condition, confidence score, alternative
    predictions, and uncertainty / validity flags.

    Supported skin condition classes:
        Dermatitis, Fungal_infections, Healthy, Hypersensitivity,
        Invalid, demodicosis, ringworm

    Input JSON schema::

        {"image_path": "<path to image file>"}

    Output JSON schema::

        {
            "image_prediction":      str,
            "confidence":            float,
            "alternatives":          [{"condition": str, "confidence": float}],
            "uncertainty_flag":      bool,
            "possible_out_of_scope": bool,
            "image_validity":        str,   # "valid" | "unusable"
            "local_interpretation":  str
        }
    """

    name: str = "image_classifier"
    description: str = (
        "Classifies a pet skin image using a locally-hosted MobileNetV3 CNN. "
        "Input: JSON string with key 'image_path' pointing to the image file. "
        "Output: JSON string with predicted skin condition, confidence, alternatives, "
        "uncertainty flag, validity status, and a local interpretation summary."
    )

    def _run(self, input_str: str) -> str:
        """Run CNN inference on the image at the given path.

        Args:
            input_str: JSON string with schema ``{"image_path": "<path>"}``.

        Returns:
            JSON string containing the classification result and confidence metadata.
            On error, returns a JSON string with ``"image_validity": "unusable"``
            and an ``"error"`` field describing what went wrong.
        """
        # --- Parse input --------------------------------------------------
        try:
            payload: Dict[str, Any] = json.loads(input_str)
            image_path: str = payload["image_path"]
        except (json.JSONDecodeError, KeyError) as exc:
            return json.dumps({
                "image_prediction": "unknown",
                "confidence": 0.0,
                "alternatives": [],
                "uncertainty_flag": True,
                "possible_out_of_scope": False,
                "image_validity": "unusable",
                "local_interpretation": "Tool input parsing failed.",
                "error": str(exc),
            })

        # --- Security: reject path traversal ------------------------------
        normalised = os.path.normpath(image_path)
        if ".." in normalised.split(os.sep):
            return json.dumps({
                "image_prediction": "unknown",
                "confidence": 0.0,
                "alternatives": [],
                "uncertainty_flag": True,
                "possible_out_of_scope": False,
                "image_validity": "unusable",
                "local_interpretation": "Rejected: path traversal detected.",
                "error": "Path traversal not permitted.",
            })

        # --- File existence check -----------------------------------------
        if not os.path.isfile(image_path):
            return json.dumps({
                "image_prediction": "unknown",
                "confidence": 0.0,
                "alternatives": [],
                "uncertainty_flag": True,
                "possible_out_of_scope": False,
                "image_validity": "unusable",
                "local_interpretation": f"Image file not found: {image_path}",
                "error": f"File not found: {image_path}",
            })

        # --- Load image bytes ---------------------------------------------
        try:
            with open(image_path, "rb") as fh:
                image_bytes: bytes = fh.read()
        except OSError as exc:
            return json.dumps({
                "image_prediction": "unknown",
                "confidence": 0.0,
                "alternatives": [],
                "uncertainty_flag": True,
                "possible_out_of_scope": False,
                "image_validity": "unusable",
                "local_interpretation": "Could not read image file.",
                "error": str(exc),
            })

        # --- Preprocess ---------------------------------------------------
        try:
            x = _preprocess_image(image_bytes)
        except Exception as exc:
            return json.dumps({
                "image_prediction": "unknown",
                "confidence": 0.0,
                "alternatives": [],
                "uncertainty_flag": True,
                "possible_out_of_scope": False,
                "image_validity": "unusable",
                "local_interpretation": "Image preprocessing failed — file may be corrupted or unsupported.",
                "error": str(exc),
            })

        # --- Model inference ----------------------------------------------
        try:
            model = _get_model()
            probs: np.ndarray = model.predict(x, verbose=0)[0]
        except RuntimeError as exc:
            return json.dumps({
                "image_prediction": "unknown",
                "confidence": 0.0,
                "alternatives": [],
                "uncertainty_flag": True,
                "possible_out_of_scope": False,
                "image_validity": "unusable",
                "local_interpretation": "CNN model is unavailable.",
                "error": str(exc),
            })

        # --- Build and return output --------------------------------------
        result = _build_output(probs, top_k=settings.MAX_ALTERNATIVES)
        return json.dumps(result)
