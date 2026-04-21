"""
Evaluation Script — Image Assessment Agent (Agent 3)
=====================================================
Tests the ImageClassifierTool and the image_agent.run() function.

Coverage:
  1. Schema validation  — output always contains all required keys
  2. Unusable image     — nonexistent file → image_validity="unusable"
  3. Corrupt image      — non-image bytes → image_validity="unusable"
  4. Low confidence     — mocked low-prob output → uncertainty_flag=True
  5. Close gap          — mocked top-2 close probs → uncertainty_flag=True
  6. "Invalid" class    — model top class is Invalid → possible_out_of_scope=True
  7. Happy path         — high confidence valid prediction → uncertainty_flag=False
  8. Security           — path traversal input → rejected with image_validity="unusable"
  9. Skip on no image   — agent.run() skips if image_available=False
  10. State update      — agent.run() writes image_assessment into state

Run with:
    cd backend
    python -m pytest tests/test_image_agent.py -v
"""

from __future__ import annotations

import json
import os
import tempfile
import types
import unittest
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REQUIRED_KEYS = {
    "image_prediction",
    "confidence",
    "alternatives",
    "uncertainty_flag",
    "possible_out_of_scope",
    "image_validity",
    "local_interpretation",
}


def _parse_tool_output(raw: str) -> Dict[str, Any]:
    """Parse tool JSON output and assert it is valid JSON."""
    data = json.loads(raw)
    assert isinstance(data, dict), "Tool output must be a JSON object"
    return data


def _make_probs(top_class_idx: int, top_conf: float, spread_rest: bool = True) -> np.ndarray:
    """Build a synthetic probability array of length 7 with a specified top class."""
    probs = np.zeros(7, dtype=np.float32)
    probs[top_class_idx] = top_conf
    remaining = 1.0 - top_conf
    others = [i for i in range(7) if i != top_class_idx]
    if spread_rest:
        for i, idx in enumerate(others):
            probs[idx] = remaining / len(others)
    else:
        # Put most of the remaining mass on index adjacent to top → small gap
        probs[others[0]] = remaining * 0.9
        for idx in others[1:]:
            probs[idx] = remaining * 0.1 / (len(others) - 1)
    return probs


# ---------------------------------------------------------------------------
# Unit tests for ImageClassifierTool
# ---------------------------------------------------------------------------

class TestImageClassifierToolSchema(unittest.TestCase):
    """All tool outputs must contain the full required schema."""

    def setUp(self) -> None:
        # Import here so TF import errors surface clearly
        from app.tools.image_tools import ImageClassifierTool
        self.tool = ImageClassifierTool()

    def _assert_schema(self, raw: str) -> Dict[str, Any]:
        data = _parse_tool_output(raw)
        missing = REQUIRED_KEYS - set(data.keys())
        self.assertEqual(missing, set(), f"Missing keys in tool output: {missing}")
        return data

    def test_nonexistent_file_returns_unusable(self) -> None:
        """A path that does not exist must return image_validity='unusable'."""
        raw = self.tool._run(json.dumps({"image_path": "/nonexistent/path/image.jpg"}))
        data = self._assert_schema(raw)
        self.assertEqual(data["image_validity"], "unusable")
        self.assertTrue(data["uncertainty_flag"])

    def test_corrupt_image_returns_unusable(self) -> None:
        """Bytes that are not a valid image must return image_validity='unusable'."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as fh:
            fh.write(b"not an image at all")
            tmp_path = fh.name
        try:
            raw = self.tool._run(json.dumps({"image_path": tmp_path}))
            data = self._assert_schema(raw)
            self.assertEqual(data["image_validity"], "unusable")
        finally:
            os.unlink(tmp_path)

    def test_path_traversal_rejected(self) -> None:
        """Path traversal attempts must be rejected."""
        raw = self.tool._run(json.dumps({"image_path": "../../etc/passwd"}))
        data = self._assert_schema(raw)
        self.assertEqual(data["image_validity"], "unusable")
        self.assertIn("traversal", data.get("error", "").lower())

    def test_bad_json_input(self) -> None:
        """Malformed JSON input must not raise an exception."""
        raw = self.tool._run("not json at all")
        data = self._assert_schema(raw)
        self.assertEqual(data["image_validity"], "unusable")

    def test_missing_image_path_key(self) -> None:
        """JSON without 'image_path' key must return graceful error."""
        raw = self.tool._run(json.dumps({"wrong_key": "value"}))
        data = self._assert_schema(raw)
        self.assertEqual(data["image_validity"], "unusable")


class TestImageClassifierToolInference(unittest.TestCase):
    """Tests for the CNN confidence + uncertainty logic using mocked model output."""

    def setUp(self) -> None:
        from app.tools.image_tools import ImageClassifierTool
        self.tool = ImageClassifierTool()

    def _run_with_mocked_probs(self, probs: np.ndarray) -> Dict[str, Any]:
        """Create a real (tiny) PNG, mock model.predict to return probs, run tool."""
        from PIL import Image as _PIL

        # Write a minimal valid PNG
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as fh:
            img = _PIL.new("RGB", (50, 50), color=(120, 80, 60))
            img.save(fh, format="PNG")
            tmp_path = fh.name

        try:
            mock_model = MagicMock()
            mock_model.predict.return_value = np.array([probs])

            with patch("app.tools.image_tools._get_model", return_value=mock_model):
                raw = self.tool._run(json.dumps({"image_path": tmp_path}))
            return json.loads(raw)
        finally:
            os.unlink(tmp_path)

    def test_high_confidence_valid_prediction(self) -> None:
        """Confidence ≥ 0.55 with clear gap → uncertainty_flag=False, image_validity='valid'."""
        # Dermatitis=0, confidence=0.85 with spread rest
        probs = _make_probs(top_class_idx=0, top_conf=0.85, spread_rest=True)
        data = self._run_with_mocked_probs(probs)

        self.assertEqual(data["image_validity"], "valid")
        self.assertFalse(data["uncertainty_flag"])
        self.assertFalse(data["possible_out_of_scope"])
        self.assertEqual(data["image_prediction"], "Dermatitis")
        self.assertAlmostEqual(data["confidence"], 0.85, places=2)

    def test_low_confidence_sets_uncertainty_flag(self) -> None:
        """Confidence < 0.55 → uncertainty_flag=True."""
        probs = _make_probs(top_class_idx=1, top_conf=0.40, spread_rest=True)
        data = self._run_with_mocked_probs(probs)

        self.assertTrue(data["uncertainty_flag"])

    def test_close_top2_gap_sets_uncertainty_flag(self) -> None:
        """Gap between top-1 and top-2 < 0.10 → uncertainty_flag=True."""
        # top=0.52, second=0.46 → gap=0.06 < 0.10
        probs = _make_probs(top_class_idx=2, top_conf=0.52, spread_rest=False)
        data = self._run_with_mocked_probs(probs)

        self.assertTrue(data["uncertainty_flag"])

    def test_invalid_class_sets_out_of_scope(self) -> None:
        """'Invalid' as top prediction → possible_out_of_scope=True, image_validity='unusable'."""
        # "Invalid" is index 4
        probs = _make_probs(top_class_idx=4, top_conf=0.90, spread_rest=True)
        data = self._run_with_mocked_probs(probs)

        self.assertEqual(data["image_prediction"], "Invalid")
        self.assertEqual(data["image_validity"], "unusable")
        self.assertTrue(data["possible_out_of_scope"])
        self.assertTrue(data["uncertainty_flag"])

    def test_healthy_prediction(self) -> None:
        """'Healthy' class with high confidence returns valid result."""
        # "Healthy" is index 2
        probs = _make_probs(top_class_idx=2, top_conf=0.92, spread_rest=True)
        data = self._run_with_mocked_probs(probs)

        self.assertEqual(data["image_prediction"], "Healthy")
        self.assertEqual(data["image_validity"], "valid")
        self.assertFalse(data["uncertainty_flag"])

    def test_alternatives_count(self) -> None:
        """Output should include alternatives (top-k minus top-1)."""
        probs = _make_probs(top_class_idx=5, top_conf=0.80, spread_rest=True)
        data = self._run_with_mocked_probs(probs)

        self.assertIsInstance(data["alternatives"], list)
        self.assertGreater(len(data["alternatives"]), 0)
        for alt in data["alternatives"]:
            self.assertIn("condition", alt)
            self.assertIn("confidence", alt)

    def test_schema_present_on_happy_path(self) -> None:
        """Every required key must be present even on a successful inference."""
        probs = _make_probs(top_class_idx=3, top_conf=0.70, spread_rest=True)
        data = self._run_with_mocked_probs(probs)

        missing = REQUIRED_KEYS - set(data.keys())
        self.assertEqual(missing, set(), f"Missing keys: {missing}")


# ---------------------------------------------------------------------------
# Integration tests for image_agent.run()
# ---------------------------------------------------------------------------

class TestImageAgentRun(unittest.TestCase):
    """Tests for the agent.run() state machine logic."""

    def _make_base_state(self, image_available: bool = True, image_path: str = "") -> dict:
        return {
            "case_id": "test-case-001",
            "raw_text_input": "My dog has a red rash",
            "image_available": image_available,
            "image_path": image_path,
            "uncertainty_flags": [],
        }

    def test_skips_when_no_image(self) -> None:
        """Agent must return state unchanged when image_available=False."""
        from app.agents.image_agent import agent

        state = self._make_base_state(image_available=False)
        result = agent.run(state)

        self.assertNotIn("image_assessment", result)

    def test_unusable_when_path_missing(self) -> None:
        """Agent sets image_validity='unusable' when image_path does not exist."""
        from app.agents.image_agent import agent

        state = self._make_base_state(image_available=True, image_path="/does/not/exist.jpg")
        result = agent.run(state)

        self.assertIn("image_assessment", result)
        self.assertEqual(result["image_assessment"]["image_validity"], "unusable")

    def test_state_updated_on_success(self) -> None:
        """Agent writes image_assessment into state after successful inference."""
        from app.agents.image_agent import agent
        from PIL import Image as _PIL

        # Create a real PNG file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as fh:
            _PIL.new("RGB", (50, 50), color=(200, 100, 50)).save(fh, format="PNG")
            tmp_path = fh.name

        try:
            mock_model = MagicMock()
            mock_model.predict.return_value = np.array([
                _make_probs(top_class_idx=0, top_conf=0.88, spread_rest=True)
            ])

            fake_llm_response = json.dumps({
                "image_prediction": "Dermatitis",
                "confidence": 0.88,
                "alternatives": [],
                "uncertainty_flag": False,
                "possible_out_of_scope": False,
                "image_validity": "valid",
                "local_interpretation": "Within supported conditions.",
            })

            state = self._make_base_state(image_available=True, image_path=tmp_path)

            with patch("app.tools.image_tools._get_model", return_value=mock_model), \
                 patch("app.agents.image_agent.agent.ollama_client.chat", return_value=fake_llm_response):
                result = agent.run(state)

            self.assertIn("image_assessment", result)
            self.assertEqual(result["image_assessment"]["image_validity"], "valid")
        finally:
            os.unlink(tmp_path)

    def test_uncertainty_flag_propagates_to_state(self) -> None:
        """When uncertainty_flag=True, 'image_low_confidence' is added to uncertainty_flags."""
        from app.agents.image_agent import agent
        from PIL import Image as _PIL

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as fh:
            _PIL.new("RGB", (50, 50)).save(fh, format="PNG")
            tmp_path = fh.name

        try:
            mock_model = MagicMock()
            # Low confidence → uncertainty_flag will be True
            mock_model.predict.return_value = np.array([
                _make_probs(top_class_idx=1, top_conf=0.35, spread_rest=True)
            ])

            fake_llm_response = json.dumps({
                "image_prediction": "Fungal_infections",
                "confidence": 0.35,
                "alternatives": [],
                "uncertainty_flag": True,
                "possible_out_of_scope": False,
                "image_validity": "valid",
                "local_interpretation": "Low confidence result.",
            })

            state = self._make_base_state(image_available=True, image_path=tmp_path)

            with patch("app.tools.image_tools._get_model", return_value=mock_model), \
                 patch("app.agents.image_agent.agent.ollama_client.chat", return_value=fake_llm_response):
                result = agent.run(state)

            self.assertIn("image_low_confidence", result.get("uncertainty_flags", []))
        finally:
            os.unlink(tmp_path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
