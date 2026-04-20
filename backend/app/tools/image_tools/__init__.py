"""
Image Classifier Tool
======================
TODO (Image Assessment team): implement this tool.

Expected output schema from _run():
{
  "image_prediction": str,
  "confidence": float,
  "alternatives": [{"condition": str, "confidence": float}],
  "uncertainty_flag": bool,
  "possible_out_of_scope": bool,
  "image_validity": str         # "valid" | "invalid" | "unusable"
}
"""
from crewai.tools import BaseTool


class ImageClassifierTool(BaseTool):
    name: str = "image_classifier"
    description: str = "TODO: describe what this tool does."

    def _run(self, input_str: str) -> str:
        raise NotImplementedError
