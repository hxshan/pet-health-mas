"""
Intake / Entity Extractor Tool
================================
TODO (Intake Agent team): implement this tool.

Expected output schema from _run():
{
  "species": str,
  "breed": str,
  "age": str,
  "sex": str,
  "weight": str,
  "raw_symptoms": [str]
}
"""
from crewai.tools import BaseTool


class EntityExtractorTool(BaseTool):
    name: str = "entity_extractor"
    description: str = "TODO: describe what this tool does."

    def _run(self, input_str: str) -> str:
        raise NotImplementedError
