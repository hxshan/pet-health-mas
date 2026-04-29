from crewai.tools import BaseTool
from app.llm.ollama_client import chat_json
from app.llm.prompts import intake_messages


class EntityExtractorTool(BaseTool):
    name: str = "entity_extractor"
    description: str = "Extract structured pet case using LLM"

    def _run(self, input_str: str):

        messages = intake_messages(input_str)

        result = chat_json(messages)

        if not isinstance(result, dict):
            return self._fallback()

        result.setdefault("species", "unknown")
        result.setdefault("breed", "unknown")
        result.setdefault("age", "unknown")
        result.setdefault("sex", "unknown")
        result.setdefault("weight", "unknown")
        result.setdefault("raw_symptoms", [])

        return result

    def _fallback(self):
        return {
            "species": "unknown",
            "breed": "unknown",
            "age": "unknown",
            "sex": "unknown",
            "weight": "unknown",
            "raw_symptoms": []
        }