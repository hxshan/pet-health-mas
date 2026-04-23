from crewai.tools import BaseTool
from app.llm.ollama_client import chat_json
from app.llm.prompts import intake_messages


class EntityExtractorTool(BaseTool):
    name: str = "entity_extractor"
    description: str = "Extract structured pet case using LLM"

    def _run(self, input_str: str):

        messages = intake_messages(input_str)

        result = chat_json(messages)

        return result