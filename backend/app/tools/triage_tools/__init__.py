"""
Triage / Urgency Calculator Tool
===================================
TODO (Triage Agent team): implement this tool.

Expected output schema from _run():
{
  "urgency_level": str,     # "non-urgent" | "monitor" | "vet_soon" | "urgent"
  "recommendation": str
}
"""
from crewai.tools import BaseTool


class UrgencyCalculatorTool(BaseTool):
    name: str = "urgency_calculator"
    description: str = "TODO: describe what this tool does."

    def _run(self, input_str: str) -> str:
        raise NotImplementedError
