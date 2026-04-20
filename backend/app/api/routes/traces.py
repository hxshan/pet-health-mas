"""
Traces route — returns observability / event log for a case.
TODO: wire up to a persistent store once one is added.
"""
from fastapi import APIRouter

router = APIRouter()


@router.get("/{case_id}")
def get_traces(case_id: str):
    """
    Return trace events for a given case_id.
    TODO: look up events from a database or in-memory store.
    """
    return {"case_id": case_id, "trace_events": []}
