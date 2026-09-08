"""API routers for Agentic Viewer."""

from __future__ import annotations

from agentic_viewer.routers.chat import router as chat_router
from agentic_viewer.routers.datasets import router as datasets_router
from agentic_viewer.routers.evaluations import router as evaluations_router
from agentic_viewer.routers.ground_truth import router as ground_truth_router
from agentic_viewer.routers.prompts import router as prompts_router
from agentic_viewer.routers.runs import router as runs_router
from agentic_viewer.routers.wrong_cases import router as wrong_cases_router

__all__ = [
    "chat_router",
    "datasets_router",
    "evaluations_router",
    "ground_truth_router",
    "prompts_router",
    "runs_router",
    "wrong_cases_router",
]
