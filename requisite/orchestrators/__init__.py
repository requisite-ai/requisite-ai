"""
Multi-agent orchestration backends.

Backend SDKs (``langgraph``, ...) are imported lazily by each backend
module, so importing ``requisite.orchestrators`` never requires every
optional orchestration framework to be installed.
"""

from requisite.orchestrators.base import BaseOrchestrator, WorkflowResult
from requisite.orchestrators.factory import OrchestratorRegistry, default_registry

__all__ = ["BaseOrchestrator", "WorkflowResult", "OrchestratorRegistry", "default_registry"]
