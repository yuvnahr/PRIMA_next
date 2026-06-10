"""Action execution context and audit records."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from action.execution_policy import ExecutionPolicy


@dataclass(frozen=True, slots=True)
class ActionAuditRecord:
    """Immutable audit record for action and tool execution."""

    event: str
    message: str
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Serialize the audit record into plain Python values."""
        return {
            "event": self.event,
            "message": self.message,
            "metadata": dict(self.metadata),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass(slots=True)
class ActionContext:
    """Workflow-provided execution context for the action layer."""

    plan: Any
    cognitive_state: Any | None = None
    world_prediction: Any | None = None
    uncertainty: Any | None = None
    policy: ExecutionPolicy = field(default_factory=ExecutionPolicy)
    metadata: dict[str, Any] = field(default_factory=dict)
    audit_log: list[ActionAuditRecord] = field(default_factory=list)

    def audit(self, event: str, message: str, metadata: dict[str, Any] | None = None) -> ActionAuditRecord:
        """Append and return an audit record when auditing is enabled."""
        record = ActionAuditRecord(event=event, message=message, metadata=metadata or {})
        if self.policy.audit_enabled:
            self.audit_log.append(record)
        return record

    def to_dict(self) -> dict[str, Any]:
        """Serialize the context into plain Python values."""
        return {
            "plan_id": str(getattr(self.plan, "plan_id", "")),
            "policy": self.policy.to_dict(),
            "metadata": dict(self.metadata),
            "audit_log": [record.to_dict() for record in self.audit_log],
        }
