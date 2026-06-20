"""Tool argument validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from action.tool_invocation import ToolInvocation

TYPE_MAP = {
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "dict": dict,
    "list": list,
}


@dataclass(frozen=True, slots=True)
class ToolParameterSpec:
    """Declarative parameter validation spec."""

    name: str
    parameter_type: str
    required: bool = True
    allowed_values: tuple[Any, ...] = ()
    min_value: float | None = None
    max_value: float | None = None
    min_length: int | None = None
    max_length: int | None = None
    allow_empty: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "parameter_type", str(self.parameter_type))
        object.__setattr__(self, "allowed_values", tuple(self.allowed_values))
        if self.min_length is not None:
            object.__setattr__(self, "min_length", max(0, int(self.min_length)))
        if self.max_length is not None:
            object.__setattr__(self, "max_length", max(0, int(self.max_length)))

    def to_dict(self) -> dict[str, Any]:
        """Serialize the parameter spec into plain Python values."""
        return {
            "name": self.name,
            "parameter_type": self.parameter_type,
            "required": self.required,
            "allowed_values": list(self.allowed_values),
            "min_value": self.min_value,
            "max_value": self.max_value,
            "min_length": self.min_length,
            "max_length": self.max_length,
            "allow_empty": self.allow_empty,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class ToolValidationResult:
    """Result of validating tool arguments."""

    is_valid: bool
    errors: tuple[str, ...] = ()
    sanitized_arguments: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "errors", tuple(str(error) for error in self.errors))
        object.__setattr__(self, "sanitized_arguments", dict(self.sanitized_arguments))

    def to_dict(self) -> dict[str, Any]:
        """Serialize the validation result into plain Python values."""
        return {
            "is_valid": self.is_valid,
            "errors": list(self.errors),
            "sanitized_arguments": dict(self.sanitized_arguments),
        }


class ToolValidator:
    """Validate tool invocations against registered schemas."""

    def validate(self, invocation: ToolInvocation, schema: tuple[ToolParameterSpec, ...]) -> ToolValidationResult:
        """Validate invocation arguments and return sanitized values."""
        errors: list[str] = []
        sanitized: dict[str, Any] = {}
        allowed_names = {spec.name for spec in schema}

        for unexpected_name in sorted(set(invocation.arguments) - allowed_names):
            errors.append(f"Unexpected argument '{unexpected_name}'.")

        for spec in schema:
            exists = spec.name in invocation.arguments
            if spec.required and not exists:
                errors.append(f"Missing required argument '{spec.name}'.")
                continue
            if not exists:
                continue

            value = invocation.arguments[spec.name]
            expected_type = TYPE_MAP.get(spec.parameter_type)
            if expected_type is None:
                errors.append(f"Unsupported parameter type '{spec.parameter_type}' for '{spec.name}'.")
                continue
            if not _matches_expected_type(value, expected_type):
                errors.append(f"Argument '{spec.name}' must be {spec.parameter_type}.")
                continue
            if spec.allowed_values and value not in spec.allowed_values:
                errors.append(f"Argument '{spec.name}' is not an allowed value.")
                continue
            if isinstance(value, int | float):
                if spec.min_value is not None and float(value) < spec.min_value:
                    errors.append(f"Argument '{spec.name}' is below minimum {spec.min_value}.")
                    continue
                if spec.max_value is not None and float(value) > spec.max_value:
                    errors.append(f"Argument '{spec.name}' exceeds maximum {spec.max_value}.")
                    continue
            length = len(value) if isinstance(value, str | list | dict) else None
            if length == 0 and not spec.allow_empty:
                errors.append(f"Argument '{spec.name}' cannot be empty.")
                continue
            if length is not None:
                if spec.min_length is not None and length < spec.min_length:
                    errors.append(f"Argument '{spec.name}' is shorter than {spec.min_length}.")
                    continue
                if spec.max_length is not None and length > spec.max_length:
                    errors.append(f"Argument '{spec.name}' is longer than {spec.max_length}.")
                    continue

            safe_value = _sanitize_value(value)
            if safe_value is _UNSAFE:
                errors.append(f"Argument '{spec.name}' contains unsupported runtime values.")
                continue
            sanitized[spec.name] = safe_value

        return ToolValidationResult(is_valid=not errors, errors=tuple(errors), sanitized_arguments=sanitized)


class _UnsafeValue:
    """Sentinel for values that must not cross the tool execution boundary."""


_UNSAFE = _UnsafeValue()


def _matches_expected_type(value: Any, expected_type: type[Any]) -> bool:
    if expected_type is int:
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type is float:
        return isinstance(value, int | float) and not isinstance(value, bool)
    return isinstance(value, expected_type)


def _sanitize_value(value: Any) -> Any | _UnsafeValue:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, tuple | list):
        items = []
        for item in value:
            sanitized = _sanitize_value(item)
            if sanitized is _UNSAFE:
                return _UNSAFE
            items.append(sanitized)
        return items
    if isinstance(value, dict):
        sanitized_dict: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                return _UNSAFE
            sanitized_item = _sanitize_value(item)
            if sanitized_item is _UNSAFE:
                return _UNSAFE
            sanitized_dict[key] = sanitized_item
        return sanitized_dict
    return _UNSAFE
