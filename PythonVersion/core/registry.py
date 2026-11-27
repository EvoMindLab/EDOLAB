"""Small named registries for EDOLAB OO instances."""

from __future__ import annotations

from importlib import import_module
from typing import Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, kind: str):
        self.kind = kind
        self._items: dict[str, T] = {}
        self._lazy_items: dict[str, tuple[str, str]] = {}

    def register_lazy(self, name: str, module: str, attribute: str) -> None:
        """Declare an instance without importing unrelated implementations.

        Loading errors propagate when that instance is requested. A concrete
        registration (including a decorator during import) takes precedence.
        """
        if name not in self._items:
            self._lazy_items[name] = (module, attribute)

    def register(self, name: str, item: T | None = None):
        def decorator(value: T) -> T:
            self._items[name] = value
            self._lazy_items.pop(name, None)
            return value

        if item is not None:
            return decorator(item)
        return decorator

    def get(self, name: str) -> T:
        target = self._lazy_items.get(name)
        if name not in self._items and target is not None:
            module, attribute = target
            value = getattr(import_module(module), attribute)
            self.register(name, value)
        try:
            return self._items[name]
        except KeyError as exc:
            available = ", ".join(self.names()) or "<none>"
            raise KeyError(f"No registered {self.kind} named {name!r}. Available: {available}") from exc

    def names(self) -> list[str]:
        return sorted(self._items.keys() | self._lazy_items.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._items or name in self._lazy_items


algorithm_registry: Registry[type] = Registry("algorithm")
problem_registry: Registry[type] = Registry("problem")
metric_registry: Registry[type] = Registry("metric")
