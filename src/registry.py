from __future__ import annotations

from typing import Any, Callable, Dict, Optional


class Registry:
    """Simple name -> callable registry.

    The registered callable should construct and return an object when called.
    """

    def __init__(self) -> None:
        self._items: Dict[str, Callable[..., Any]] = {}

    def register(self, name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(fn_or_cls: Callable[..., Any]) -> Callable[..., Any]:
            key = name.strip()
            if not key:
                raise ValueError("Registry name must be a non-empty string")
            if key in self._items:
                raise KeyError(f"Registry already contains an item named '{key}'")
            self._items[key] = fn_or_cls
            return fn_or_cls
        return decorator

    def get(self, name: str) -> Callable[..., Any]:
        try:
            return self._items[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._items.keys())) or "<empty>"
            raise KeyError(f"Unknown registry name '{name}'. Available: {available}") from exc

    def build(self, name: str, *args: Any, **kwargs: Any) -> Any:
        builder = self.get(name)
        return builder(*args, **kwargs)


# Concrete registries
MODEL_REGISTRY = Registry()
OPTIMIZER_REGISTRY = Registry()
SCHEDULER_REGISTRY = Registry()
LOSS_REGISTRY = Registry()
DATAMODULE_REGISTRY = Registry()


# Convenience decorators
def register_model(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    return MODEL_REGISTRY.register(name)


def register_optimizer(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    return OPTIMIZER_REGISTRY.register(name)


def register_scheduler(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    return SCHEDULER_REGISTRY.register(name)


def register_loss(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    return LOSS_REGISTRY.register(name)


def register_datamodule(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    return DATAMODULE_REGISTRY.register(name)


