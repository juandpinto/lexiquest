from abc import ABC, abstractmethod
from typing import Any, Mapping


class BaseAgent(ABC):
    def __init__(self, name: str):
        self.name = name
        self.state: dict[str, Any] = {}

    @abstractmethod
    def __call__(self, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        pass

    def update_state(self, key: str, value: Any):
        self.state[key] = value

    def get_state(self) -> Mapping[str, Any]:
        return self.state
