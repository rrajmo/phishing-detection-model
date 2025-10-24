from abc import ABC, abstractmethod
from typing import Any

class BaseModel(ABC):
    @abstractmethod
    def fit(self, X: Any, y: Any, categorical_feature: Any = None) -> Any:
        pass

    @abstractmethod
    def predict(self, X: Any) -> Any:
        pass

    @abstractmethod
    def predict_probabilities(self, X: Any) -> Any:
        pass

    @abstractmethod
    def save(self, file: str) -> Any:
        pass

    @abstractmethod
    def load(self, file: str) -> Any:
        pass
