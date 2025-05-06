from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseModel(ABC):
    def __init__(self, api_key: str, model_name: str):
        self.api_key = api_key
        self.model_name = model_name

    @abstractmethod
    def generate_response(self, prompt: str) -> str:
        """Generate response based on input prompt"""
        pass