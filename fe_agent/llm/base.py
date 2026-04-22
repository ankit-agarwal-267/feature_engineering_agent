import json
from typing import List, Dict, Any, Optional, Protocol
from fe_agent.config.config_schema import LLMConfig

class LLMResponse:
    def __init__(self, content: str, raw_response: Dict[str, Any]):
        self.content = content
        self.raw_response = raw_response

class AbstractLLMProvider(Protocol):
    def chat(self, system_prompt: str, user_message: str) -> LLMResponse:
        ...

class OllamaProvider(AbstractLLMProvider):
    def __init__(self, config: LLMConfig):
        self.config = config
        self.base_url = f"{config.base_url.rstrip('/')}/api/chat"
        try:
            import httpx
            self._httpx = httpx
        except ImportError:
            self._httpx = None

    def chat(self, system_prompt: str, user_message: str) -> LLMResponse:
        if self._httpx is None:
            raise ImportError("The 'httpx' library is required for Ollama. Install it with 'pip install httpx'.")
            
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            "stream": False,
            "options": {
                "temperature": self.config.temperature
            }
        }
        
        try:
            with self._httpx.Client(timeout=self.config.timeout_seconds) as client:
                response = client.post(self.base_url, json=payload)
                response.raise_for_status()
                data = response.json()
                content = data.get("message", {}).get("content", "")
                return LLMResponse(content=content, raw_response=data)
        except Exception as e:
            raise RuntimeError(f"Ollama API call failed: {str(e)}")
