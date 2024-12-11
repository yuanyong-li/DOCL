import requests
from typing import Optional, List

from langchain.llms.base import LLM
from pydantic import Field

class OllamaLLM(LLM):
    model_name: str = Field(default="llama-3")  # 使用 Pydantic 的 Field 定义默认模型名称
    api_base_url: str = Field(default="http://localhost:11434")  # 默认 API 基础 URL

    @property
    def _llm_type(self) -> str:
        return "ollama"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        # 向 Ollama API 发送请求并获取响应
        response = requests.post(
            f"{self.api_base_url}/api/generate",
            json={
                "model": self.model_name,
                "prompt": prompt,
                "stop": stop
            }
        )
        response_data = response.json()
        return response_data.get("text", "")


if __name__ == "__main__":
    ollama_llm = OllamaLLM(model_name="llama-3")
    print(response)