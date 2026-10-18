"""LLM client abstraction.

Talks to any OpenAI-compatible chat completions endpoint. Gemini is
supported through Google's official OpenAI-compatible endpoint
(LLM_PROVIDER=gemini, see Config for the base URL and default model),
so no Gemini-specific SDK or code path is required.
"""

import json
import re
from typing import Optional, Dict, Any, List

from openai import OpenAI

from ..config import Config


class LLMClient:
    """LLM Client - talks to any OpenAI-compatible endpoint (OpenAI, Gemini, etc.)"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None
    ):
        """
        Initialize LLM Client

        Args:
            api_key: API key for the LLM provider
            base_url: Base URL for OpenAI-compatible endpoints
            model: Model name to use

        Raises:
            ValueError: If LLM_API_KEY is not configured
        """
        self.provider = Config.LLM_PROVIDER or "openai"
        self.api_key = api_key or Config.LLM_API_KEY
        self.base_url = base_url or Config.LLM_BASE_URL
        self.model = model or Config.LLM_MODEL_NAME

        if not self.api_key:
            raise ValueError("LLM_API_KEY is not configured")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: Optional[Dict] = None
    ) -> str:
        """
        Send chat request to LLM
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens in response
            response_format: Response format specification (e.g., for JSON mode)
            
        Returns:
            LLM response text
            
        Raises:
            Exception: If LLM API call fails
        """

        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if response_format:
            kwargs["response_format"] = response_format

        response = self.client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content
        # 部分模型（如MiniMax M2.5）会在content中包含<think>思考内容，需要移除
        content = re.sub(r'<think>[\s\S]*?</think>', '', content).strip()
        return content

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4096
    ) -> Dict[str, Any]:
        """
        Send chat request and parse JSON response
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens in response
            
        Returns:
            Parsed JSON object from LLM response
            
        Raises:
            ValueError: If response is not valid JSON or parsing fails
        """
        response = self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"}
        )
        # Clean up markdown code block markers
        cleaned_response = response.strip()
        cleaned_response = re.sub(r'^```(?:json)?\s*\n?', '', cleaned_response, flags=re.IGNORECASE)
        cleaned_response = re.sub(r'\n?```\s*$', '', cleaned_response)
        cleaned_response = cleaned_response.strip()

        try:
            return json.loads(cleaned_response)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"LLM returned invalid JSON (provider: {self.provider}): {cleaned_response[:200]}... "
                f"Error: {str(e)}"
            )

