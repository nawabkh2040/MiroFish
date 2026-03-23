"""LLM client abstraction.

Supports OpenAI-compatible models by default and can optionally
use Gemini when LLM_PROVIDER=gemini is configured in the
environment (see Config).
"""

import json
import re
from typing import Optional, Dict, Any, List

from openai import OpenAI

from ..config import Config


class LLMClient:
    """LLM Client - Supports both OpenAI and Gemini providers"""
    
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

        # Initialize provider-specific client
        if self.provider == "gemini":
            # Lazy import so that environments not using Gemini
            # don't require the extra dependency at runtime.
            try:
                import google.generativeai as genai  # type: ignore
                genai.configure(api_key=self.api_key)
                self._gemini = genai
                self.client = None
            except ImportError:
                raise ImportError(
                    "google-generativeai package required for Gemini support. "
                    "Install with: pip install google-generativeai"
                )
        else:
            # Default: OpenAI-compatible endpoint
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            self._gemini = None
    
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

        if self.provider == "gemini":
            return self._chat_gemini(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
            )

        # Default OpenAI-compatible path
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

    def _chat_gemini(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        response_format: Optional[Dict],
    ) -> str:
        """Send chat messages via Gemini.

        This adapts the existing OpenAI-style message list to
        the google-generativeai Python client.
        """

        if not self._gemini:
            raise RuntimeError("Gemini client not initialised")

        # Separate system instruction and conversational turns
        system_parts = [m["content"] for m in messages if m["role"] == "system"]
        system_instruction = "\n\n".join(system_parts) if system_parts else None

        contents: List[Dict[str, Any]] = []
        for m in messages:
            if m["role"] == "system":
                continue
            role = "user" if m["role"] in ("user", "assistant") else "user"
            contents.append({"role": role, "parts": [m["content"]]})

        generation_config: Dict[str, Any] = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }

        # If the caller requested JSON, set the mime type accordingly
        if response_format and response_format.get("type") == "json_object":
            generation_config["response_mime_type"] = "application/json"

        genai = self._gemini
        if system_instruction:
            model = genai.GenerativeModel(self.model, system_instruction=system_instruction)
        else:
            model = genai.GenerativeModel(self.model)

        response = model.generate_content(
            contents,
            generation_config=generation_config,
        )

        content = response.text or ""
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

