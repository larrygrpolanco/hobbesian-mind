# core/llm_interface.py
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()


class LLMClient:
    """Interface for DeepSeek LLM API using OpenAI-compatible endpoint"""

    def __init__(self, model="deepseek-chat", api_key=None):
        # Model to use - default is DeepSeek's chat model
        self.model = model
        # API key from param, environment, or .env file (loaded by load_dotenv)
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")

        if not self.api_key:
            raise ValueError(
                "DeepSeek API key not found. Please set DEEPSEEK_API_KEY environment variable or in .env file."
            )

        # Initialize with DeepSeek's base URL
        self.client = AsyncOpenAI(
            api_key=self.api_key, base_url="https://api.deepseek.com"
        )

    async def generate(
        self, prompt, temperature=0.7, max_tokens=None, system_message=None
    ):
        """Generate text response from the LLM"""
        # Build message array
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        # Call DeepSeek API
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,  # Non-streaming response
        )

        return response.choices[0].message.content
