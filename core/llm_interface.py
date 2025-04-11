# core/llm_interface.py
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()


class LLMClient:
    """Interface for LLM APIs using OpenAI-compatible endpoints"""

    def __init__(self, model="deepseek-chat", api_key=None):
        self.model = model
        
        # Determine which API to use based on the model name
        if model.startswith("gpt-") or model.startswith("text-davinci-") or model.startswith("claude-"):
            # Using OpenAI or compatible API
            self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
            self.base_url = None  # Use default OpenAI URL
            
            if not self.api_key:
                raise ValueError(
                    "OpenAI API key not found. Please set OPENAI_API_KEY environment variable or in .env file."
                )
        else:
            # Using DeepSeek API or other custom endpoint
            self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
            self.base_url = "https://api.deepseek.com"
            
            if not self.api_key:
                raise ValueError(
                    "DeepSeek API key not found. Please set DEEPSEEK_API_KEY environment variable or in .env file."
                )

        # Initialize client with appropriate configuration
        if self.base_url:
            self.client = AsyncOpenAI(
                api_key=self.api_key, 
                base_url=self.base_url
            )
        else:
            self.client = AsyncOpenAI(
                api_key=self.api_key
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

        try:
            # Call the appropriate API
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,  # Non-streaming response
            )
            return response.choices[0].message.content
        except Exception as e:
            # Log the error for debugging
            print(f"API Error: {str(e)}")
            raise
            
    async def generate_with_context(
        self, prompt, conversation_context, temperature=0.7, max_tokens=None, system_message=None
    ):
        """Generate text response from the LLM with conversation context"""
        # Build message array with context
        messages = []
        
        if system_message:
            messages.append({"role": "system", "content": system_message})
            
        # Add all conversation context
        for entry in conversation_context:
            messages.append({"role": entry["role"], "content": entry["content"]})
            
        # Add the current prompt
        messages.append({"role": "user", "content": prompt})

        try:
            # Call the appropriate API
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,  # Non-streaming response
            )
            return response.choices[0].message.content
        except Exception as e:
            # Log the error for debugging
            print(f"API Error: {str(e)}")
            raise
