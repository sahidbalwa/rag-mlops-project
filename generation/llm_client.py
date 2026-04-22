"""
Module for interacting with large language models.
=== FILE: generation/llm_client.py ===
"""

import os
import time
import logging
from typing import Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
try:
    from langchain_community.llms import HuggingFaceHub
except ImportError:
    HuggingFaceHub = None

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class LLMClient:
    """
    Connects to the specified LLM Provider to generate answers.
    Supports OpenAI and HuggingFace models.
    """

    def __init__(self, provider: str = None, model_name: str = None, temperature: float = 0.0):
        """
        Initializes the LLM connection.

        Args:
            provider (str): 'openai' or 'huggingface'. Defaults to env var LLM_PROVIDER.
            model_name (str): specific model name if needed to override.
            temperature (float): sampling temperature.
        """
        self.provider = provider or os.getenv("LLM_PROVIDER", "openai").lower()
        self.temperature = temperature
        
        logger.info(f"Initializing LLMClient with provider: {self.provider}")
        
        try:
            if self.provider == "openai":
                self.model_name = model_name or "gpt-4-turbo-preview"
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OPENAI_API_KEY environment variable is missing.")
                self.llm = ChatOpenAI(
                    model_name=self.model_name,
                    temperature=self.temperature,
                    openai_api_key=api_key
                )
            elif self.provider == "huggingface":
                self.model_name = model_name or "mistralai/Mistral-7B-Instruct-v0.2"
                if HuggingFaceHub is None:
                    raise ImportError("HuggingFace hub module is missing.")
                self.llm = HuggingFaceHub(
                    repo_id=self.model_name,
                    model_kwargs={"temperature": self.temperature, "max_length": 1024}
                )
            else:
                raise ValueError(f"Unknown LLM provider: {self.provider}")
        except Exception as e:
            logger.error(f"Error initializing LLM client: {e}")
            raise

    def generate(self, prompt: str) -> str:
        """
        Sends a prompt to the LLM and retrieves the response.

        Args:
            prompt (str): The complete prompt string containing context and the question.

        Returns:
            str: The LLM's response block.
        """
        start_time = time.time()
        logger.info("Generating response from LLM...")
        
        try:
            # For ChatOpenAI, invoke returns an AIMessage. For HuggingFaceHub it returns a string.
            response = self.llm.invoke(prompt)
            if hasattr(response, 'content'):
                answer = response.content
            else:
                answer = str(response)
                
            elapsed = time.time() - start_time
            logger.info(f"Generated response in {elapsed:.3f} seconds.")
            return answer
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            raise
