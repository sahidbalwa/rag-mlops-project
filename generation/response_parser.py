"""
Module for parsing, validating, and structuring the raw LLM output.
=== FILE: generation/response_parser.py ===
"""

import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class ResponseParser:
    """
    Validates and structures the text output from the LLM.
    """

    @staticmethod
    def parse(raw_response: str) -> Dict[str, Any]:
        """
        Parses the raw text returned by the LLM and infers structured attributes.

        Args:
            raw_response (str): The unformatted text output from the LLM.

        Returns:
            Dict[str, Any]: Structured dictionary with keys: answer, has_answer, word_count.
        """
        logger.info("Parsing LLM response.")
        raw_response = raw_response.strip()
        
        # Heuristic check: Did the LLM explicitly state it doesn't know?
        unknown_phrases = [
            "i do not know",
            "not contained in the context",
            "i don't have enough information",
            "cannot find the answer"
        ]
        
        lower_response = raw_response.lower()
        has_answer = not any(phrase in lower_response for phrase in unknown_phrases)

        # Basic word count logic
        word_count = len(raw_response.split())

        parsed = {
            "answer": raw_response,
            "has_answer": has_answer,
            "word_count": word_count
        }

        logger.info(f"Response validation complete. Has Answer: {has_answer}, Word Count: {word_count}")
        return parsed
