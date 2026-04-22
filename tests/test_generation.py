"""
Tests for generation components like response parsing.
=== FILE: tests/test_generation.py ===
"""

import unittest
from unittest.mock import patch, MagicMock
from generation.response_parser import ResponseParser

class TestResponseParser(unittest.TestCase):
    
    def test_parse_valid_answer(self):
        answer_text = "The quick brown fox jumps over the lazy dog."
        parsed = ResponseParser.parse(answer_text)
        
        self.assertEqual(parsed["answer"], answer_text)
        self.assertTrue(parsed["has_answer"])
        self.assertEqual(parsed["word_count"], 9)
        
    def test_parse_unknown_answer(self):
        answer_text = "I do not know the answer based on the context."
        parsed = ResponseParser.parse(answer_text)
        
        self.assertEqual(parsed["answer"], answer_text)
        self.assertFalse(parsed["has_answer"])

if __name__ == "__main__":
    unittest.main()
