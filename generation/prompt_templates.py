"""
Module for loading and managing prompt templates from configurations.
=== FILE: generation/prompt_templates.py ===
"""

import logging
from configs.loader import load_yaml_config
from langchain_core.prompts import PromptTemplate

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class PromptManager:
    """
    Loads prompt templates from the YAML configuration,
    so they are properly versioned and detached from the code.
    """

    def __init__(self, config_file: str = "prompts.yaml"):
        """
        Initializes the PromptManager by parsing the prompt YAML.

        Args:
            config_file (str): The name of the config file in the configs directory.
        """
        try:
            self.prompts_dict = load_yaml_config(config_file)
            logger.info(f"Successfully loaded prompts from {config_file}.")
        except Exception as e:
            logger.error(f"Failed to load prompt configs: {e}")
            self.prompts_dict = {}

    def get_qa_prompt(self) -> PromptTemplate:
        """
        Retrieves the main Q&A prompt template.

        Returns:
            PromptTemplate: LangChain PromptTemplate object ready to format context and question.
        """
        template_str = self.prompts_dict.get("qa_prompt", "{context}\n{question}")
        return PromptTemplate(
            template=template_str,
            input_variables=["context", "question"]
        )

    def get_refinement_prompt(self) -> PromptTemplate:
        """
        Retrieves the refinement prompt template.

        Returns:
            PromptTemplate: LangChain PromptTemplate object ready to format the answer.
        """
        template_str = self.prompts_dict.get("refinement_prompt", "{answer}")
        return PromptTemplate(
            template=template_str,
            input_variables=["answer"]
        )
