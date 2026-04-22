"""
Module for loading configuration YAML files securely.
"""

import os
import yaml
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def load_yaml_config(file_name: str) -> Dict[str, Any]:
    """
    Loads a YAML configuration file from the configs directory.
    
    Args:
        file_name (str): Name of the config file (e.g., config.yaml)
        
    Returns:
        Dict[str, Any]: Parsed YAML configuration as a dictionary.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, file_name)
    
    if not os.path.exists(file_path):
        logger.error(f"Configuration file not found: {file_path}")
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            logger.info(f"Successfully loaded config from {file_name}")
            return config if config else {}
    except Exception as e:
        logger.error(f"Error parsing YAML config {file_name}: {e}")
        raise
