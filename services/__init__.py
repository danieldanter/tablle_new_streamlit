"""
Services package for external API integration.
"""

from services.azure_openai import (
    check_azure_config,
    answer_with_azure_openai,
    add_descriptions_with_azure_openai
)
from services.mistral_api import (
    check_mistral_config,
    process_document_with_mistral_file
    # Remove or comment out this line:
    # process_document_with_mistral_url,
)

__all__ = [
    'check_azure_config',
    'answer_with_azure_openai',
    'add_descriptions_with_azure_openai',
    'check_mistral_config',
    'process_document_with_mistral_file'
    # Remove or comment out this line:
    # 'process_document_with_mistral_url'
]