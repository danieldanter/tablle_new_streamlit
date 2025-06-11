"""
Configuration module for Document Enhancer application.
Handles loading environment variables and setting up configuration values.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from dataclasses import dataclass

@dataclass
class AppConfig:
    """Application configuration container"""
    # Azure OpenAI settings
    AZURE_DOMAIN: str
    AZURE_API_KEY: str
    AZURE_DEPLOYMENT_NAME: str
    AZURE_API_VERSION: str
    
    # Docling settings
    DOCLING_DPI_SCALE: float
    
    # Mistral API settings
    MISTRAL_API_KEY: str
    MISTRAL_API_URL: str
    
    # AWS settings for Textract
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    AWS_DEFAULT_REGION: str
    AWS_TEXTRACT_BUCKET: str
    
    # Paths
    PROJECT_DIR: Path
    DOCS_DIR: Path

# Create project directories
PROJECT_DIR = Path.cwd() / "document_enhancer_data"
PROJECT_DIR.mkdir(exist_ok=True)
DOCS_DIR = PROJECT_DIR / "documents"
DOCS_DIR.mkdir(exist_ok=True)

def initialize_config():
    """
    Initialize and return application configuration.
    Loads environment variables and sets up configuration values.
    
    Returns:
        AppConfig: Application configuration object
    """
    # Load environment variables if not already loaded
    load_dotenv()
    
    # Get Azure OpenAI settings from environment variables
    azure_domain = os.getenv("AZURE_DOMAIN", "")
    azure_api_key = os.getenv("AZURE_API_KEY", "")
    azure_deployment_name = os.getenv("AZURE_DEPLOYMENT_NAME", "gpt-4o-mini")
    azure_api_version = os.getenv("AZURE_API_VERSION", "2024-02-15-preview")
    
    # Get Docling settings from environment variables
    docling_dpi_scale = float(os.getenv("DOCLING_DPI_SCALE", "2.0"))
    
    # Get Mistral API settings
    mistral_api_key = os.getenv("MISTRAL_API_KEY", "")
    mistral_api_url = os.getenv("MISTRAL_API_URL", "https://api.aimlapi.com/v1/ocr")
    
    # Get AWS settings for Textract
    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID", "")
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    aws_default_region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    aws_textract_bucket = os.getenv("AWS_TEXTRACT_BUCKET", "")  # Optional, will be auto-generated if not provided
    
    # Create and return the config object
    return AppConfig(
        AZURE_DOMAIN=azure_domain,
        AZURE_API_KEY=azure_api_key,
        AZURE_DEPLOYMENT_NAME=azure_deployment_name,
        AZURE_API_VERSION=azure_api_version,
        DOCLING_DPI_SCALE=docling_dpi_scale,
        MISTRAL_API_KEY=mistral_api_key,
        MISTRAL_API_URL=mistral_api_url,
        AWS_ACCESS_KEY_ID=aws_access_key_id,
        AWS_SECRET_ACCESS_KEY=aws_secret_access_key,
        AWS_DEFAULT_REGION=aws_default_region,
        AWS_TEXTRACT_BUCKET=aws_textract_bucket,
        PROJECT_DIR=PROJECT_DIR,
        DOCS_DIR=DOCS_DIR
    )

# Create a default config instance
config = initialize_config()

# For direct imports like "from config import DOCS_DIR"
AZURE_DOMAIN = config.AZURE_DOMAIN
AZURE_API_KEY = config.AZURE_API_KEY
AZURE_DEPLOYMENT_NAME = config.AZURE_DEPLOYMENT_NAME
AZURE_API_VERSION = config.AZURE_API_VERSION
DOCLING_DPI_SCALE = config.DOCLING_DPI_SCALE
MISTRAL_API_KEY = config.MISTRAL_API_KEY
MISTRAL_API_URL = config.MISTRAL_API_URL
AWS_ACCESS_KEY_ID = config.AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY = config.AWS_SECRET_ACCESS_KEY
AWS_DEFAULT_REGION = config.AWS_DEFAULT_REGION
AWS_TEXTRACT_BUCKET = config.AWS_TEXTRACT_BUCKET