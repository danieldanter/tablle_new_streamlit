"""
File utilities for document processing.
Provides functions for working with files and directories.
"""

import os
import tempfile
import shutil
from pathlib import Path
import uuid

def create_unique_directory(base_dir, prefix="doc_"):
    """
    Create a unique directory with the given prefix.
    
    Args:
        base_dir (Path): Base directory to create the unique directory in
        prefix (str): Prefix for the directory name
        
    Returns:
        Path: Path to the created directory
    """
    # Create a unique ID
    unique_id = str(uuid.uuid4())[:8]
    dir_name = f"{prefix}{unique_id}"
    
    # Create the directory
    dir_path = Path(base_dir) / dir_name
    dir_path.mkdir(parents=True, exist_ok=True)
    
    return dir_path

def save_uploaded_file(uploaded_file, directory):
    """
    Save an uploaded file to the given directory.
    
    Args:
        uploaded_file: Streamlit uploaded file
        directory (Path): Directory to save the file to
        
    Returns:
        Path: Path to the saved file
    """
    # Ensure directory exists
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    
    # Save the file
    file_path = directory / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return file_path

def get_file_extension(file_path):
    """
    Get the file extension of a file path.
    
    Args:
        file_path (Path or str): Path to the file
        
    Returns:
        str: File extension (lowercase, with dot)
    """
    return Path(file_path).suffix.lower()

def create_temp_directory():
    """
    Create a temporary directory.
    
    Returns:
        Path: Path to the temporary directory
    """
    return Path(tempfile.mkdtemp())