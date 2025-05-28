"""
Image utilities for document processing.
Provides functions for working with images in documents.
"""

import os
import shutil
import base64
from pathlib import Path
import streamlit as st

def get_image_base64(image_path):
    """
    Convert an image to base64 for embedding in Markdown or HTML.
    
    Args:
        image_path (Path): Path to the image file
        
    Returns:
        str: Base64-encoded image data
    """
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")
    except Exception as e:
        st.error(f"Error encoding image {image_path}: {e}")
        return None

def fix_image_paths(content, old_img_dir, new_img_dir):
    """
    Fix image paths in Markdown content and copy images to new location.
    
    Args:
        content (str): Markdown content with image references
        old_img_dir (Path): Original directory containing images
        new_img_dir (Path): New directory to copy images to
        
    Returns:
        str: Updated Markdown content with fixed image paths
    """
    import re
    
    # Ensure paths are Path objects
    old_img_dir = Path(old_img_dir)
    new_img_dir = Path(new_img_dir)
    
    # Find all image references
    img_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
    img_matches = list(re.finditer(img_pattern, content))
    
    # Replace each image reference
    for i, match in enumerate(reversed(img_matches)):
        alt_text = match.group(1)
        img_path = match.group(2)
        
        # Get the image filename only
        img_filename = os.path.basename(img_path.replace('\\', '/'))
        
        # Create the new relative path
        new_path = f"images/{img_filename}"
        
        # Copy the image to the new location if it exists
        old_full_path = old_img_dir / img_filename
        new_full_path = new_img_dir / img_filename
        
        if old_full_path.exists():
            try:
                shutil.copy2(old_full_path, new_full_path)
            except Exception as e:
                st.warning(f"Could not copy image {img_filename}: {e}")
        
        # Replace in the content
        start, end = match.span()
        content = content[:start] + f"![{alt_text}]({new_path})" + content[end:]
    
    return content