"""
UI utilities for the Streamlit interface.
Provides functions for rendering UI components.
"""

import re
import os
import time
import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path
from utils.image_utils import get_image_base64
from config import config

def show_config_status():
    """
    Display configuration status in the Streamlit UI.
    
    Returns:
        bool: True if configuration is valid, False otherwise
    """
    if not config.AZURE_DOMAIN or not config.AZURE_API_KEY:
        st.warning("⚠️ Azure OpenAI settings not found in environment variables. Please check your .env file.")
        with st.expander("Show required environment variables"):
            st.code("""
            # Required Azure OpenAI settings
            AZURE_DOMAIN=your-resource-name
            AZURE_API_KEY=your-api-key
            # Optional settings with defaults
            AZURE_DEPLOYMENT_NAME=gpt-4o-mini
            AZURE_API_VERSION=2024-02-15-preview
            DOCLING_DPI_SCALE=2.0
            """)
        return False
    else:
        st.success(f"✅ Azure OpenAI configured: {config.AZURE_DOMAIN} / {config.AZURE_DEPLOYMENT_NAME}")
        return True

def display_quick_extract():
    """
    Display quick extract content in a scrollable container with proper page formatting.
    Enhances the display by adding styling for headers, page breaks, and document structure.
    """
    # Apply styling for the container and document elements
    st.markdown(
        """
        <style>
        .quick-extract-container {
            height: 800px;
            overflow-y: auto;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 30px;
            background-color: white;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            font-family: Arial, sans-serif;
            line-height: 1.6;
        }
        /* Page title styling */
        .page-title {
            font-size: 24px;
            font-weight: bold;
            margin-top: 30px;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 1px solid #ddd;
            color: #333;
        }
        /* First page title should not have a top margin */
        .page-title:first-child {
            margin-top: 0;
        }
        /* Section heading styling */
        .section-heading {
            font-size: 20px;
            font-weight: bold;
            margin-top: 25px;
            margin-bottom: 15px;
            color: #444;
        }
        /* Paragraph styling */
        .paragraph {
            margin-bottom: 15px;
            text-align: justify;
        }
        /* List item styling */
        .list-item {
            margin-bottom: 10px;
            padding-left: 20px;
            position: relative;
        }
        .list-item:before {
            content: "•";
            position: absolute;
            left: 0;
            color: #555;
        }
        /* Page break styling */
        .page-break {
            margin: 30px 0;
            border-top: 1px dashed #999;
            position: relative;
            height: 20px;
        }
        .page-break:after {
            content: "Page Break";
            position: absolute;
            top: -10px;
            right: 20px;
            background: white;
            padding: 0 10px;
            color: #999;
            font-size: 12px;
            font-style: italic;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Get the quick extract content
    content = st.session_state.quick_extract_content
    
    # Process the content to add HTML formatting
    # First, escape any HTML in the content
    import html
    content = html.escape(content)
    
    # Format page headers (## Page X)
    content = re.sub(r'## Page (\d+)', r'<div class="page-break"></div><div class="page-title">Page \1</div>', content)
    
    # Format document title (# Document Title)
    content = re.sub(r'^# (.+)$', r'<div class="page-title">\1</div>', content, flags=re.MULTILINE)
    
    # Format section headings
    content = re.sub(r'(?m)^(#+)\s+(.+)$', r'<div class="section-heading">\2</div>', content)
    
    # Format bullet points/list items
    content = re.sub(r'(?m)^\s[•\-]\s+(.+)$', r'<div class="list-item">\1</div>', content)
    content = re.sub(r'(?m)^\s*(\d+)[\.\)]\s+(.+)$', r'<div class="list-item"><strong>\1.</strong> \2</div>', content)
    
    # Format paragraphs (lines with text that aren't headers or list items)
    lines = content.split('\n')
    formatted_lines = []
    for line in lines:
        # Skip lines that are already formatted
        if '<div class="' in line:
            formatted_lines.append(line)
        # Skip empty lines
        elif not line.strip():
            formatted_lines.append('<br>')
        # Format regular text lines as paragraphs
        else:
            formatted_lines.append(f'<div class="paragraph">{line}</div>')
    
    formatted_content = '\n'.join(formatted_lines)
    
    # Display the content in a scrollable container
    st.markdown(f'<div class="quick-extract-container">{formatted_content}</div>', unsafe_allow_html=True)

def render_markdown_with_images(content, img_dir, max_height=800, width="100%"):
    """
    Replace image references with base64 embedded images for preview.
    Handles both absolute temp paths and relative paths automatically.
    
    Args:
        content (str): Markdown content with image references
        img_dir (Path): Directory containing images
        max_height (int): Maximum height of the preview container
        width (str): Width of the preview container
    """
    if isinstance(img_dir, str):
        img_dir = Path(img_dir)
    
    img_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
    
    def replace_image(match):
        alt_text = match.group(1)
        image_path_str = match.group(2)
        
        # Try different ways to find the image
        possible_paths = []
        
        # 1. If it's already a relative path (images/filename.png)
        if image_path_str.startswith('images/'):
            possible_paths.append(img_dir / image_path_str.replace('images/', ''))
        
        # 2. If it's an absolute path, get just the filename
        filename = os.path.basename(image_path_str.replace('\\', '/'))
        possible_paths.extend([
            Path(image_path_str),  # Original absolute path (temp)
            img_dir / filename,    # Project images folder
        ])
        
        # Try each possible path until we find one that exists
        for image_path in possible_paths:
            if image_path.exists():
                try:
                    img_base64 = get_image_base64(image_path)
                    if img_base64:
                        return f'<img src="data:image/png;base64,{img_base64}" alt="{alt_text}" style="max-width:100%;">'
                except Exception as e:
                    continue
        
        # If no image found, return a placeholder
        return f'<div style="border:1px solid #ccc; padding:10px; text-align:center; background:#f5f5f5;">Image not found: {alt_text}</div>'
    
    # Replace all image references
    html_content = re.sub(img_pattern, replace_image, content)
    
    # Create the scrollable container using Streamlit's native components
    st.markdown(
        f"""
        <style>
        .document-container {{
            height: {max_height}px;
            overflow-y: auto;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 30px;
            background-color: white;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            font-family: Arial, sans-serif;
            line-height: 1.6;
        }}
        </style>
        """, 
        unsafe_allow_html=True
    )
    
    # Create a container for the document
    doc_container = st.container()
    
    # Apply the CSS class and render the content
    with doc_container:
        st.markdown(f'<div class="document-container">{html_content}</div>', unsafe_allow_html=True)

def auto_refresh(interval_seconds=5):
    """
    Create a hidden component that auto-refreshes the page at specified intervals.
    Only active when processing is happening in the background.
    
    Args:
        interval_seconds (int): Refresh interval in seconds
    """
    # Only add auto-refresh when we're waiting for processing to complete
    if st.session_state.processing_status == "quick_extract_done" or st.session_state.processing_status == "docling_in_progress":
        # JavaScript for auto-refresh
        auto_refresh_js = f"""
        <script>
            // This script auto-refreshes the page to check background processing status
            setTimeout(function() {{
                window.location.reload();
            }}, {interval_seconds * 1000});
        </script>
        """
        # Inject the JavaScript
        components.html(auto_refresh_js, height=0, width=0)