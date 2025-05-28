"""
Markdown utilities for document processing.
Provides functions for working with Markdown content.
"""

import re
from pathlib import Path

def extract_title(markdown_content):
    """
    Extract the title from Markdown content.
    
    Args:
        markdown_content (str): Markdown content
        
    Returns:
        str: Title of the document
    """
    # Look for the first level 1 heading
    match = re.search(r'^# (.+)$', markdown_content, re.MULTILINE)
    if match:
        return match.group(1).strip()
    
    # If no level 1 heading found, look for the first level 2 heading
    match = re.search(r'^## (.+)$', markdown_content, re.MULTILINE)
    if match:
        return match.group(1).strip()
    
    # If no heading found, return a default title
    return "Untitled Document"

def extract_sections(markdown_content):
    """
    Extract sections from Markdown content.
    
    Args:
        markdown_content (str): Markdown content
        
    Returns:
        list: List of sections with heading level, title, and content
    """
    # Find all headings
    heading_pattern = r'^(#{1,6}) (.+)$'
    headings = list(re.finditer(heading_pattern, markdown_content, re.MULTILINE))
    
    sections = []
    
    # Process each heading and its content
    for i, heading in enumerate(headings):
        heading_level = len(heading.group(1))
        heading_title = heading.group(2).strip()
        
        # Determine the end of this section (next heading or end of content)
        if i < len(headings) - 1:
            section_content = markdown_content[heading.end():headings[i+1].start()]
        else:
            section_content = markdown_content[heading.end():]
        
        sections.append({
            'level': heading_level,
            'title': heading_title,
            'content': section_content.strip()
        })
    
    return sections

def extract_image_references(markdown_content):
    """
    Extract all image references from Markdown content.
    
    Args:
        markdown_content (str): Markdown content
        
    Returns:
        list: List of dictionaries with alt_text and path for each image
    """
    img_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
    matches = list(re.finditer(img_pattern, markdown_content))
    
    images = []
    for match in matches:
        alt_text = match.group(1)
        image_path = match.group(2)
        
        images.append({
            'alt_text': alt_text,
            'path': image_path,
            'start': match.start(),
            'end': match.end(),
            'original': match.group(0)
        })
    
    return images