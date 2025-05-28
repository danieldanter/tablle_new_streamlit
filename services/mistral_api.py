"""
Mistral API service for OCR and document processing.
Provides functions for interacting with Mistral OCR API.
"""

import os
import base64
import re
import requests
import json
from pathlib import Path
import streamlit as st
from config import config

def check_mistral_config():
    """
    Check if Mistral API configuration is valid.
    
    Returns:
        bool: True if configuration is valid, False otherwise
    """
    if not config.MISTRAL_API_KEY:
        return False
    return True

def process_document_with_mistral_file(file_path, include_image_base64=True, image_limit=10):
    """
    Process a document with Mistral OCR using a local file.
    
    Args:
        file_path (Path): Path to the document file
        include_image_base64 (bool): Whether to include base64-encoded images in the response
        image_limit (int): Maximum number of images to include
        
    Returns:
        dict: Results of the document processing
    """
    try:
        # Ensure file_path is a Path object and exists
        file_path = Path(file_path)
        if not file_path.exists():
            return {"error": f"File not found: {file_path}"}
        
        # Read and encode the file
        with open(file_path, "rb") as f:
            file_content = f.read()
            
        # Encode the file as base64
        base64_encoded = base64.b64encode(file_content).decode('utf-8')
        
        # Determine content type based on file extension
        file_extension = file_path.suffix.lower()
        if file_extension in ['.pdf']:
            content_type = "application/pdf"
        elif file_extension in ['.jpg', '.jpeg']:
            content_type = "image/jpeg"
        elif file_extension in ['.png']:
            content_type = "image/png"
        else:
            content_type = "application/octet-stream"
            
        # Construct the data URL
        data_url = f"data:{content_type};base64,{base64_encoded}"
        
        # Prepare the API request
        headers = {
            "Authorization": f"Bearer {config.MISTRAL_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Debug info
        st.info(f"Processing document: {file_path.name}")
        
        payload = {
            "model": "mistral-ocr-latest",
            "document": {
                "type": "document_url",
                "document_url": data_url
            },
            "include_image_base64": include_image_base64,
            "image_limit": image_limit
        }
        
        # Make the API call
        st.info("Sending request to Mistral OCR API...")
        response = requests.post(
            "https://api.mistral.ai/v1/ocr",
            headers=headers,
            json=payload,
            timeout=180  # Extended timeout for large documents
        )
        
        # Check for successful response
        response.raise_for_status()
        
        # Get the response data
        response_data = response.json()
        
        # Debug info
        st.success("Received response from Mistral OCR API")
        st.info(f"Response contains {len(response_data.get('pages', []))} pages")
        
        # Check if there are any images in the response
        has_images = False
        for page in response_data.get('pages', []):
            if page.get('images'):
                has_images = True
                st.info(f"Page {page['index']} has {len(page['images'])} images")
                
        if not has_images:
            st.warning("No images found in the OCR response")
        
        # Return the response as a dictionary
        return response_data
    
    except requests.exceptions.RequestException as e:
        st.error(f"Error calling Mistral OCR API: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_details = e.response.json()
                st.error(f"API Error Details: {json.dumps(error_details, indent=2)}")
            except:
                st.error(f"Status Code: {e.response.status_code}, Response: {e.response.text}")
        return {"error": str(e)}
    except Exception as e:
        st.error(f"Error processing document with Mistral OCR: {str(e)}")
        return {"error": str(e)}

def extract_and_save_images(ocr_results, output_dir):
    """
    Extract and save images from Mistral OCR results.
    
    Args:
        ocr_results (dict): Mistral OCR API response
        output_dir (Path): Directory to save images
        
    Returns:
        dict: Mapping of image IDs to file paths
    """
    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Image mapping dictionary
    image_mapping = {}
    
    # Debug information
    st.info(f"Extracting images to: {output_dir}")
    
    # Process each page
    for page in ocr_results.get("pages", []):
        page_idx = page.get("index", 0)
        
        # Process images in the page
        for img_idx, image in enumerate(page.get("images", [])):
            image_id = image.get("id")
            base64_data = image.get("image_base64")
            
            if not image_id or not base64_data:
                st.warning(f"Missing image data for image on page {page_idx}")
                continue
            
            try:
                # Parse the base64 data to determine image format
                img_match = re.match(r"data:image/(png|jpeg|jpg);base64,(.*)", base64_data)
                if not img_match:
                    # Try without the data URL prefix
                    ext = "png"  # Default to PNG
                    img_b64 = base64_data
                else:
                    img_format, img_b64 = img_match.groups()
                    ext = "jpg" if img_format in ["jpeg", "jpg"] else "png"
                
                # Create a unique filename
                filename = f"page_{page_idx}_image_{img_idx}.{ext}"
                filepath = output_dir / filename
                
                # Decode and save the image
                with open(filepath, "wb") as f:
                    f.write(base64.b64decode(img_b64))
                
                # Add to mapping
                image_mapping[image_id] = str(filepath)
                st.success(f"Saved image to: {filepath}")
            
            except Exception as e:
                st.warning(f"Error saving image {image_id}: {str(e)}")
    
    st.info(f"Extracted {len(image_mapping)} images")
    return image_mapping

def process_markdown_content(markdown_content, image_mapping):
    """
    Process Markdown content to update image references.
    
    Args:
        markdown_content (str): Markdown content from Mistral OCR
        image_mapping (dict): Mapping of image IDs to file paths
        
    Returns:
        str: Processed Markdown content with updated image references
    """
    # Ensure we have a string
    if not isinstance(markdown_content, str):
        st.warning(f"Markdown content is not a string: {type(markdown_content)}")
        markdown_content = str(markdown_content)
    
    # Check if there are image references already in the content
    has_image_refs = '![' in markdown_content and '](' in markdown_content
    
    # If there are NO image references in the markdown content, add them
    # This should only happen if the Mistral API doesn't include proper image references
    if not has_image_refs and image_mapping:
        st.info("No image references found in markdown, adding them at appropriate positions")
        # Instead of adding all images at the beginning, we'll add them contextually
        # For now, let's not add them automatically - let the OCR content speak for itself
        pass
    
    # Replace existing image references with local file paths
    for image_id, file_path in image_mapping.items():
        # Convert to Path object and use as_posix() for consistent forward slashes
        normalized_path = Path(file_path).as_posix()
        
        # Simple string replacement approach to avoid regex issues
        # Replace image references that use the image ID
        old_ref = f"]({image_id})"
        new_ref = f"]({normalized_path})"
        markdown_content = markdown_content.replace(old_ref, new_ref)
        
        # Also handle cases where the image ID might be used differently
        # Sometimes OCR APIs use different formats
        old_ref_alt = f"({image_id})"
        new_ref_alt = f"({normalized_path})"
        markdown_content = markdown_content.replace(old_ref_alt, new_ref_alt)
    
    return markdown_content
def combine_page_markdown(ocr_results, image_dir):
    """
    Combine Markdown content from all pages and process image references.
    
    Args:
        ocr_results (dict): Mistral OCR API response
        image_dir (Path): Directory where images are saved
        
    Returns:
        str: Combined and processed Markdown content
    """
    # Extract and save images first
    image_mapping = extract_and_save_images(ocr_results, image_dir)
    
    # Process each page
    all_markdown = []
    for page in ocr_results.get("pages", []):
        page_markdown = page.get("markdown", "")
        
        # Process image references in the page content
        processed_markdown = process_markdown_content(page_markdown, image_mapping)
        
        # Add to the list
        all_markdown.append(processed_markdown)
    
    # Combine all pages with page separators
    combined_markdown = "\n\n---\n\n".join(all_markdown)
    
    # Check if there are any image references in the combined content
    has_images_in_content = '![' in combined_markdown and '](' in combined_markdown
    
    # Only add images at the beginning if there are NO images in the content
    # This prevents duplication
    if not has_images_in_content and image_mapping:
        st.info("No images found in page content, adding them at the beginning")
        # Add image references at the beginning as fallback
        image_refs = ""
        for image_id, file_path in image_mapping.items():
            normalized_path = Path(file_path).as_posix()
            image_refs += f"\n\n![Image {image_id}]({normalized_path})"
        combined_markdown = image_refs + "\n\n" + combined_markdown
    elif has_images_in_content:
        st.info(f"Images are already embedded in the page content - not adding duplicates")
    
    # Debug info
    st.info(f"Combined markdown has {len(combined_markdown)} characters")
    
    return combined_markdown