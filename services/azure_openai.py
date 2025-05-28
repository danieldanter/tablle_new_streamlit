"""
Azure OpenAI service for document processing.
Provides functions for interacting with Azure OpenAI API.
"""
import base64
import requests
import time
import re
from pathlib import Path
from PIL import Image
import math
import streamlit as st
from config import config

# Constants for image description prompts
# Constants for image description prompts
ENGLISH_PROMPT_TEMPLATE = """This image appears in a document. Here's the context from the document: 
{context}
IMPORTANT: Pay special attention to the title of the chart or graph to understand EXACTLY what is being measured.
IMPORTANT: First determine if this image contains MULTIPLE separate charts/graphs/visualizations.

Please analyze this image as follows:
1. FIRST, categorize this image as ONE of the following:
- DATA VISUALIZATION (chart, graph, diagram, infographic)
- PHOTOGRAPH (historical, documentary, portrait)
- ILLUSTRATION (drawing, sketch, artwork)
- DOCUMENT (form, certificate, letterhead)
- LOGO/ICON (brand symbol, decorative element)

2. SECOND, provide a brief descriptive summary (1-2 sentences) of what the image shows.
- NEVER use phrases like "specific value" or "certain value"
- ALWAYS name specifically what is being measured (e.g., employee count, revenue, production volume)
- Use the title or labels in the image to identify what the data represents

3. THIRD, ONLY if this is a DATA VISUALIZATION:
   If the image contains MULTIPLE independed visualizations:  
   - Clearly state "This image contains [number] separate visualizations."
   - For EACH visualization, label it as "VISUALIZATION #1:", "VISUALIZATION #2:", etc.
   - For EACH visualization separately:
     a) Extract ALL numeric data points, labels, and categories
     b) Include a separate markdown table for EACH dataset
     c) Identify trends, comparisons, or patterns for EACH visualization
   
   If the image contains only ONE visualization:
   - Extract ALL numeric data points, labels, and categories
   - Include a markdown table with the complete dataset
   - Identify trends, comparisons, or patterns

4. End the description with a "]" marker
For all other image types, just provide the 1-2 sentence description from step 2, with no additional analysis and end the description with a "]" marker.
DO NOT include headings like "Summary:" or "Data Table:" in your response.
If this appears to be a logo, icon, or purely decorative element, just respond with "This is a logo or decorative element" and nothing else.
"""

GERMAN_PROMPT_TEMPLATE = """Dieses Bild erscheint in einem Dokument. Hier ist der Kontext aus dem Dokument:
{context}
WICHTIG: Achte besonders auf den Titel des Diagramms oder der Grafik, um zu verstehen, WAS genau gemessen wird.
WICHTIG: Stelle zuerst fest, ob dieses Bild MEHRERE separate Diagramme/Grafiken/Visualisierungen enthält.

Bitte analysiere dieses Bild wie folgt:
1. ERSTENS, kategorisiere dieses Bild als EINES der folgenden:
- DATENVISUALISIERUNG (Diagramm, Grafik, Schema, Infografik)
- FOTOGRAFIE (historisch, dokumentarisch, Portrait)
- ILLUSTRATION (Zeichnung, Skizze, Kunstwerk)
- DOKUMENT (Formular, Zertifikat, Briefkopf)
- LOGO/ICON (Markensymbol, dekoratives Element)

2. ZWEITENS, gib eine kurze beschreibende Zusammenfassung (1-2 Sätze), was das Bild zeigt.
- Verwende dabei NIEMALS die Phrase "bestimmten Wertes" oder "bestimmter Wert"
- Benenne stattdessen IMMER konkret, was gemessen wird (z.B. Mitarbeiteranzahl, Umsatz, Produktionsmenge)
- Verwende den Titel oder Beschriftungen im Bild, um zu identifizieren, worum es sich handelt

3. DRITTENS, NUR wenn es sich um eine DATENVISUALISIERUNG handelt:
   Wenn das Bild MEHRERE unabhängige Visualisierungen enthält:
   - Gib klar an: "Dieses Bild enthält [Anzahl] separate Visualisierungen."
   - Bezeichne für JEDE Visualisierung als "VISUALISIERUNG #1:", "VISUALISIERUNG #2:", usw.
   - Für JEDE Visualisierung separat:
     a) Extrahiere ALLE numerischen Datenpunkte, Beschriftungen und Kategorien
     b) Füge eine separate Markdown-Tabelle für JEDEN Datensatz ein
     c) Identifiziere Trends, Vergleiche oder Muster für JEDE Visualisierung
   
   Wenn das Bild nur EINE Visualisierung enthält:
   - Extrahiere ALLE numerischen Datenpunkte, Beschriftungen und Kategorien
   - Füge eine Markdown-Tabelle mit dem vollständigen Datensatz ein
   - Identifiziere Trends, Vergleiche oder Muster

4. Ende die beschreibung mit einer "]" makierung
Für alle anderen Bildtypen gib nur die 1-2 Sätze Beschreibung aus Schritt 2, ohne zusätzliche Analyse und Ende die beschreibung mit einer "]" makierung.
Verwende KEINE Überschriften wie "Zusammenfassung:" oder "Datentabelle:" in deiner Antwort.
Wenn es sich um ein Logo, Icon oder rein dekoratives Element handelt, antworte einfach mit "Dies ist ein Logo oder dekoratives Element" und nichts weiter.
"""
# System prompts for different image types
SYSTEM_PROMPTS = {
    "data_viz": "You are an expert at analyzing data visualizations, charts, and figures. Extract and organize all data from visualizations in detail.",
    "photo": "You are an expert at analyzing photographs and visual content. Describe what is shown in concise, clear language.",
    "general": "You are an expert at analyzing and describing images in context. Provide clear, concise descriptions of what you see."
}

def check_azure_config():
    """
    Check if Azure OpenAI configuration is valid.
    
    Returns:
        bool: True if configuration is valid, False otherwise
    """
    if not config.AZURE_DOMAIN or not config.AZURE_API_KEY:
        return False
    return True

def answer_with_azure_openai(question, content, azure_domain, api_key, deployment_name, api_version):
    """
    Use Azure OpenAI to answer a question based on document content.
    
    Args:
        question (str): The question to answer
        content (str): The document content to use as context
        azure_domain (str): Azure OpenAI domain
        api_key (str): Azure OpenAI API key
        deployment_name (str): Azure OpenAI deployment name
        api_version (str): Azure OpenAI API version
        
    Returns:
        str: The answer to the question
    """
    try:
        # Azure OpenAI endpoint URL
        url = f"https://{azure_domain}.openai.azure.com/openai/deployments/{deployment_name}/chat/completions?api-version={api_version}"
        
        # Headers and payload
        headers = {
            "Content-Type": "application/json",
            "api-key": api_key
        }
        
        # Create payload
        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers questions about documents. Provide accurate, concise answers based on the document content."
                },
                {
                    "role": "user",
                    "content": f"Here is a document:\n\n{content}\n\nAnswer this question about the document: {question}"
                }
            ],
            "max_tokens": 800,
            "temperature": 0.1
        }
        
        # Call the Azure OpenAI API
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        
        if response.status_code == 200:
            response_data = response.json()
            return response_data["choices"][0]["message"]["content"]
        else:
            error_msg = f"Error: {response.status_code}"
            try:
                error_data = response.json()
                error_msg += f" - {error_data.get('error', {}).get('message', '')}"
            except:
                error_msg += f" - {response.text}"
            return error_msg
    
    except Exception as e:
        return f"Error: {str(e)}"

def detect_language_fast(text):
    """
    A fast language detection function that checks for common German words.
    Only distinguishes between German and English.
    
    Args:
        text (str): Text to analyze
        
    Returns:
        str: 'de' for German, 'en' for English (default)
    """
    # Convert to lowercase for comparison
    text_lower = text.lower()
    
    # Common German-specific words that rarely appear in English
    german_specific = ['der', 'die', 'das', 'und', 'ist', 'für', 'mit', 'auf', 'dass', 
                       'nicht', 'sind', 'ein', 'eine', 'auch', 'es', 'sich', 'dem', 
                       'zu', 'wurde', 'haben', 'hat', 'ich', 'wird', 'als', 'wir', 'bei']
    
    # Create a regex pattern to match whole words only
    pattern = r'\b(' + '|'.join(german_specific) + r')\b'
    
    # Count matches
    german_word_count = len(re.findall(pattern, text_lower))
    
    # If we find several German words, assume it's German
    # Threshold can be adjusted - higher means more confidence needed
    if german_word_count > 5:
        return 'de'
    else:
        return 'en'

def add_descriptions_with_azure_openai(md_file, img_dir, azure_domain, api_key, deployment_name, api_version, batch_size=1):
    """
    Use Azure OpenAI's vision-enabled models to generate detailed descriptions for images in a Markdown file.
    Supports multiple languages and adapts to document context.
    
    Args:
        md_file (Path): Path to the Markdown file
        img_dir (Path): Path to the directory containing images
        azure_domain (str): Azure OpenAI domain
        api_key (str): Azure OpenAI API key
        deployment_name (str): Azure OpenAI deployment name
        api_version (str): Azure OpenAI API version
        batch_size (int): Number of images to process in a batch (max 10)
        
    Returns:
        Path: Path to the updated Markdown file
    """
    # Start image processing timer
    image_start = time.time()   
    
    # Ensure paths are Path objects
    md_file = Path(md_file)
    img_dir = Path(img_dir)
    
    # Enforce max batch size limit (Azure OpenAI limit is 10 images per request)
    batch_size = min(batch_size, 10)
    
    # Load the Markdown content
    with open(md_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find all image references using regex
    image_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
    image_matches = list(re.finditer(image_pattern, content))
    
    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    # Track successful and failed images
    successful_images = 0
    skipped_images = 0
    failed_images = []
    
    # Maximum number of retries for failed API calls
    max_retries = 3
    
    # First pass - identify and remove any existing image descriptions
    # This ensures we don't end up with duplicate descriptions
    description_pattern = r'(\!\[[^\]]*\]\([^)]*\))\s*\n\n\*Image description:.+?(?:\*|\n\n)'
    content = re.sub(description_pattern, r'\1\n\n', content, flags=re.DOTALL)
    
    # Detect document language - use a sample of text for efficiency
    text_sample = content[:2000]  # Use first 2000 chars for detection
    document_language = detect_language_fast(text_sample)
    st.info(f"Document language detected: {'German' if document_language == 'de' else 'English'}")
    
    # Try to import langdetect for more accurate detection if available
    try:
        from langdetect import detect
        # Only use if we're confident it's installed
        document_language = detect(text_sample)
        st.info(f"Document language detected (detailed): {document_language}")
    except ImportError:
        # Keep using our fast detection
        pass
    
    # Single image processing
    for i, match in enumerate(image_matches):
        progress_text.text(f"Processing image {i+1}/{len(image_matches)}")
        progress_bar.progress((i) / len(image_matches))
        
        alt_text = match.group(1)
        image_path_rel = match.group(2)
        
        # Get the image filename and path - handle both absolute and relative paths
        image_filename = Path(image_path_rel).name
        
        # Try different possible paths
        possible_paths = [
            Path(image_path_rel),  # Original path (might be temp)
            img_dir / image_filename,  # Project images folder
        ]
        
        # Find the actual image path
        image_path = None
        for path in possible_paths:
            if path.exists():
                image_path = path
                break
        
        if not image_path:
            st.warning(f"Image not found: {image_filename}")
            failed_images.append({"filename": image_filename, "reason": "File not found"})
            continue
        
        # Check if this image might be a logo or icon
        should_skip = False
        try:
            # Check if filename or alt text suggests it's a logo or icon
            if ("logo" in image_filename.lower() or 
                "icon" in image_filename.lower() or 
                "logo" in alt_text.lower() or 
                "icon" in alt_text.lower()):
                should_skip = True
            
            # Check image size - small images are likely logos or icons
            img = Image.open(image_path)
            width, height = img.size
            img_size = image_path.stat().st_size
            
            # Skip very small images (likely icons or decorative elements)
            if width < 100 or height < 100:
                should_skip = True
            
            # Skip images with very small file size (likely simple graphics)
            if img_size < 10 * 1024:  # Less than 10KB
                should_skip = True
            
            if should_skip:
                st.info(f"Skipping likely logo/icon: {image_filename}")
                skipped_images += 1
                continue
            
            # If image is too large (> 4MB), resize it
            temp_path = None
            if img_size > 4 * 1024 * 1024:
                st.info(f"Resizing large image: {image_filename} ({img_size/1024/1024:.2f} MB)")
                
                # Calculate new dimensions to reduce file size while maintaining aspect ratio
                scale_factor = min(1, math.sqrt(4 * 1024 * 1024 / img_size))
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                
                # Resize the image
                img = img.resize((new_width, new_height), Image.LANCZOS)
                
                # Save the resized image to a temporary file
                import tempfile
                temp_fd, temp_path = tempfile.mkstemp(suffix=f"_temp_{image_filename}")
                temp_path = Path(temp_path)
                img.save(temp_path, optimize=True)
                image_path = temp_path
        except Exception as e:
            st.warning(f"Could not process image size: {e}")
        
        # Skip this image if it was determined to be a logo/icon
        if should_skip:
            continue
        
        retries = 0
        success = False
        
        while retries < max_retries and not success:
            try:
                # Extract context from before AND after the image for better understanding
                # Get context from before the image (up to 500 chars)
                start_pos = max(0, match.start() - 500)
                context_before = content[start_pos:match.start()].strip()

                #print()
                #print(match)

                #print(f"Context before image: {context_before} ")  # Debug output
                
                # Get context from after the image (up to 500 chars)
                end_pos = min(len(content), match.end() + 500)
                context_after = content[match.end():end_pos].strip()

                #print(f"Context after image: {context_after} ")  # Debug output
                #print()
                
                # Combine contexts
                combined_context = f"Context before image: {context_before}\n\nContext after image: {context_after}"
                
                # Encode the image as base64
                with open(image_path, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode("utf-8")
                
                # Azure OpenAI endpoint URL
                url = f"https://{azure_domain}.openai.azure.com/openai/deployments/{deployment_name}/chat/completions?api-version={api_version}"
                
                # Headers and payload
                headers = {
                    "Content-Type": "application/json",
                    "api-key": api_key
                }
                
                # Select the appropriate prompt based on detected language
                if document_language == 'de':
                    prompt = GERMAN_PROMPT_TEMPLATE.format(context=combined_context)
                else:
                    prompt = ENGLISH_PROMPT_TEMPLATE.format(context=combined_context)
                
                # Select system prompt based on document content and context
                if "chart" in combined_context.lower() or "graph" in combined_context.lower() or "data" in combined_context.lower():
                    system_content = SYSTEM_PROMPTS["data_viz"]
                elif "photograph" in combined_context.lower() or "photo" in combined_context.lower() or "picture" in combined_context.lower():
                    system_content = SYSTEM_PROMPTS["photo"]
                else:
                    # General purpose system prompt
                    system_content = SYSTEM_PROMPTS["general"]
                
                # Updated payload with dynamic system message and appropriate language
                payload = {
                    "messages": [
                        {
                            "role": "system",
                            "content": system_content
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": prompt
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    }
                                }
                            ]
                        }
                    ],
                    "max_tokens": 2000,
                    "temperature": 0.1
                }
                
                # Call the Azure OpenAI API with timeout
                response = requests.post(url, json=payload, headers=headers, timeout=60)
                
                if response.status_code == 200:
                    response_data = response.json()
                    description = response_data["choices"][0]["message"]["content"].strip()
                    
                    # Check if AI detected it as a logo or decorative element and skip if so
                    logo_indicators = ["this is a logo", "decorative element", "dies ist ein logo", "dekoratives element"]
                    if any(indicator in description.lower() for indicator in logo_indicators):
                        st.info(f"AI detected logo/icon: {image_filename}")
                        skipped_images += 1
                        success = True
                        continue
                    
                    # Remove any "Image X Description:" text that might be in the response
                    description = re.sub(r'^Image\s+\d+\s+Description\s*:', '', description).strip()
                    
                    # Add language-appropriate label
                    if document_language == 'de':
                        desc_label = "[KI generierte Bildbeschreibung:"
                    else:
                        desc_label = "[AI generated image description:"
                    
                    # Prepare the new markdown with description - directly after the image
                    original_img_md = match.group(0)
                    new_img_md = f"{original_img_md}\n\n{desc_label} {description}"
                    
                    # Replace in content - only replace the exact match to avoid confusion
                    content = content.replace(original_img_md, new_img_md)
                    success = True
                    successful_images += 1
                else:
                    error_msg = f"Error with Azure OpenAI API: {response.status_code}"
                    try:
                        error_data = response.json()
                        error_msg += f" - {error_data.get('error', {}).get('message', '')}"
                    except:
                        error_msg += f" - {response.text}"
                    
                    st.warning(error_msg)
                    retries += 1
                    
                    # If this is a rate limit issue (429), wait longer
                    if response.status_code == 429:
                        time.sleep(10 * retries)  # Progressive backoff
                    else:
                        time.sleep(2 * retries)
            
            except requests.exceptions.Timeout:
                st.warning(f"Request timed out for image {image_filename}")
                retries += 1
                time.sleep(5 * retries)  # Progressive backoff
            
            except Exception as e:
                st.error(f"Error processing image {i+1}: {e}")
                retries += 1
                time.sleep(2 * retries)
        
        if not success:
            failed_images.append({"filename": image_filename, "reason": "API error after retries"})
        
        # Clean up temporary file if it exists
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
            except:
                pass
        
        # Add some delay to avoid rate limits
        time.sleep(1)
    
    progress_bar.progress(1.0)

    total_processing_time = time.time() - image_start
    
    # Report results
    # Report results
    progress_text.text(f"Processing complete in {total_processing_time:.2f}s! Successfully described {successful_images} images, skipped {skipped_images} logos/icons, failed on {len(failed_images)}.")
    if failed_images:
        st.warning(f"Failed to process {len(failed_images)} images. Check logs for details.")
        with st.expander("Show failed images"):
            for img in failed_images:
                st.write(f"- {img['filename']}: {img['reason']}")
    
    # Save the updated Markdown
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    # Return the path to the updated file
    return md_file