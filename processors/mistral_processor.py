"""
Mistral OCR processor for document conversion.
Provides document processing using Mistral OCR capabilities with AI image descriptions.
"""
import streamlit as st
import time
from pathlib import Path
import os
import shutil
import tempfile
from config import config
from processors.base_processor import BaseProcessor, ProcessingResult
from services.azure_openai import answer_with_azure_openai, add_descriptions_with_azure_openai
from services.mistral_api import (
    process_document_with_mistral_file,
    combine_page_markdown,
    check_mistral_config
)

class MistralProcessor(BaseProcessor):
    """
    Mistral OCR processor that leverages Mistral's OCR capabilities
    for document processing with optional AI image descriptions.
    """
    
    def process_document(self, file_path, doc_dir, images_dir, process_images=True, batch_size=1, status_area=None):
        """
        Process a document with Mistral OCR and optionally add AI descriptions to images.
        
        Args:
            file_path (Path): Path to the document file
            doc_dir (Path): Directory to save processed document
            images_dir (Path): Directory to save extracted images
            process_images (bool): Whether to add AI descriptions to images
            batch_size (int): Number of images to process in a batch for AI descriptions
            status_area: Streamlit container for status updates
            
        Returns:
            ProcessingResult: Results of the document processing
        """
        # Define status_area if not provided
        if status_area is None:
            if 'processing_status_area' in st.session_state and st.session_state.processing_status_area is not None:
                status_area = st.session_state.processing_status_area
            else:
                status_area = st.container()
        
        # Start timer
        start_time = self._start_timer()
        image_processing_time = 0.0
        
        try:
            # Ensure file_path is a Path object
            file_path = Path(file_path)
            if not file_path.exists():
                status_area.error(f"File not found: {file_path}")
                return ProcessingResult(
                    content="",
                    output_file=Path(),
                    images_dir=images_dir,
                    processing_time=0,
                    error="File not found"
                )
            
            # Check Mistral configuration
            if not check_mistral_config():
                status_area.error("Mistral API configuration is missing. Please check your .env file.")
                return ProcessingResult(
                    content="",
                    output_file=Path(),
                    images_dir=images_dir,
                    processing_time=0,
                    error="Mistral API configuration is missing"
                )
            
            # Update status message
            status_area.info("Starting Mistral OCR processing...")
            
            # Ensure paths are Path objects
            doc_dir = Path(doc_dir)
            images_dir = Path(images_dir)
            
            # Process the document with Mistral OCR
            status_area.info(f"Sending document to Mistral OCR API: {file_path.name}")
            
            # Always include images for proper processing
            include_image_base64 = True
            
            # Process the document
            ocr_results = process_document_with_mistral_file(
                file_path=file_path,
                include_image_base64=include_image_base64,
                image_limit=10 * 10  # Use batch_size as a multiplier for image limit 
            )
            
            # Check for errors
            if isinstance(ocr_results, dict) and "error" in ocr_results:
                status_area.error(f"Error from Mistral OCR API: {ocr_results['error']}")
                return ProcessingResult(
                    content="",
                    output_file=Path(),
                    images_dir=images_dir,
                    processing_time=self._stop_timer(start_time),
                    error=ocr_results["error"]
                )
            
            # Process the OCR results
            status_area.info("Processing Mistral OCR results...")
            
            # Use the combine_page_markdown function to process images and markdown
            from services.mistral_api import combine_page_markdown
            markdown_content = combine_page_markdown(ocr_results, images_dir)
            
            # Save the markdown content to a file
            output_file = doc_dir / f"{file_path.stem}_mistral.md"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(markdown_content)
            
            # Calculate Mistral processing time
            mistral_processing_time = self._stop_timer(start_time)
            
            status_area.success(f"Mistral OCR processing complete in {mistral_processing_time:.2f}s!")
            
            # Add AI descriptions to images if requested and Azure OpenAI is configured
            if process_images:
                # Check if Azure OpenAI is configured for image descriptions
                from services.azure_openai import check_azure_config
                if check_azure_config():
                    status_area.info("Adding AI descriptions to images...")
                    img_start_time = self._start_timer()
                    
                    try:
                        # Use Azure OpenAI to add descriptions to images
                        enhanced_md_file = add_descriptions_with_azure_openai(
                            output_file, 
                            images_dir, 
                            config.AZURE_DOMAIN, 
                            config.AZURE_API_KEY, 
                            config.AZURE_DEPLOYMENT_NAME,
                            config.AZURE_API_VERSION,
                            batch_size
                        )
                        
                        # Read the enhanced markdown content
                        with open(enhanced_md_file, 'r', encoding='utf-8') as f:
                            markdown_content = f.read()
                        
                        # Calculate image processing time
                        image_processing_time = self._stop_timer(img_start_time)
                        status_area.success(f"Added AI descriptions to images in {image_processing_time:.2f}s!")
                        
                    except Exception as e:
                        status_area.warning(f"Could not add AI descriptions: {str(e)}")
                        image_processing_time = 0
                else:
                    status_area.info("Azure OpenAI not configured - skipping AI image descriptions")
            
            # Calculate total processing time
            total_processing_time = mistral_processing_time + image_processing_time
            
            # CRITICAL: Update session state directly
            st.session_state["markdown_content"] = markdown_content
            st.session_state["md_file_path"] = str(output_file)
            st.session_state["img_dir"] = str(images_dir)
            st.session_state["processing_status"] = "docling_done"  # Keep this for compatibility
            st.session_state["docling_processing_time"] = mistral_processing_time  # For timing info
            st.session_state["image_processing_time"] = image_processing_time
            if st.session_state.processing_start_time:
                st.session_state["total_processing_time"] = time.time() - st.session_state.processing_start_time
            
            # Final status message
            if image_processing_time > 0:
                status_area.success(f"Mistral OCR processing complete in {total_processing_time:.2f}s! (OCR: {mistral_processing_time:.2f}s, AI Descriptions: {image_processing_time:.2f}s)")
            else:
                status_area.success(f"Mistral OCR processing complete in {total_processing_time:.2f}s!")
            
            # Return the processing result
            return ProcessingResult(
                content=markdown_content,
                output_file=output_file,
                images_dir=images_dir,
                processing_time=mistral_processing_time,
                image_processing_time=image_processing_time
            )
        
        except Exception as e:
            # Calculate processing time even if there's an error
            processing_time = self._stop_timer(start_time)
            
            error_message = str(e)
            status_area.error(f"Error during Mistral OCR processing: {error_message}")
            
            # Log the full error for debugging
            import traceback
            status_area.error(f"Full error: {traceback.format_exc()}")
            
            # Return error result
            return ProcessingResult(
                content="",
                output_file=Path(),
                images_dir=images_dir,
                processing_time=processing_time,
                error=error_message
            )
    
    def answer_question(self, question, context=None):
        """
        Answer a question based on the Mistral-processed content.
        
        Args:
            question (str): The question to answer
            context (str, optional): Content to use as context (if None, use session state)
            
        Returns:
            str: The answer to the question
        """
        # Use session state content if context not provided
        if context is None and 'markdown_content' in st.session_state:
            context = st.session_state.markdown_content
        
        if not context:
            return "No document content available to answer questions."
        
        try:
            # Use Azure OpenAI to answer the question
            return answer_with_azure_openai(
                question=question,
                content=context,
                azure_domain=config.AZURE_DOMAIN,
                api_key=config.AZURE_API_KEY,
                deployment_name=config.AZURE_DEPLOYMENT_NAME,
                api_version=config.AZURE_API_VERSION
            )
        except Exception as e:
            return f"Error answering question: {str(e)}"