"""
Docling processor for high-quality document conversion.
Provides extensive document processing with image extraction and enhancement.
"""

import os
import time
import tempfile
import shutil
from pathlib import Path

import streamlit as st
from config import config
from processors.base_processor import BaseProcessor, ProcessingResult
from services.azure_openai import answer_with_azure_openai, add_descriptions_with_azure_openai
from utils.image_utils import fix_image_paths
from typing import Union

class DoclingProcessor(BaseProcessor):
    """
    Docling document processor that provides high-quality conversion
    with image extraction and AI enhancement.
    """
    
    def process_document(self, file_path, doc_dir, images_dir, process_images=True, batch_size=1, status_area=None):
        """
        Process a document with Docling.
        
        Args:
            file_path (Path): Path to the document file
            doc_dir (Path): Directory to save processed document
            images_dir (Path): Directory to save extracted images
            process_images (bool): Whether to process images with AI
            batch_size (int): Number of images to process in a batch
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
            # Update status message
            status_area.info("Starting Docling processing...")
            
            # Convert file_path to string if it's not already
            if not isinstance(file_path, (str, Path)):
                raise TypeError(f"file_path must be a string or Path object, got {type(file_path)}")
            
            # Convert to Path objects
            file_path = Path(file_path)
            doc_dir = Path(doc_dir)
            images_dir = Path(images_dir)
            
            # Check if file exists
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Use a temporary directory for initial conversion
            import tempfile
            temp_dir = Path(tempfile.mkdtemp())
            
            # Import the necessary modules here
            from docling.document_converter import DocumentConverter, PdfFormatOption
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling_core.types.doc import ImageRefMode
            
            # Get file extension
            file_extension = file_path.suffix.lower()
            
            # Set up format options
            format_options = {}
            allowed_formats = []
            
            if file_extension in ['.pdf']:
                # Add PDF format options
                pipeline_options = PdfPipelineOptions()
                pipeline_options.generate_picture_images = True
                pipeline_options.images_scale = config.DOCLING_DPI_SCALE
                format_options[InputFormat.PDF] = PdfFormatOption(pipeline_options=pipeline_options)
                allowed_formats.append(InputFormat.PDF)
            elif file_extension in ['.docx', '.doc']:
                allowed_formats.append(InputFormat.DOCX)
            elif file_extension in ['.pptx', '.ppt']:
                allowed_formats.append(InputFormat.PPTX)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
            
            # Create the document converter
            doc_converter = DocumentConverter(
                allowed_formats=allowed_formats,
                format_options=format_options
            )
            
            # Convert the document directly
            t0 = time.perf_counter()
            status_area.info(f"Converting '{file_path.name}' with Docling...")
            
            # Define output paths
            md_file_temp = temp_dir / f"{file_path.stem}.md"
            img_dir_temp = temp_dir / "images"
            img_dir_temp.mkdir(exist_ok=True)
            
            # Convert & export
            conv_result = doc_converter.convert(str(file_path))
            doc = conv_result.document
            doc.save_as_markdown(
                filename=md_file_temp,
                artifacts_dir=img_dir_temp,
                image_mode=ImageRefMode.REFERENCED,
            )
            
            status_area.success(f"Converted '{file_path.name}' → '{md_file_temp}' in {time.perf_counter()-t0:.2f}s")
            
            # Record Docling processing time
            docling_time = self._stop_timer(start_time)
            
            # Read the markdown content
            with open(md_file_temp, 'r', encoding='utf-8') as f:
                markdown_content = f.read()
            
            # Fix image paths and copy images to project directory
            from utils.image_utils import fix_image_paths
            markdown_content = fix_image_paths(markdown_content, img_dir_temp, images_dir)
            
            # Save the fixed markdown content to the project directory
            md_file = doc_dir / f"{file_path.stem}.md"
            with open(md_file, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            
            # Process images if selected
            if process_images:
                status_area.info("Adding AI descriptions to images...")
                img_start_time = self._start_timer()
                
                from services.azure_openai import add_descriptions_with_azure_openai
                md_file = add_descriptions_with_azure_openai(
                    md_file, 
                    images_dir, 
                    config.AZURE_DOMAIN, 
                    config.AZURE_API_KEY, 
                    config.AZURE_DEPLOYMENT_NAME,
                    config.AZURE_API_VERSION,
                    batch_size
                )
                
                # Calculate image processing time
                image_processing_time = self._stop_timer(img_start_time)
                status_area.success("Added AI descriptions to images")
            
            # Read the final Markdown content
            with open(md_file, 'r', encoding='utf-8') as f:
                final_markdown_content = f.read()
            
            # Calculate total processing time
            total_processing_time = docling_time + image_processing_time
            
            # CRITICAL: Update session state directly
            st.session_state["markdown_content"] = final_markdown_content
            st.session_state["md_file_path"] = str(md_file)
            st.session_state["img_dir"] = str(images_dir)
            st.session_state["processing_status"] = "docling_done"
            st.session_state["docling_error"] = None
            st.session_state["docling_processing_time"] = docling_time
            st.session_state["image_processing_time"] = image_processing_time
            
            # Calculate total processing time
            if st.session_state.processing_start_time:
                st.session_state.total_processing_time = time.time() - st.session_state.processing_start_time
            
            if image_processing_time > 0:
                status_area.success(f"Document processing complete in {total_processing_time:.2f}s! (Docling: {docling_time:.2f}s, Images: {image_processing_time:.2f}s)")
            else:
                status_area.success(f"Document processing complete in {total_processing_time:.2f}s!")
            
            # Return the processing result
            return ProcessingResult(
                content=final_markdown_content,
                output_file=md_file,
                images_dir=images_dir,
                processing_time=docling_time,
                image_processing_time=image_processing_time
            )
        
        except Exception as e:
            # Calculate processing time even if there's an error
            processing_time = self._stop_timer(start_time)
            
            error_message = str(e)
            status_area.error(f"Error during Docling processing: {error_message}")
            
            # Log the full error for debugging
            import traceback
            status_area.error(f"Full error: {traceback.format_exc()}")
            
            # Update session state
            if st.session_state.processing_start_time:
                st.session_state.total_processing_time = time.time() - st.session_state.processing_start_time
            st.session_state.docling_error = error_message
            st.session_state.processing_status = "docling_error"
            
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
        Answer a question based on the Docling-processed content.
        
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
    
    def convert_document_to_markdown(
        self,
        source: Union[str, Path],
        *,
        out_dir: Union[str, Path, None] = None,
        dpi_scale: float = 2.0,
        verbose: bool = True,
    ) -> tuple:
        """
        Convert various document formats (PDF, DOCX, PPTX) to Markdown using Docling.
        Returns the path to the Markdown file and the images directory.
        
        Args:
            source: Path to the source document (PDF, DOCX, or PPTX)
            out_dir: Output directory for the Markdown and images
            dpi_scale: Resolution scale for extracted images
            verbose: Whether to print conversion information
            
        Returns:
            tuple: (markdown_file_path, images_directory_path)
        """
        # Import Docling here to avoid import errors if not installed
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling_core.types.doc import ImageRefMode
        
        # Ensure source is a Path object and exists
        src = Path(source).expanduser().resolve()
        if not src.exists():
            raise FileNotFoundError(f"Source file not found: {src}")
        
        # Where to put results - ensure out_dir is a valid Path
        if out_dir is None:
            out_root = src.with_suffix("")
        else:
            out_root = Path(out_dir)
        
        # Create output directory if it doesn't exist
        out_root.mkdir(parents=True, exist_ok=True)
        
        md_file = out_root / f"{src.stem}.md"
        img_dir = out_root / "images"  # auto-created by Docling
        img_dir.mkdir(exist_ok=True)  # Ensure the directory exists
        
        # Determine the file format based on extension
        file_extension = src.suffix.lower()
        
        # Create format-specific options
        format_options = {}
        allowed_formats = []
        
        if file_extension in ['.pdf']:
            # Add PDF format options
            pipeline_options = PdfPipelineOptions()
            pipeline_options.generate_picture_images = True
            pipeline_options.images_scale = dpi_scale
            format_options[InputFormat.PDF] = PdfFormatOption(pipeline_options=pipeline_options)
            allowed_formats.append(InputFormat.PDF)
        elif file_extension in ['.docx', '.doc']:
            # Add DOCX format options
            allowed_formats.append(InputFormat.DOCX)
            # Note: We don't need any specific format options for DOCX, using the defaults
        elif file_extension in ['.pptx', '.ppt']:
            # Add PPTX format options
            allowed_formats.append(InputFormat.PPTX)
            # Note: We don't need any specific format options for PPTX, using the defaults
        else:
            raise ValueError(f"Unsupported file format: {file_extension}. Supported formats: .pdf, .docx, .doc, .pptx, .ppt")
        
        # Create the document converter with the appropriate format options
        doc_converter = DocumentConverter(
            allowed_formats=allowed_formats,
            format_options=format_options
        )
        
        # Convert & export
        t0 = time.perf_counter()
        conv_result = doc_converter.convert(str(src))
        doc = conv_result.document
        doc.save_as_markdown(
            filename=md_file,
            artifacts_dir=img_dir,
            image_mode=ImageRefMode.REFERENCED,
        )
        
        if verbose:
            st.write(f"Converted '{src.name}' → '{md_file}' in {time.perf_counter()-t0:.2f}s")
        
        return md_file, img_dir