"""
Amazon Textract processor using Textractor library for better markdown output.
Uses the official amazon-textract-textractor library for cleaner implementation.
"""
import streamlit as st
import time
import json
import boto3
from pathlib import Path
import os
import tempfile
from botocore.exceptions import ClientError, NoCredentialsError
from config import config
from processors.base_processor import BaseProcessor, ProcessingResult
from services.azure_openai import answer_with_azure_openai, add_descriptions_with_azure_openai

# Import textractor
try:
    from textractor import Textractor
    from textractor.data.constants import TextractFeatures
    TEXTRACTOR_AVAILABLE = True
except ImportError:
    TEXTRACTOR_AVAILABLE = False
    st.warning("amazon-textract-textractor not installed. Install with: pip install amazon-textract-textractor")

class AmazonProcessor(BaseProcessor):
    """
    Amazon Textract processor using the official Textractor library.
    Provides clean markdown output with proper structure detection.
    """
    
    def __init__(self):
        """Initialize the Amazon Textract processor."""
        self.textract_client = None
        self.s3_client = None
        self.textractor = None
    
    def _check_aws_config(self):
        """
        Check if AWS configuration is valid and initialize Textractor.
        
        Returns:
            bool: True if configuration is valid, False otherwise
        """
        try:
            # Check if textractor is available
            if not TEXTRACTOR_AVAILABLE:
                st.error("amazon-textract-textractor library is not installed.")
                st.info("Install it with: pip install amazon-textract-textractor")
                return False
            
            # Load environment variables
            from dotenv import load_dotenv
            load_dotenv()
            
            # Get credentials from environment
            aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
            aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
            aws_default_region = os.getenv('AWS_DEFAULT_REGION', 'eu-central-1')
            
            # Check if credentials exist
            if not aws_access_key_id or not aws_secret_access_key:
                st.error("AWS credentials not found in environment variables.")
                return False
            
            # Create boto3 clients for direct use if needed
            self.textract_client = boto3.client(
                'textract',
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                region_name=aws_default_region
            )
            
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                region_name=aws_default_region
            )
            
            # Initialize Textractor
            self.textractor = Textractor(region_name=aws_default_region)
            
            st.info(f"AWS Textract configured successfully in region: {aws_default_region}")
            return True
            
        except Exception as e:
            st.error(f"AWS configuration error: {str(e)}")
            return False
    
    def _determine_processing_method(self, file_path):
        """
        Determine if we need S3 for processing.
        
        Args:
            file_path (Path): Path to the document file
            
        Returns:
            dict: Processing configuration
        """
        file_path = Path(file_path)
        file_size = file_path.stat().st_size
        file_extension = file_path.suffix.lower()
        
        # Size threshold for S3
        DIRECT_LIMIT = 5 * 1024 * 1024  # 5MB
        
        processing_config = {
            'use_s3': False,
            'supported': True
        }
        
        # Check file type support
        if file_extension not in ['.pdf', '.jpg', '.jpeg', '.png', '.tiff']:
            processing_config['supported'] = False
            return processing_config
        
        # PDFs and large files need S3 for async processing
        if file_extension == '.pdf' or file_size > DIRECT_LIMIT:
            processing_config['use_s3'] = True
        
        return processing_config
    
    def _upload_to_s3(self, file_path):
        """
        Upload file to S3 for processing.
        
        Args:
            file_path (Path): Path to the document file
            
        Returns:
            str: S3 path (s3://bucket/key) or None if failed
        """
        try:
            # Get bucket configuration
            bucket_name = 'textract-bucket-506'  # Your bucket
            
            # Generate unique object key
            import uuid
            timestamp = int(time.time())
            unique_id = str(uuid.uuid4())[:8]
            object_key = f"textract-input/{timestamp}-{unique_id}-{file_path.name}"
            
            # Upload file
            st.info(f"Uploading file to S3...")
            self.s3_client.upload_file(str(file_path), bucket_name, object_key)
            
            s3_path = f"s3://{bucket_name}/{object_key}"
            st.success(f"Uploaded file to: {s3_path}")
            
            return s3_path, bucket_name, object_key
                
        except Exception as e:
            st.error(f"S3 upload error: {str(e)}")
            return None, None, None
    
    def _cleanup_s3(self, bucket_name, object_key):
        """Clean up S3 file after processing."""
        try:
            if bucket_name and object_key:
                self.s3_client.delete_object(Bucket=bucket_name, Key=object_key)
                st.info("Cleaned up S3 file")
        except:
            pass
    
    def _extract_figures_info(self, document):
        """
        Extract information about figures detected in the document.
        
        Args:
            document: Textractor document object
            
        Returns:
            list: List of figure information
        """
        figures = []
        figure_num = 1
        
        # Iterate through pages
        for page in document.pages:
            # Check for layout elements
            if hasattr(page, 'layout'):
                # Look for figures in layout
                for layout_element in page.layout.children:
                    if hasattr(layout_element, 'layout_type') and 'FIGURE' in str(layout_element.layout_type):
                        figures.append({
                            'page': page.page_num,
                            'number': figure_num,
                            'bbox': layout_element.bbox if hasattr(layout_element, 'bbox') else None,
                            'text': layout_element.text if hasattr(layout_element, 'text') else f"Figure {figure_num}"
                        })
                        figure_num += 1
        
        return figures
    
    def process_document(self, file_path, doc_dir, images_dir, process_images=True, batch_size=1, status_area=None):
        """
        Process a document with Amazon Textract using Textractor library.
        
        Args:
            file_path (Path): Path to the document file
            doc_dir (Path): Directory to save processed document
            images_dir (Path): Directory to save extracted images
            process_images (bool): Whether to add AI descriptions to images
            batch_size (int): Number of images to process in a batch
            status_area: Streamlit container for status updates
            
        Returns:
            ProcessingResult: Results of the document processing
        """
        # Define status_area if not provided
        if status_area is None:
            status_area = st.container()
        
        # Start timer
        start_time = self._start_timer()
        image_processing_time = 0.0
        
        # Initialize variables
        s3_path = None
        bucket_name = None
        object_key = None
        
        try:
            # Check if file_path is None
            if file_path is None:
                status_area.error("No file path provided for processing")
                return ProcessingResult(
                    content="",
                    output_file=Path(),
                    images_dir=images_dir if images_dir else Path(),
                    processing_time=0,
                    error="No file path provided"
                )
            
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
            
            # Check AWS configuration
            if not self._check_aws_config():
                return ProcessingResult(
                    content="",
                    output_file=Path(),
                    images_dir=images_dir,
                    processing_time=0,
                    error="AWS Textract configuration is missing"
                )
            
            # Update status
            status_area.info("Starting Amazon Textract processing with Textractor...")
            
            # Ensure paths are Path objects
            doc_dir = Path(doc_dir)
            images_dir = Path(images_dir)
            
            # Determine processing method
            processing_config = self._determine_processing_method(file_path)
            
            if not processing_config['supported']:
                status_area.error(f"Unsupported file format: {file_path.suffix}")
                return ProcessingResult(
                    content="",
                    output_file=Path(),
                    images_dir=images_dir,
                    processing_time=0,
                    error="Unsupported file format"
                )
            
            # Define features to extract
            features = [
                TextractFeatures.LAYOUT,
                TextractFeatures.TABLES,
                TextractFeatures.FORMS
            ]
            
            # Process based on file location
            if processing_config['use_s3']:
                # Upload to S3
                s3_path, bucket_name, object_key = self._upload_to_s3(file_path)
                if not s3_path:
                    raise Exception("Failed to upload file to S3")
                
                # Process from S3 (async for PDFs)
                if file_path.suffix.lower() == '.pdf':
                    status_area.info("Processing PDF asynchronously from S3...")
                    document = self.textractor.start_document_analysis(
                        file_source=s3_path,
                        features=features,
                        save_image=False  # Don't save images, we'll handle separately
                    )
                else:
                    status_area.info("Processing document from S3...")
                    document = self.textractor.analyze_document(
                        file_source=s3_path,
                        features=features,
                        save_image=False
                    )
            else:
                # Process directly
                status_area.info("Processing document directly...")
                document = self.textractor.analyze_document(
                    file_source=str(file_path),
                    features=features,
                    save_image=False
                )
            
            # Extract figure information
            figures = self._extract_figures_info(document)
            if figures:
                status_area.info(f"Detected {len(figures)} figures in the document")
            
            # Convert to markdown using Textractor's built-in method
            status_area.info("Converting to Markdown...")
            markdown_content = document.to_markdown()
            
            # Add figure placeholders if any were detected
            if figures:
                markdown_content += "\n\n## Detected Figures\n\n"
                for fig in figures:
                    markdown_content += f"- **Figure {fig['number']}** on page {fig['page']}: {fig['text']}\n"
                markdown_content += "\n*Note: To extract actual images from PDF, use Docling or Mistral processors.*\n"
            
            # Save the markdown content
            output_file = doc_dir / f"{file_path.stem}_textract.md"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(markdown_content)
            
            # Also save the raw response for debugging
            if hasattr(document, 'response'):
                json_file = doc_dir / f"{file_path.stem}_textract_response.json"
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(document.response, f, indent=2)
                st.info(f"Saved raw response to: {json_file}")
            
            # Clean up S3 if used
            if bucket_name and object_key:
                self._cleanup_s3(bucket_name, object_key)
            
            # Calculate processing time
            textract_processing_time = self._stop_timer(start_time)
            
            status_area.success(f"Amazon Textract processing complete in {textract_processing_time:.2f}s!")
            
            # Note about image processing
            if process_images and figures:
                status_area.info(f"Note: Textract detected {len(figures)} figures but cannot extract the actual images.")
                status_area.info("Consider using Docling or Mistral processors for image extraction.")
            
            # Update session state
            st.session_state["markdown_content"] = markdown_content
            st.session_state["md_file_path"] = str(output_file)
            st.session_state["img_dir"] = str(images_dir)
            st.session_state["processing_status"] = "docling_done"
            st.session_state["docling_processing_time"] = textract_processing_time
            st.session_state["image_processing_time"] = image_processing_time
            if st.session_state.processing_start_time:
                st.session_state["total_processing_time"] = time.time() - st.session_state.processing_start_time
            
            # Return the processing result
            return ProcessingResult(
                content=markdown_content,
                output_file=output_file,
                images_dir=images_dir,
                processing_time=textract_processing_time,
                image_processing_time=image_processing_time
            )
        
        except Exception as e:
            # Calculate processing time
            processing_time = self._stop_timer(start_time)
            
            error_message = str(e)
            
            # Create error display
            error_container = status_area.container()
            error_container.error(f"Error during Amazon Textract processing: {error_message}")
            
            # Log full error
            import traceback
            full_traceback = traceback.format_exc()
            error_container.error(f"Full error details:")
            error_container.code(full_traceback)
            
            # Clean up S3 if there was an error
            if bucket_name and object_key:
                try:
                    self._cleanup_s3(bucket_name, object_key)
                except:
                    pass
            
            # Update session state
            st.session_state["processing_status"] = "error"
            st.session_state["docling_error"] = error_message
            
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
        Answer a question based on the Textract-processed content.
        
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


def check_textract_config():
    """
    Check if Amazon Textract configuration is valid.
    
    Returns:
        bool: True if configuration is valid, False otherwise
    """
    try:
        # Check if textractor is available
        if not TEXTRACTOR_AVAILABLE:
            return False
            
        from dotenv import load_dotenv
        load_dotenv()
        
        # Get credentials from environment
        aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
        aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        aws_default_region = os.getenv('AWS_DEFAULT_REGION', 'eu-central-1')
        
        # Check if credentials exist
        if not aws_access_key_id or not aws_secret_access_key:
            return False
            
        # Try to create Textract client
        textract_client = boto3.client(
            'textract',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=aws_default_region
        )
        
        # Test credentials
        textract_client._service_model.service_name
        
        return True
    except Exception as e:
        print(f"AWS config check failed: {e}")
        return False