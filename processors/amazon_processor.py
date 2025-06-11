"""
Amazon Textract processor for document conversion.
Provides document processing using Amazon Textract OCR capabilities with AI image descriptions.
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

class AmazonProcessor(BaseProcessor):
    """
    Amazon Textract processor that leverages AWS Textract's OCR capabilities
    for document processing with optional AI image descriptions.
    """
    
    def __init__(self):
        """Initialize the Amazon Textract processor."""
        self.textract_client = None
        self.s3_client = None
        self.s3_bucket = None
    
    def _check_aws_config(self):
        """
        Check if AWS configuration is valid.
        
        Returns:
            bool: True if configuration is valid, False otherwise
        """
        try:
            # Load environment variables explicitly
            from dotenv import load_dotenv
            load_dotenv()
            
            # Get credentials from environment
            aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
            aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
            aws_default_region = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
            
            # Check if credentials exist
            if not aws_access_key_id or not aws_secret_access_key:
                st.error("AWS credentials not found in environment variables.")
                return False
            
            # Try to create Textract client with explicit credentials
            self.textract_client = boto3.client(
                'textract',
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                region_name=aws_default_region
            )
            
            # Test credentials by accessing service model (no API call needed)
            service_name = self.textract_client._service_model.service_name
            if service_name == 'textract':
                return True
            else:
                return False
            
        except NoCredentialsError:
            st.error("AWS credentials not found. Please configure AWS credentials.")
            return False
        except Exception as e:
            st.error(f"AWS configuration error: {str(e)}")
            return False
    
    def _setup_s3_if_needed(self, file_path):
        """
        Set up S3 bucket and upload file if needed for async processing or large files.
        
        Args:
            file_path (Path): Path to the document file
            
        Returns:
            tuple: (bucket_name, object_key) or (None, None) if direct processing is possible
        """
        file_path = Path(file_path)
        file_size = file_path.stat().st_size
        file_extension = file_path.suffix.lower()
        
        # For PDFs, always try direct processing first if under 5MB
        if file_extension == '.pdf' and file_size <= 5 * 1024 * 1024:
            return None, None  # Try direct processing first
        
        # Check if we can use direct processing (Bytes parameter)
        # Direct processing limits: 5MB for images
        if file_extension in ['.jpg', '.jpeg', '.png'] and file_size <= 5 * 1024 * 1024:
            return None, None  # Can use direct processing
        
        # Need S3 for large files or multi-page PDFs
        try:
            # Load environment variables explicitly
            from dotenv import load_dotenv
            load_dotenv()
            
            # Get region and bucket name
            aws_default_region = os.getenv('AWS_DEFAULT_REGION', 'eu-central-1')
            bucket_name = os.getenv('AWS_TEXTRACT_BUCKET')
            
            # Create S3 client with explicit credentials
            aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
            aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
            
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                region_name=aws_default_region
            )
            
            # Use your permanent bucket if specified, otherwise create temporary
            if bucket_name:
                st.info(f"Using permanent S3 bucket: {bucket_name}")
            else:
                # Fallback to temporary bucket creation
                import time
                timestamp = str(int(time.time()))
                bucket_name = f"textract-temp-eu-{timestamp}"
                
                try:
                    if aws_default_region == 'us-east-1':
                        self.s3_client.create_bucket(Bucket=bucket_name)
                    else:
                        self.s3_client.create_bucket(
                            Bucket=bucket_name,
                            CreateBucketConfiguration={'LocationConstraint': aws_default_region}
                        )
                    st.info(f"Created temporary S3 bucket: {bucket_name}")
                except ClientError as e:
                    st.error(f"Could not create temporary S3 bucket: {e}")
                    return None, None
            
            # Upload file to S3
            object_key = f"textract-input/{file_path.name}"
            
            try:
                self.s3_client.upload_file(str(file_path), bucket_name, object_key)
                st.info(f"Uploaded file to S3: s3://{bucket_name}/{object_key}")
                return bucket_name, object_key
            except ClientError as e:
                st.error(f"Could not upload file to S3: {e}")
                return None, None
                
        except Exception as e:
            st.error(f"S3 setup error: {e}")
            return None, None
    
    def _process_with_textract_direct(self, file_path):
        """
        Process document using Textract's direct API (Bytes parameter).
        
        Args:
            file_path (Path): Path to the document file
            
        Returns:
            dict: Textract response
        """
        # Ensure file_path is converted to string for file operations
        file_path_str = str(file_path)
        
        with open(file_path_str, 'rb') as document:
            response = self.textract_client.detect_document_text(
                Document={'Bytes': document.read()}
            )
        return response
    
    def _process_with_textract_s3(self, bucket_name, object_key, use_async=False):
        """
        Process document using Textract's S3 API.
        
        Args:
            bucket_name (str): S3 bucket name
            object_key (str): S3 object key
            use_async (bool): Whether to use async processing
            
        Returns:
            dict: Textract response
        """
        document_location = {
            'S3Object': {
                'Bucket': bucket_name,
                'Name': object_key
            }
        }
        
        if use_async:
            # Start async job
            response = self.textract_client.start_document_text_detection(
                DocumentLocation=document_location
            )
            job_id = response['JobId']
            
            # Wait for completion
            st.info(f"Started async Textract job: {job_id}")
            
            while True:
                response = self.textract_client.get_document_text_detection(JobId=job_id)
                status = response['JobStatus']
                
                if status == 'SUCCEEDED':
                    return response
                elif status == 'FAILED':
                    raise Exception(f"Textract job failed: {response.get('StatusMessage', 'Unknown error')}")
                else:
                    st.info(f"Textract job status: {status}")
                    time.sleep(5)
        else:
            # Synchronous processing
            response = self.textract_client.detect_document_text(
                Document=document_location
            )
            return response
    
    def _cleanup_s3(self, bucket_name, object_key):
        """
        Clean up temporary S3 objects and bucket (if temporary).
        
        Args:
            bucket_name (str): S3 bucket name
            object_key (str): S3 object key
        """
        try:
            if self.s3_client and bucket_name and object_key:
                # Delete the object first
                self.s3_client.delete_object(Bucket=bucket_name, Key=object_key)
                st.info("Cleaned up temporary S3 file")
                
                # Only delete bucket if it's a temporary bucket (not your permanent one)
                if bucket_name.startswith('textract-temp-'):
                    try:
                        self.s3_client.delete_bucket(Bucket=bucket_name)
                        st.info("Cleaned up temporary S3 bucket")
                    except ClientError as e:
                        # Bucket might not be empty or might not exist, that's okay
                        st.info(f"Could not delete bucket (this is normal): {e}")
                else:
                    st.info(f"Keeping permanent bucket: {bucket_name}")
        except Exception as e:
            st.warning(f"Could not clean up S3 resources: {e}")
            # Don't fail the whole process for cleanup issues
    
    def _convert_textract_to_markdown(self, textract_response):
        """
        Convert Textract response to Markdown format.
        
        Args:
            textract_response (dict): Textract API response
            
        Returns:
            str: Markdown content
        """
        blocks = textract_response.get('Blocks', [])
        
        # Group blocks by page
        pages = {}
        lines = {}
        words = {}
        
        for block in blocks:
            block_type = block.get('BlockType')
            block_id = block.get('Id')
            page_num = block.get('Page', 1)
            
            if block_type == 'PAGE':
                if page_num not in pages:
                    pages[page_num] = []
                pages[page_num].append(block)
            elif block_type == 'LINE':
                if page_num not in lines:
                    lines[page_num] = []
                lines[page_num].append(block)
            elif block_type == 'WORD':
                if page_num not in words:
                    words[page_num] = []
                words[page_num].append(block)
        
        # Build markdown content
        markdown_content = []
        
        # Don't add automatic title - just extract the text
        
        # Process each page
        for page_num in sorted(pages.keys()):
            if len(pages) > 1:  # Only add page headers for multi-page documents
                markdown_content.append(f"## Page {page_num}\n")
            
            # Get lines for this page and sort by position
            page_lines = lines.get(page_num, [])
            
            # Sort lines by their vertical position (top to bottom)
            page_lines.sort(key=lambda line: line.get('Geometry', {}).get('BoundingBox', {}).get('Top', 0))
            
            # Add line text
            for line in page_lines:
                line_text = line.get('Text', '').strip()
                if line_text:
                    # Check if this looks like a heading (simple heuristic)
                    if self._is_likely_heading(line_text, line):
                        markdown_content.append(f"### {line_text}\n")
                    else:
                        markdown_content.append(f"{line_text}\n")
            
            markdown_content.append("\n")  # Add space between pages
        
        return "\n".join(markdown_content)
    
    def _is_likely_heading(self, text, line_block):
        """
        Simple heuristic to determine if a line is likely a heading.
        
        Args:
            text (str): Line text
            line_block (dict): Textract line block
            
        Returns:
            bool: True if likely a heading
        """
        # Simple heuristics for heading detection
        if len(text) < 5:  # Very short text
            return False
        
        if len(text) > 100:  # Very long text is unlikely to be a heading
            return False
        
        # Check if text is mostly uppercase
        if text.isupper() and len(text) > 3:
            return True
        
        # Check if text ends with colon
        if text.endswith(':'):
            return True
        
        # Check font size if available (Textract doesn't always provide this)
        # You could enhance this with more sophisticated analysis
        
        return False
    
    def process_document(self, file_path, doc_dir, images_dir, process_images=True, batch_size=1, status_area=None):
        """
        Process a document with Amazon Textract and optionally add AI descriptions to images.
        
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
            
            # Check AWS configuration
            if not self._check_aws_config():
                status_area.error("Amazon Textract configuration is missing. Please check your AWS credentials.")
                return ProcessingResult(
                    content="",
                    output_file=Path(),
                    images_dir=images_dir,
                    processing_time=0,
                    error="AWS Textract configuration is missing"
                )
            
            # Update status message
            status_area.info("Starting Amazon Textract processing...")
            
            # Ensure paths are Path objects
            doc_dir = Path(doc_dir)
            images_dir = Path(images_dir)
            
            # Determine processing method
            bucket_name, object_key = self._setup_s3_if_needed(file_path)
            
            # Process the document with Textract
            if bucket_name and object_key:
                status_area.info(f"Processing document via S3: {file_path.name}")
                
                # For large files or PDFs, consider async processing
                file_size = file_path.stat().st_size
                use_async = file_size > 10 * 1024 * 1024  # Use async for files > 10MB
                
                try:
                    textract_response = self._process_with_textract_s3(bucket_name, object_key, use_async)
                except Exception as e:
                    # If S3 processing fails, try to fall back to direct processing for smaller files
                    if file_size <= 5 * 1024 * 1024:
                        status_area.warning(f"S3 processing failed: {e}. Trying direct processing...")
                        textract_response = self._process_with_textract_direct(file_path)
                    else:
                        raise e
                
                # Clean up S3 file
                self._cleanup_s3(bucket_name, object_key)
            else:
                status_area.info(f"Processing document directly: {file_path.name}")
                try:
                    textract_response = self._process_with_textract_direct(file_path)
                except Exception as e:
                    if "UnsupportedDocumentException" in str(e):
                        status_area.error(f"PDF format not supported by Textract: {file_path.name}")
                        status_area.info("Try using Docling or Mistral processors for this PDF format.")
                        raise Exception(f"Unsupported PDF format. Please try Docling or Mistral processors.")
                    else:
                        raise e
            
            # Convert Textract response to markdown
            status_area.info("Converting Textract results to Markdown...")
            markdown_content = self._convert_textract_to_markdown(textract_response)
            
            # Save the markdown content to a file
            output_file = doc_dir / f"{file_path.stem}_textract.md"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(markdown_content)
            
            # Calculate Textract processing time
            textract_processing_time = self._stop_timer(start_time)
            
            status_area.success(f"Amazon Textract processing complete in {textract_processing_time:.2f}s!")
            
            # Add AI descriptions to images if requested and Azure OpenAI is configured
            # Note: Textract doesn't extract images like Docling/Mistral, so this would mainly be
            # for any images that might be referenced in the document or for future enhancement
            if process_images:
                # Check if Azure OpenAI is configured for image descriptions
                from services.azure_openai import check_azure_config
                if check_azure_config():
                    status_area.info("Checking for images to process...")
                    
                    # Since Textract doesn't extract images directly, we could add logic here
                    # to handle images if they exist in the document or are referenced
                    # For now, we'll skip this since Textract focuses on text extraction
                    status_area.info("Amazon Textract focuses on text extraction. Image processing skipped.")
                else:
                    status_area.info("Azure OpenAI not configured - skipping AI image descriptions")
            
            # Calculate total processing time
            total_processing_time = textract_processing_time + image_processing_time
            
            # CRITICAL: Update session state directly
            st.session_state["markdown_content"] = markdown_content
            st.session_state["md_file_path"] = str(output_file)
            st.session_state["img_dir"] = str(images_dir)
            st.session_state["processing_status"] = "docling_done"  # Keep this for compatibility
            st.session_state["docling_processing_time"] = textract_processing_time  # For timing info
            st.session_state["image_processing_time"] = image_processing_time
            if st.session_state.processing_start_time:
                st.session_state["total_processing_time"] = time.time() - st.session_state.processing_start_time
            
            # Final status message
            if image_processing_time > 0:
                status_area.success(f"Amazon Textract processing complete in {total_processing_time:.2f}s! (OCR: {textract_processing_time:.2f}s, AI Descriptions: {image_processing_time:.2f}s)")
            else:
                status_area.success(f"Amazon Textract processing complete in {total_processing_time:.2f}s!")
            
            # Return the processing result
            return ProcessingResult(
                content=markdown_content,
                output_file=output_file,
                images_dir=images_dir,
                processing_time=textract_processing_time,
                image_processing_time=image_processing_time
            )
        
        except Exception as e:
            # Calculate processing time even if there's an error
            processing_time = self._stop_timer(start_time)
            
            error_message = str(e)
            status_area.error(f"Error during Amazon Textract processing: {error_message}")
            
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
        # Load environment variables explicitly
        from dotenv import load_dotenv
        load_dotenv()
        
        # Get credentials from environment
        aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
        aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        aws_default_region = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
        
        # Check if credentials exist
        if not aws_access_key_id or not aws_secret_access_key:
            return False
            
        # Try to create Textract client with explicit credentials
        textract_client = boto3.client(
            'textract',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=aws_default_region
        )
        
        # Test credentials by getting service endpoints (this doesn't require actual API calls)
        textract_client._service_model.service_name
        
        return True
    except NoCredentialsError:
        return False
    except Exception as e:
        # For debugging - you can remove this later
        print(f"AWS config check failed: {e}")
        return False