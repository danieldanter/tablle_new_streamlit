
import streamlit as st
import os
import time
from pathlib import Path
from dotenv import load_dotenv

# Import from our modules (we'll create these next)
from config import (
    initialize_config,
    DOCS_DIR, 
    PROJECT_DIR
)
from utils.ui_utils import (
    display_quick_extract, 
    render_markdown_with_images,
    auto_refresh,
    show_config_status
)
from processors.processor_factory import get_processor
from services.azure_openai import check_azure_config  # Add this import here

# Load environment variables and initialize configuration
load_dotenv()
config = initialize_config()

# Set page title and layout
st.set_page_config(
    page_title="Document Enhancer with AI",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize session state variables
def init_session_state():
    """Initialize all session state variables in one place"""
    if 'markdown_content' not in st.session_state:
        st.session_state.markdown_content = None
    if 'md_file_path' not in st.session_state:
        st.session_state.md_file_path = None
    if 'img_dir' not in st.session_state:
        st.session_state.img_dir = None
    if 'doc_id' not in st.session_state:
        st.session_state.doc_id = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'processing_status' not in st.session_state:
        st.session_state.processing_status = "idle"  # idle, quick_extract_done, docling_in_progress, docling_done
    if 'quick_extract_content' not in st.session_state:
        st.session_state.quick_extract_content = None
    if 'quick_extract_file' not in st.session_state:
        st.session_state.quick_extract_file = None
    if 'start_docling_processing' not in st.session_state:
        st.session_state.start_docling_processing = False
    if 'docling_error' not in st.session_state:
        st.session_state.docling_error = None
    # Timing variables
    if 'quick_extract_time' not in st.session_state:
        st.session_state.quick_extract_time = None
    if 'docling_processing_time' not in st.session_state:
        st.session_state.docling_processing_time = None
    if 'image_processing_time' not in st.session_state:
        st.session_state.image_processing_time = None
    if 'total_processing_time' not in st.session_state:
        st.session_state.total_processing_time = None
    if 'processing_start_time' not in st.session_state:
        st.session_state.processing_start_time = None
    # Processor selection
    if 'selected_processor' not in st.session_state:
        st.session_state.selected_processor = "docling"  # Default processor

def start_document_processing(uploaded_file, process_images=True, batch_size=1):
    """Start document processing with selected processor"""
    try:
        # Save chat history temporarily
        chat_history = st.session_state.chat_history if 'chat_history' in st.session_state else []
        
        # Reset processing state
        st.session_state.processing_status = "idle"
        st.session_state.quick_extract_content = None
        st.session_state.quick_extract_file = None
        st.session_state.docling_error = None
        st.session_state.markdown_content = None
        st.session_state.md_file_path = None
        st.session_state.img_dir = None
        st.session_state.start_docling_processing = False
        
        # Reset timing information
        st.session_state.quick_extract_time = None
        st.session_state.docling_processing_time = None
        st.session_state.image_processing_time = None
        st.session_state.total_processing_time = None
        
        # Start overall processing timer
        st.session_state.processing_start_time = time.time()
        
        # Restore chat history
        st.session_state.chat_history = chat_history
        
        # Create a unique ID for this document
        import uuid
        doc_id = str(uuid.uuid4())[:8]
        st.session_state.doc_id = doc_id
        
        # Create a directory for this document
        doc_dir = DOCS_DIR / doc_id
        doc_dir.mkdir(exist_ok=True)
        
        # Create a directory for images
        images_dir = doc_dir / "images"
        images_dir.mkdir(exist_ok=True)
        
        # Save the uploaded file to the document directory
        file_path = doc_dir / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Get the appropriate processor
        processor = get_processor("quick")  # Start with quick processor
        
        # STAGE 1: Perform quick extraction
        with st.spinner("Extracting basic text content..."):
            quick_result = processor.process_document(
                file_path=file_path,
                doc_dir=doc_dir,
                images_dir=images_dir,
                process_images=process_images,
                batch_size=batch_size
            )
            
            # Store quick extract results
            st.session_state.quick_extract_content = quick_result.content
            st.session_state.quick_extract_file = quick_result.output_file
            st.session_state.quick_extract_time = quick_result.processing_time
            
            # Store document info for later processing
            st.session_state.docling_file_path = file_path
            st.session_state.docling_doc_dir = doc_dir
            st.session_state.docling_images_dir = images_dir
            st.session_state.docling_process_images = process_images
            st.session_state.docling_batch_size = batch_size
            
            # Set status to quick extraction done
            st.session_state.processing_status = "quick_extract_done"
        
        # Show success message for quick extraction
        status_container = st.container()
        status_container.success(f"Quick extraction complete in {st.session_state.quick_extract_time:.2f}s! Document ID: {doc_id}")
        status_container.info(f"Full processing with {st.session_state.selected_processor} will start automatically...")
        
        # Set flag to start full processing in the next rerun
        st.session_state.start_docling_processing = True
        
        # Trigger a rerun to start full processing
        st.rerun()
        return True
    except Exception as e:
        st.error(f"Error during document processing: {str(e)}")
        return False

def main():
    # Initialize session state
    init_session_state()
    
    # App title and description
    st.title("Document Enhancer with AI")
    st.markdown("""
    This app converts documents to Markdown, enhances images with AI-generated descriptions,
    and allows you to ask questions about the document content.
    """)
    
    # Show project info
    st.info(f"Documents and images are stored in: {PROJECT_DIR}")
    
    # Check configuration
    config_valid = check_azure_config() and show_config_status()
    
    # Check if full processing should start or continue
    processing_started = False
    if st.session_state.start_docling_processing or st.session_state.processing_status == "processing_in_progress":
        # If we haven't started processing yet
        if st.session_state.start_docling_processing:
            # Reset the flag
            st.session_state.start_docling_processing = False
            # Set the status
            st.session_state.processing_status = "processing_in_progress"
        # Set a flag to indicate we're running processing this cycle
        processing_started = True
    
    # Create two columns for the main layout
    col1, col2 = st.columns([1, 1])
    
    # Upload & Convert Section (Left Column)
    # Upload & Convert Section (Left Column)
    with col1:
        st.header("Upload & Convert")
        
        # Only show upload controls if we're not currently processing
        if st.session_state.processing_status not in ["processing_in_progress"]:
            uploaded_file = st.file_uploader("Choose a document", type=["pdf", "docx", "doc", "pptx", "ppt"])
            
            # Processor selection
            st.session_state.selected_processor = st.selectbox(
                "Select processing method",
                ["docling", "mistral", "amazon"],
                index=0 if st.session_state.selected_processor == "docling" else 
                    1 if st.session_state.selected_processor == "mistral" else 
                    2 if st.session_state.selected_processor == "amazon" else 0
            )
            
            # Processor-specific options
            if st.session_state.selected_processor == "docling":
                row1, row2 = st.columns(2)
                with row1:
                    process_images = st.checkbox("Add AI descriptions to images", value=True)
                with row2:
                    batch_size = st.slider("Images per API call", min_value=1, max_value=10, value=1, 
                                        disabled=not process_images)
            
            elif st.session_state.selected_processor == "mistral":  # Mistral OCR options
                row1, row2 = st.columns(2)
                with row1:
                    process_images = st.checkbox("Add AI descriptions to images", value=True, 
                                            help="Uses Azure OpenAI to generate detailed descriptions for images")
                with row2:
                    batch_size = st.slider("Images per batch for AI descriptions", min_value=1, max_value=10, value=1, 
                                        disabled=not process_images,
                                        help="Number of images to process for AI descriptions (requires Azure OpenAI)")
                    
                # Add a note about Azure OpenAI requirement for Mistral
                if process_images:
                    if not check_azure_config():
                        st.warning("⚠️ Azure OpenAI is not configured. AI image descriptions will be skipped. Configure Azure OpenAI in your .env file to enable this feature.")
                    else:
                        st.info("✅ Azure OpenAI is configured. AI image descriptions will be added to your images.")
            
            elif st.session_state.selected_processor == "amazon":  # Amazon Textract options
                row1, row2 = st.columns(2)
                with row1:
                    process_images = st.checkbox("Add AI descriptions to images", value=False, 
                                            help="Uses Azure OpenAI to generate detailed descriptions for images (Textract focuses on text)")
                with row2:
                    batch_size = st.slider("Images per batch for AI descriptions", min_value=1, max_value=10, value=1, 
                                        disabled=not process_images,
                                        help="Number of images to process for AI descriptions (requires Azure OpenAI)")
                
                # AWS configuration check
                from processors.amazon_processor import check_textract_config
                aws_configured = check_textract_config()
                
                if not aws_configured:
                    st.warning("⚠️ AWS Textract is not configured. Please configure AWS credentials to use Amazon Textract.")
                    st.info("""
                    **To configure AWS Textract:**
                    1. Install AWS CLI: `pip install awscli`
                    2. Configure credentials: `aws configure`
                    3. Or set environment variables: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION
                    """)
                else:
                    st.success("✅ AWS Textract is configured and ready to use.")
                
                # Add a note about image processing
                if process_images:
                    if not check_azure_config():
                        st.warning("⚠️ Azure OpenAI is not configured. AI image descriptions will be skipped. Configure Azure OpenAI in your .env file to enable this feature.")
                    else:
                        st.info("✅ Azure OpenAI is configured. AI image descriptions will be added to your images.")
            
            # Display current settings
            with st.expander("Current settings"):
                st.write(f"Selected processor: {st.session_state.selected_processor}")
                
                if st.session_state.selected_processor in ["docling", "mistral"]:
                    st.write(f"Azure OpenAI Settings:")
                    st.write(f"- Domain: {config.AZURE_DOMAIN}")
                    st.write(f"- Deployment: {config.AZURE_DEPLOYMENT_NAME}")
                    st.write(f"- API Version: {config.AZURE_API_VERSION}")
                
                if st.session_state.selected_processor == "amazon":
                    st.write(f"AWS Textract Settings:")
                    import boto3
                    try:
                        session = boto3.Session()
                        st.write(f"- Region: {session.region_name}")
                        st.write(f"- Profile: {session.profile_name or 'default'}")
                    except:
                        st.write("- Configuration: Please check AWS credentials")
                
                if st.session_state.selected_processor == "docling":
                    st.write(f"Docling Settings:")
                    st.write(f"- DPI Scale: {config.DOCLING_DPI_SCALE}")
                
                if st.session_state.selected_processor == "mistral":
                    st.write(f"Mistral API Settings:")
                    st.write(f"- API URL: {config.MISTRAL_API_URL}")
            
            # Process button
            config_valid_for_processor = config_valid
            if st.session_state.selected_processor == "amazon":
                config_valid_for_processor = config_valid and check_textract_config()

            process_button = st.button("Process Document", disabled=not uploaded_file or not config_valid_for_processor)
            if process_button and uploaded_file:
                start_document_processing(uploaded_file, process_images, batch_size)
        else:
            # Show processing in progress message
            st.info("Document processing in progress... Please wait.")
    
    # Document Preview Section (Right Column)
    with col2:
        st.header("Document Preview")

            # Debug section to see what's in session state
        with st.expander("Debug Session State", expanded=True):
            st.write("SESSION STATE VARIABLES:")
            st.write(f"processing_status: {st.session_state.get('processing_status')}")
            st.write(f"markdown_content exists: {'markdown_content' in st.session_state}")
            if 'markdown_content' in st.session_state:
                st.write(f"markdown_content type: {type(st.session_state.markdown_content)}")
                st.write(f"markdown_content length: {len(str(st.session_state.markdown_content))}")
            st.write(f"md_file_path exists: {'md_file_path' in st.session_state}")
            if 'md_file_path' in st.session_state:
                st.write(f"md_file_path value: {st.session_state.md_file_path!r}")
            st.write(f"img_dir exists: {'img_dir' in st.session_state}")
            if 'img_dir' in st.session_state:
                st.write(f"img_dir value: {st.session_state.img_dir!r}")
        
        has_quick_extract = st.session_state.quick_extract_content is not None
        has_full_markdown = (
                st.session_state.markdown_content is not None and 
                st.session_state.md_file_path is not None and 
                st.session_state.img_dir is not None
            )
        
        if has_full_markdown or has_quick_extract:
            # Add the radio toggle
            preview_mode = st.radio(
                "Choose document version to preview:",
                options=["Quick Extract", f"{st.session_state.selected_processor.capitalize()}-enhanced"],
                index=1 if has_full_markdown else 0,
                horizontal=True
            )
            
            # Document info
            with st.expander("Document Info", expanded=False):
                st.info(f"Document ID: {st.session_state.doc_id}")
                
                # Add timing information
                if st.session_state.total_processing_time:
                    st.success(f"Total processing time: {st.session_state.total_processing_time:.2f} seconds")
                    timing_cols = st.columns(3)
                    with timing_cols[0]:
                        if st.session_state.quick_extract_time:
                            st.info(f"Quick extract: {st.session_state.quick_extract_time:.2f}s")
                    with timing_cols[1]:
                        if st.session_state.docling_processing_time:
                            st.info(f"Full processing: {st.session_state.docling_processing_time:.2f}s")
                    with timing_cols[2]:
                        if st.session_state.image_processing_time:
                            st.info(f"Image analysis: {st.session_state.image_processing_time:.2f}s")
                
                if has_full_markdown:
                    st.info(f"Markdown file: {st.session_state.md_file_path}")
                    st.info(f"Images directory: {st.session_state.img_dir}")
                    if st.session_state.md_file_path:
                        # Debug info
                        st.info(f"Debug - File path value: {st.session_state.md_file_path!r}")
                        file_path = Path(st.session_state.md_file_path)
                        
                        # Check if it's a valid file
                        if file_path.is_file():
                            try:
                                with open(file_path, "r", encoding="utf-8") as f:
                                    markdown_data = f.read()
                                st.download_button(
                                    label="Download Enhanced Markdown",
                                    data=markdown_data,
                                    file_name=f"{file_path.stem}_enhanced.md",
                                    mime="text/markdown"
                                )
                            except Exception as e:
                                st.error(f"Error reading file: {e}")
                        else:
                            st.error(f"Cannot find file at path: {file_path}")
                elif has_quick_extract:
                    st.info(f"Quick extract file: {st.session_state.quick_extract_file}")
            
            # Show the selected preview version
            if preview_mode.startswith("Quick") and has_quick_extract:
                st.info("Showing quick extract version (preliminary content)")
                display_quick_extract()
            elif has_full_markdown:
                st.success(f"Showing fully processed document with images")
                try:
                    if st.session_state.markdown_content and st.session_state.img_dir:
                        img_dir_path = Path(st.session_state.img_dir) if isinstance(st.session_state.img_dir, str) else st.session_state.img_dir
                        render_markdown_with_images(st.session_state.markdown_content, img_dir_path, max_height=800)
                    else:
                        st.error("Missing markdown content or image directory")
                        st.info("Showing quick extract as fallback")
                        display_quick_extract()
                except Exception as e:
                    st.error(f"Error rendering markdown: {e}")
                    display_quick_extract()
        else:
            st.info("Process a document to see the preview here.")
    
    # Ask Questions Section (below both columns, full width)
    if st.session_state.markdown_content or st.session_state.quick_extract_content:
        st.header("Ask Questions About the Document")
        
        # Display chat history
        for i, chat in enumerate(st.session_state.chat_history):
            st.markdown(
                f"<div style='background-color: #E6F0FF; padding: 10px; border-radius: 10px; margin-bottom: 10px;'><b>You:</b> {chat['question']}</div>",
                unsafe_allow_html=True
            )
            
            # Display both answers if present
            if "quick_answer" in chat and "docling_answer" in chat:
                colA, colB = st.columns(2)
                with colA:
                    st.markdown(
                        f"<div style='background-color: #F0F0F0; padding: 10px; border-radius: 10px; margin-bottom: 20px;'><b>Quick Extract:</b><br>{chat['quick_answer']}</div>",
                        unsafe_allow_html=True
                    )
                with colB:
                    st.markdown(
                        f"<div style='background-color: #F9F9F9; padding: 10px; border-radius: 10px; margin-bottom: 20px;'><b>{st.session_state.selected_processor.capitalize()}-enhanced:</b><br>{chat['docling_answer']}</div>",
                        unsafe_allow_html=True
                    )
        
        # Question input
        question_container = st.container()
        with question_container:
            question = st.text_area("Ask a question about the document", height=100)
            if st.button("Ask Question"):
                if question and config_valid:
                    with st.spinner("Getting answers from both versions..."):
                        # Get the appropriate processor for answering
                        processor = get_processor("quick")
                        quick_answer = processor.answer_question(question)
                        
                        # Get enhanced answer if available
                        docling_answer = "Enhanced version is not ready."
                        if st.session_state.markdown_content:
                            processor = get_processor(st.session_state.selected_processor)
                            docling_answer = processor.answer_question(question)
                        
                        # Save both answers to chat history
                        st.session_state.chat_history.append({
                            "question": question,
                            "quick_answer": quick_answer,
                            "docling_answer": docling_answer,
                        })
                        
                        # Clear input and rerun
                        st.rerun()
                else:
                    st.warning("Please enter a question and ensure the configuration is valid.")
        
        # Add clear chat history button
        if st.session_state.chat_history:
            if st.button("Clear Chat History"):
                st.session_state.chat_history = []
                st.rerun()
    
    # Separate Processing Status Area
    st.markdown("---")
    processing_status_area = st.container()
    
    # If we're supposed to start full processing, do it here
    if processing_started and st.session_state.processing_status == "processing_in_progress":
        with processing_status_area:
            # Get the selected processor
            processor = get_processor(st.session_state.selected_processor)
            
            # Process the document
            processor.process_document(
                file_path=st.session_state.docling_file_path,
                doc_dir=st.session_state.docling_doc_dir,
                images_dir=st.session_state.docling_images_dir,
                process_images=st.session_state.docling_process_images,
                batch_size=st.session_state.docling_batch_size,
                status_area=processing_status_area
            )
            
            # Clear temporary processing variables
            st.session_state.docling_file_path = None
            st.session_state.docling_doc_dir = None
            st.session_state.docling_images_dir = None
            st.session_state.docling_process_images = None
            st.session_state.docling_batch_size = None
            
            # Trigger a rerun to update the UI
            st.rerun()
    
    # Auto-refresh when background processing is happening
    auto_refresh(interval_seconds=5)





if __name__ == "__main__":
    main()