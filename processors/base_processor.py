"""
Base processor interface for document processing.
All document processors should inherit from this base class.
"""

import time
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class ProcessingResult:
    """Container for document processing results"""
    content: str  # The processed content
    output_file: Path  # Path to the output file
    images_dir: Path  # Path to the images directory
    processing_time: float  # Time taken to process the document
    image_processing_time: float = 0.0  # Time taken for image processing (if applicable)
    error: str = None  # Error message (if any)

class BaseProcessor(ABC):
    """
    Abstract base class for document processors.
    Defines the interface that all document processors must implement.
    """
    
    @abstractmethod
    def process_document(self, file_path, doc_dir, images_dir, process_images=True, batch_size=1, status_area=None):
        """
        Process a document and return the results.
        
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
        pass
    
    @abstractmethod
    def answer_question(self, question, context=None):
        """
        Answer a question about the document content.
        
        Args:
            question (str): The question to answer
            context (str, optional): Additional context for the question
            
        Returns:
            str: The answer to the question
        """
        pass
    
    def _start_timer(self):
        """Start a processing timer"""
        return time.time()
    
    def _stop_timer(self, start_time):
        """
        Stop a processing timer and return elapsed time.
        
        Args:
            start_time (float): The start time from _start_timer()
            
        Returns:
            float: Elapsed time in seconds
        """
        return time.time() - start_time