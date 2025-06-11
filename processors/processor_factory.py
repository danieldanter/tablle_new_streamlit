"""
Factory for creating document processors.
Provides a single function to get the appropriate processor instance.
"""

from processors.quick_processor import QuickProcessor
from processors.docling_processor import DoclingProcessor
from processors.mistral_processor import MistralProcessor
from processors.amazon_processor import AmazonProcessor

# Singleton instances of each processor
_quick_processor = None
_docling_processor = None
_mistral_processor = None
_amazon_processor = None

def get_processor(processor_type):
    """
    Get the appropriate processor instance based on the type.
    Uses singleton pattern to avoid creating multiple instances.
    
    Args:
        processor_type (str): Type of processor to get ('quick', 'docling', 'mistral', 'amazon')
        
    Returns:
        BaseProcessor: The appropriate processor instance
    """
    global _quick_processor, _docling_processor, _mistral_processor, _amazon_processor
    
    processor_type = processor_type.lower()
    
    if processor_type == 'quick':
        if _quick_processor is None:
            _quick_processor = QuickProcessor()
        return _quick_processor
    
    elif processor_type == 'docling':
        if _docling_processor is None:
            _docling_processor = DoclingProcessor()
        return _docling_processor
    
    elif processor_type == 'mistral':
        if _mistral_processor is None:
            _mistral_processor = MistralProcessor()
        return _mistral_processor
    
    elif processor_type == 'amazon':
        if _amazon_processor is None:
            _amazon_processor = AmazonProcessor()
        return _amazon_processor
    
    else:
        raise ValueError(f"Unknown processor type: {processor_type}. Available types: quick, docling, mistral, amazon")