"""
Test suite for the RAG Document Q&A system.
"""
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch

from src.document_processor import DocumentProcessor
from src.config import settings


class TestDocumentProcessor:
    """Test cases for DocumentProcessor class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.processor = DocumentProcessor()
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def test_pypdf_migration_fix(self):
        """Test that pypdf is used instead of PyPDF2 to fix infinite loop vulnerability."""
        # This test verifies the fix for the infinite loop issue when comments aren't followed by characters
        import sys
        import src.document_processor as doc_proc
        
        # Verify that the document processor uses pypdf
        assert hasattr(doc_proc, 'pypdf'), "DocumentProcessor should import pypdf"
        
        # Verify that PyPDF2 is not being used 
        import inspect
        source_code = inspect.getsource(doc_proc)
        assert 'import pypdf' in source_code, "DocumentProcessor should import pypdf"
        assert 'import PyPDF2' not in source_code, "DocumentProcessor should not import PyPDF2"
        assert 'pypdf.PdfReader' in source_code, "DocumentProcessor should use pypdf.PdfReader"
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_load_text_document(self):
        """Test loading plain text documents."""
        # Create test file
        test_file = self.temp_dir / "test.txt"
        test_content = "This is a test document with some content."
        test_file.write_text(test_content)
        
        # Load document
        result = self.processor.load_document(str(test_file))
        
        assert result == test_content
    
    def test_load_markdown_document(self):
        """Test loading markdown documents."""
        test_file = self.temp_dir / "test.md"
        test_content = "# Test Document\n\nThis is a **markdown** document."
        test_file.write_text(test_content)
        
        result = self.processor.load_document(str(test_file))
        
        assert result == test_content
    
    def test_unsupported_format(self):
        """Test handling of unsupported file formats."""
        test_file = self.temp_dir / "test.xyz"
        test_file.write_text("content")
        
        with pytest.raises(ValueError, match="Unsupported format"):
            self.processor.load_document(str(test_file))
    
    def test_file_not_found(self):
        """Test handling of non-existent files."""
        with pytest.raises(FileNotFoundError):
            self.processor.load_document("nonexistent_file.txt")
    
    def test_large_file_rejection(self):
        """Test rejection of files that are too large."""
        # Mock settings to have a very small max file size
        with patch.object(settings, 'max_document_size_mb', 0.000001):  # Very small limit
            test_file = self.temp_dir / "large.txt"
            test_file.write_text("This content will exceed the tiny limit")
            
            with pytest.raises(ValueError, match="Document too large"):
                self.processor.load_document(str(test_file))
    
    def test_chunk_document(self):
        """Test document chunking functionality."""
        text = "This is a long document that should be split into multiple chunks. " * 50
        
        documents = self.processor.chunk_document(text)
        
        assert len(documents) > 1
        assert all(hasattr(doc, 'page_content') for doc in documents)
        assert all(hasattr(doc, 'metadata') for doc in documents)
        
        # Check chunk metadata
        for i, doc in enumerate(documents):
            assert doc.metadata['chunk_id'] == i
            assert 'chunk_size' in doc.metadata
    
    def test_process_document_integration(self):
        """Test the complete document processing pipeline."""
        test_file = self.temp_dir / "integration_test.txt"
        test_content = "This is a test document for integration testing. " * 100
        test_file.write_text(test_content)
        
        documents = self.processor.process_document(str(test_file))
        
        assert len(documents) > 0
        assert all(hasattr(doc, 'page_content') for doc in documents)
        assert all(hasattr(doc, 'metadata') for doc in documents)
        
        # Check metadata
        first_doc = documents[0]
        assert first_doc.metadata['source'] == str(test_file)
        assert first_doc.metadata['filename'] == test_file.name
        assert first_doc.metadata['file_type'] == 'txt'
        assert 'word_count' in first_doc.metadata


class TestRAGEngine:
    """Test cases for RAGEngine class (mocked tests)."""
    
    @patch('src.rag_engine.ChatOpenAI')
    @patch('src.rag_engine.DocumentIndex')
    def test_rag_engine_initialization(self, mock_index, mock_llm):
        """Test RAG engine initialization."""
        from src.rag_engine import RAGEngine
        
        engine = RAGEngine("test_index")
        
        assert engine.index_name == "test_index"
        mock_index.assert_called_once_with("test_index")
        mock_llm.assert_called_once()
    
    @patch('src.rag_engine.ChatOpenAI')
    @patch('src.rag_engine.DocumentIndex')
    def test_query_with_no_documents(self, mock_index, mock_llm):
        """Test querying when no documents are found."""
        from src.rag_engine import RAGEngine
        
        # Mock empty search results
        mock_index_instance = mock_index.return_value
        mock_index_instance.search.return_value = []
        
        engine = RAGEngine("test_index")
        result = engine.query("test question")
        
        assert "couldn't find any relevant information" in result.answer.lower()
        assert len(result.source_documents) == 0


class TestUtils:
    """Test cases for utility functions."""
    
    def test_format_file_size(self):
        """Test file size formatting."""
        from src.utils import format_file_size
        
        assert format_file_size(0) == "0 B"
        assert format_file_size(1024) == "1.0 KB"
        assert format_file_size(1024 * 1024) == "1.0 MB"
        assert format_file_size(1536) == "1.5 KB"
    
    def test_validate_document_format(self):
        """Test document format validation."""
        from src.utils import validate_document_format
        
        supported = ['pdf', 'txt', 'docx']
        
        assert validate_document_format("test.pdf", supported) == True
        assert validate_document_format("test.txt", supported) == True
        assert validate_document_format("test.xyz", supported) == False
        assert validate_document_format("test.PDF", supported) == True  # Case insensitive
    
    def test_clean_text(self):
        """Test text cleaning function."""
        from src.utils import clean_text
        
        dirty_text = "  This   has    excessive   whitespace  \n\n  "
        clean = clean_text(dirty_text)
        
        assert clean == "This has excessive whitespace"
        
        # Test with empty input
        assert clean_text("") == ""
        assert clean_text(None) == ""
    
    def test_extract_keywords(self):
        """Test keyword extraction."""
        from src.utils import extract_keywords
        
        text = "This is a sample document about machine learning and artificial intelligence. " \
               "Machine learning is very important in artificial intelligence applications."
        
        keywords = extract_keywords(text, max_keywords=5)
        
        assert len(keywords) <= 5
        assert 'machine' in keywords
        assert 'learning' in keywords
        assert 'artificial' in keywords
        assert 'intelligence' in keywords
    
    def test_chunk_list(self):
        """Test list chunking function."""
        from src.utils import chunk_list
        
        test_list = list(range(10))
        chunks = chunk_list(test_list, 3)
        
        assert len(chunks) == 4  # [0,1,2], [3,4,5], [6,7,8], [9]
        assert chunks[0] == [0, 1, 2]
        assert chunks[1] == [3, 4, 5]
        assert chunks[2] == [6, 7, 8]
        assert chunks[3] == [9]
    
    def test_validate_openai_api_key(self):
        """Test OpenAI API key validation."""
        from src.utils import validate_openai_api_key
        
        valid_key = "sk-1234567890abcdef1234567890abcdef"
        invalid_key1 = "invalid_key"
        invalid_key2 = "sk-short"
        
        assert validate_openai_api_key(valid_key) == True
        assert validate_openai_api_key(invalid_key1) == False
        assert validate_openai_api_key(invalid_key2) == False


class TestProgressTracker:
    """Test cases for ProgressTracker utility."""
    
    def test_progress_tracker_initialization(self):
        """Test progress tracker initialization."""
        from src.utils import ProgressTracker
        
        tracker = ProgressTracker(100, "Test Progress")
        
        assert tracker.total == 100
        assert tracker.current == 0
        assert tracker.description == "Test Progress"
    
    def test_progress_tracker_update(self):
        """Test progress tracker updates."""
        from src.utils import ProgressTracker
        
        tracker = ProgressTracker(10, "Test")
        
        tracker.update(5)
        assert tracker.current == 5
        
        tracker.update(3)
        assert tracker.current == 8
        
        tracker.update(2)
        assert tracker.current == 10


@pytest.fixture
def sample_documents():
    """Fixture providing sample documents for testing."""
    return [
        {
            "filename": "doc1.txt",
            "content": "This is the first document about artificial intelligence and machine learning."
        },
        {
            "filename": "doc2.txt", 
            "content": "This is the second document discussing natural language processing and deep learning."
        },
        {
            "filename": "doc3.txt",
            "content": "This document covers computer vision and neural networks in detail."
        }
    ]


@pytest.fixture
def temp_documents_dir(sample_documents):
    """Fixture that creates temporary documents for testing."""
    temp_dir = Path(tempfile.mkdtemp())
    
    # Create sample documents
    for doc in sample_documents:
        doc_path = temp_dir / doc["filename"]
        doc_path.write_text(doc["content"])
    
    yield temp_dir
    
    # Cleanup
    shutil.rmtree(temp_dir)


class TestIntegration:
    """Integration tests for the complete system."""
    
    def test_document_processing_pipeline(self, temp_documents_dir):
        """Test the complete document processing pipeline."""
        processor = DocumentProcessor()
        
        # Process all documents in the temporary directory
        for doc_file in temp_documents_dir.glob("*.txt"):
            documents = processor.process_document(str(doc_file))
            
            assert len(documents) > 0
            assert all(doc.page_content for doc in documents)
            assert all(doc.metadata for doc in documents)
            
            # Verify metadata
            for doc in documents:
                assert doc.metadata['source'] == str(doc_file)
                assert doc.metadata['filename'] == doc_file.name
                assert doc.metadata['file_type'] == 'txt'


if __name__ == "__main__":
    pytest.main([__file__])
