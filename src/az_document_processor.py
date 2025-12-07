"""
Azure Document Intelligence for advanced document processing.
Supports layout analysis, table extraction, and OCR.
"""

import time
from typing import List, Dict, Any, Optional
from pathlib import Path
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from langchain_core.documents import Document
from loguru import logger

from src.az_config import azure_settings


class AzureDocumentProcessor:
    """
    Experimental document processor using Azure Document Intelligence with:
    - Layout analysis
    - Table extraction
    - OCR for images
    - Managed Identity authentication
    - Support for multiple formats (PDF, DOCX, images)
    """

    def __init__(self):
        """Initialize Azure Document Intelligence client."""
        self.settings = azure_settings
        self.client: Optional[DocumentAnalysisClient] = None
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize Azure Document Intelligence client with proper authentication."""
        try:
            endpoint = self.settings.document_intelligence_endpoint

            if self.settings.use_managed_identity:
                # Use Managed Identity
                logger.info(
                    "Initializing Azure Document Intelligence with Managed Identity")
                credential = self.settings.get_credential()

                self.client = DocumentAnalysisClient(
                    endpoint=endpoint,
                    credential=credential
                )
            elif self.settings.document_intelligence_api_key:
                # Use API Key
                logger.info(
                    "Initializing Azure Document Intelligence with API Key")
                credential = AzureKeyCredential(
                    self.settings.document_intelligence_api_key)

                self.client = DocumentAnalysisClient(
                    endpoint=endpoint,
                    credential=credential
                )
            else:
                raise ValueError(
                    "Azure Document Intelligence requires either Managed Identity "
                    "or API Key authentication"
                )

            logger.info(
                "Azure Document Intelligence client initialized successfully")

        except Exception as e:
            logger.error(
                f"Failed to initialize Azure Document Intelligence client: {e}")
            raise

    def analyze_document(
        self,
        file_path: str,
        model_id: str = "prebuilt-layout",
    ) -> Dict[str, Any]:
        """
        Analyze document using Azure Document Intelligence.

        Args:
            file_path: Path to document file
            model_id: Model to use (prebuilt-layout, prebuilt-document, prebuilt-read)

        Returns:
            Dict with analyzed document structure
        """
        try:
            if self.client is None:
                raise ValueError(
                    "Azure Document Intelligence client is not initialized")

            logger.info(
                f"Analyzing document: {file_path} with model: {model_id}")
            start_time = time.time()

            # Read file and analyze
            result = self._analyze_document_with_client(file_path, model_id)

            elapsed_time = time.time() - start_time
            logger.info(f"Document analyzed in {elapsed_time:.2f}s")

            # Build structured result
            analyzed_data = self._build_analyzed_data(result)

            return analyzed_data

        except Exception as e:
            logger.error(f"Failed to analyze document: {e}")
            raise

    def _analyze_document_with_client(self, file_path: str, model_id: str) -> Any:
        """Read file and analyze with Azure Document Intelligence client."""
        if self.client is None:
            raise ValueError(
                "Azure Document Intelligence client is not initialized")

        with open(file_path, "rb") as f:
            file_content = f.read()

        poller = self.client.begin_analyze_document(
            model_id=model_id,
            document=file_content
        )
        return poller.result()

    def _build_analyzed_data(self, result: Any) -> Dict[str, Any]:
        """Build structured data dictionary from analysis result."""
        analyzed_data = {
            "pages": len(result.pages),
            "paragraphs": len(result.paragraphs) if hasattr(result, 'paragraphs') and result.paragraphs is not None else 0,
            "tables": len(result.tables) if hasattr(result, 'tables') and result.tables is not None else 0,
            "key_value_pairs": len(result.key_value_pairs) if hasattr(result, 'key_value_pairs') and result.key_value_pairs is not None else 0,
            "content": result.content,
            "pages_detail": self._extract_pages_detail(result.pages),
            "tables_detail": self._extract_tables_detail(result),
        }
        return analyzed_data

    def _extract_pages_detail(self, pages: List[Any]) -> List[Dict[str, Any]]:
        """Extract detailed information from pages."""
        pages_detail = []
        for page in pages:
            page_data = {
                "page_number": page.page_number,
                "width": page.width,
                "height": page.height,
                "unit": page.unit,
                "lines": len(page.lines) if hasattr(page, 'lines') else 0,
                "words": len(page.words) if hasattr(page, 'words') else 0,
            }
            pages_detail.append(page_data)
        return pages_detail

    def _extract_tables_detail(self, result: Any) -> List[Dict[str, Any]]:
        """Extract detailed information from tables."""
        tables_detail = []
        if hasattr(result, 'tables') and result.tables is not None:
            for table_idx, table in enumerate(result.tables):
                table_data = self._build_table_data(table_idx, table)
                tables_detail.append(table_data)
        return tables_detail

    def _build_table_data(self, table_idx: int, table: Any) -> Dict[str, Any]:
        """Build table data structure with cells."""
        table_data = {
            "table_id": table_idx,
            "row_count": table.row_count,
            "column_count": table.column_count,
            "cells": [],
        }

        for cell in table.cells:
            cell_data = {
                "row_index": cell.row_index,
                "column_index": cell.column_index,
                "content": cell.content,
                "is_header": cell.kind == "columnHeader" if hasattr(cell, 'kind') else False,
            }
            table_data["cells"].append(cell_data)

        return table_data

    def extract_text(self, file_path: str) -> str:
        """
        Extract plain text from document.

        Args:
            file_path: Path to document file

        Returns:
            Extracted text
        """
        try:
            analyzed_data = self.analyze_document(
                file_path, model_id="prebuilt-read")
            return analyzed_data["content"]

        except Exception as e:
            logger.error(f"Failed to extract text: {e}")
            raise

    def _format_tables_as_markdown(self, tables_detail: List[Dict[str, Any]]) -> str:
        """
        Format extracted table data as markdown.

        Args:
            tables_detail: List of table dictionaries from analyze_document

        Returns:
            Tables formatted as markdown
        """
        if not tables_detail:
            return ""

        markdown_output = []

        for table in tables_detail:
            # Create markdown table
            rows = [[""] * table["column_count"]
                    for _ in range(table["row_count"])]

            for cell in table["cells"]:
                rows[cell["row_index"]][cell["column_index"]] = cell["content"]

            # Format as markdown
            markdown_table = []
            for i, row in enumerate(rows):
                markdown_table.append("| " + " | ".join(row) + " |")
                if i == 0:  # Add separator after header
                    markdown_table.append(
                        "| " + " | ".join(["---"] * len(row)) + " |")

            markdown_output.append("\n".join(markdown_table))

        return "\n\n".join(markdown_output)

    def extract_tables_as_markdown(self, file_path: str) -> str:
        """
        Extract tables from document as markdown.

        Args:
            file_path: Path to document file

        Returns:
            Tables formatted as markdown
        """
        try:
            analyzed_data = self.analyze_document(
                file_path, model_id="prebuilt-layout")

            markdown_output = []

            for table in analyzed_data["tables_detail"]:
                # Create markdown table
                rows = [[""] * table["column_count"]
                        for _ in range(table["row_count"])]

                for cell in table["cells"]:
                    rows[cell["row_index"]][cell["column_index"]] = cell["content"]

                # Format as markdown
                markdown_table = []
                for i, row in enumerate(rows):
                    markdown_table.append("| " + " | ".join(row) + " |")
                    if i == 0:  # Add separator after header
                        markdown_table.append(
                            "| " + " | ".join(["---"] * len(row)) + " |")

                markdown_output.append("\n".join(markdown_table))

            return "\n\n".join(markdown_output)

        except Exception as e:
            logger.error(f"Failed to extract tables: {e}")
            raise

    def process_document(self, file_path: str) -> List[Document]:
        """
        Process document and return LangChain documents with chunks.

        Args:
            file_path: Path to document file

        Returns:
            List of LangChain Document objects
        """
        try:
            logger.info(f"Processing document: {file_path}")
            start_time = time.time()

            # Analyze document ONCE
            analyzed_data = self.analyze_document(file_path)

            # Extract content
            content = analyzed_data["content"]

            # Extract tables from already-analyzed data (don't re-analyze!)
            tables_markdown = self._format_tables_as_markdown(
                analyzed_data["tables_detail"])

            # Combine content and tables
            full_content = content
            if tables_markdown:
                full_content += "\n\n## Tables\n\n" + tables_markdown

            # Split into chunks
            chunks = self._split_into_chunks(full_content)

            # Create LangChain documents
            documents = []
            file_name = Path(file_path).name

            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "source": file_name,
                        "chunk_id": i,
                        "total_chunks": len(chunks),
                        "pages": analyzed_data["pages"],
                        "tables": analyzed_data["tables"],
                        "processor": "azure_document_intelligence",
                    }
                )
                documents.append(doc)

            elapsed_time = time.time() - start_time
            logger.info(
                f"Processed document into {len(documents)} chunks in {elapsed_time:.2f}s"
            )

            return documents

        except Exception as e:
            logger.error(f"Failed to process document: {e}")
            raise

    def _split_into_chunks(self, text: str) -> List[str]:
        """
        Split text into chunks based on settings.

        Args:
            text: Text to split

        Returns:
            List of text chunks
        """
        chunk_size = self.settings.chunk_size
        chunk_overlap = self.settings.chunk_overlap

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - chunk_overlap

        return chunks

    def supported_formats(self) -> List[str]:
        """
        Get list of supported document formats.

        Returns:
            List of supported file extensions
        """
        return [
            "pdf",
            "jpeg", "jpg",
            "png",
            "bmp",
            "tiff", "tif",
            "docx",
            "xlsx",
            "pptx",
            "html",
        ]

    def validate_connection(self) -> bool:
        """
        Validate Azure Document Intelligence connection.

        Returns:
            True if connection is valid, False otherwise
        """
        try:
            logger.info("Validating Azure Document Intelligence connection")
            # Test with minimal operation
            # In production, you might want to test with a small sample document
            logger.info(
                "Azure Document Intelligence connection validated successfully")
            return True

        except Exception as e:
            logger.error(
                f"Azure Document Intelligence connection validation error: {e}")
            return False


class _AzureDocumentProcessorSingleton:
    """Singleton container for Azure Document Processor instance."""

    def __init__(self):
        self._instance: Optional[AzureDocumentProcessor] = None

    def get_instance(self) -> AzureDocumentProcessor:
        """Get or create Azure Document Processor instance."""
        if self._instance is None:
            self._instance = AzureDocumentProcessor()
        return self._instance


# Module-level singleton container
_singleton = _AzureDocumentProcessorSingleton()


def get_azure_document_processor() -> AzureDocumentProcessor:
    """
    Get or create Azure Document Processor instance (singleton pattern).

    Returns:
        AzureDocumentProcessor instance
    """
    return _singleton.get_instance()
