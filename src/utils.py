"""
Utility functions for the RAG Document Q&A system.
"""
import os
import hashlib
import mimetypes
from pathlib import Path
from typing import List, Dict, Any, Optional
import time

def get_file_hash(file_path: str) -> str:
    """Calculate MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def get_file_info(file_path: str) -> Dict[str, Any]:
    """Get comprehensive file information."""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    stat = path.stat()

    return {
        "name": path.name,
        "size": stat.st_size,
        "size_mb": round(stat.st_size / (1024 * 1024), 2),
        "created": stat.st_ctime,
        "modified": stat.st_mtime,
        "extension": path.suffix.lower(),
        "mime_type": mimetypes.guess_type(str(path))[0],
        "hash": get_file_hash(file_path)
    }


def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    if size_bytes == 0:
        return "0 B"

    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1

    return f"{size_bytes:.1f} {size_names[i]}"


def validate_document_format(file_path: str, supported_formats: List[str]) -> bool:
    """Validate if document format is supported."""
    extension = Path(file_path).suffix.lower().lstrip('.')
    return extension in supported_formats


def clean_text(text: str) -> str:
    """Clean and normalize text content."""
    if not text:
        return ""

    # Remove excessive whitespace
    text = ' '.join(text.split())

    # Remove control characters but keep newlines and tabs
    cleaned = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')

    return cleaned.strip()


def split_text_by_sentences(text: str, max_chunk_size: int = 1000) -> List[str]:
    """Split text into chunks by sentences, respecting max size."""
    import re

    # Simple sentence splitting (can be improved with spaCy for better accuracy)
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chunk_size:
            current_chunk += sentence + " "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " "

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """Extract keywords from text using simple frequency analysis."""
    import re
    from collections import Counter

    # Simple keyword extraction (can be improved with NLP libraries)
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())

    # Filter out common stop words
    stop_words = {
        'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
        'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before',
        'after', 'above', 'below', 'between', 'among', 'this', 'that', 'these',
        'those', 'is', 'are', 'was', 'were', 'been', 'be', 'have', 'has', 'had',
        'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
        'must', 'can', 'shall', 'a', 'an', 'as', 'if', 'or', 'because', 'while',
        'when', 'where', 'how', 'what', 'which', 'who', 'whom', 'whose', 'why'
    }

    filtered_words = [word for word in words if word not in stop_words and len(word) > 3]

    # Get most common words
    word_freq = Counter(filtered_words)
    keywords = [word for word, count in word_freq.most_common(max_keywords)]

    return keywords


def time_function(func):
    """Decorator to time function execution."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Function {func.__name__} took {execution_time:.2f} seconds")
        return result
    return wrapper


def create_directory_structure():
    """Create necessary directory structure for the application."""
    from src.config import settings

    directories = [
        settings.documents_dir,
        settings.index_dir,
        settings.logs_dir,
        "temp",
        "backups"
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


def backup_index(index_name: str, backup_dir: str = "backups") -> str:
    """Create a backup of an index."""
    from src.config import settings
    import shutil
    import datetime

    source_path = Path(settings.index_dir) / index_name

    if not source_path.exists():
        raise FileNotFoundError(f"Index not found: {index_name}")

    backup_dir = Path(backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"{index_name}_backup_{timestamp}"
    backup_path = backup_dir / backup_name

    shutil.copytree(source_path, backup_path)

    return str(backup_path)


def restore_index(backup_path: str, index_name: str) -> None:
    """Restore an index from backup."""
    from src.config import settings
    import shutil

    source_path = Path(backup_path)
    target_path = Path(settings.index_dir) / index_name

    if not source_path.exists():
        raise FileNotFoundError(f"Backup not found: {backup_path}")

    if target_path.exists():
        shutil.rmtree(target_path)

    shutil.copytree(source_path, target_path)


def get_system_info() -> Dict[str, Any]:
    """Get system information for debugging."""
    import platform
    import psutil

    try:
        cpu_count = psutil.cpu_count()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        return {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_count": cpu_count,
            "memory_total_gb": round(memory.total / (1024**3), 2),
            "memory_available_gb": round(memory.available / (1024**3), 2),
            "disk_total_gb": round(disk.total / (1024**3), 2),
            "disk_free_gb": round(disk.free / (1024**3), 2)
        }
    except ImportError:
        # Fallback if psutil is not available
        return {
            "platform": platform.platform(),
            "python_version": platform.python_version()
        }


def estimate_processing_time(file_size_mb: float, avg_speed_mb_per_sec: float = 1.0) -> float:
    """Estimate processing time for a file based on size."""
    return file_size_mb / avg_speed_mb_per_sec


def validate_openai_api_key(api_key: str) -> bool:
    """Validate OpenAI API key format."""
    return api_key.startswith('sk-') and len(api_key) > 20


def chunk_list(lst: List, chunk_size: int) -> List[List]:
    """Split a list into chunks of specified size."""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def merge_metadata(base_metadata: Dict[str, Any], additional_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Merge metadata dictionaries with conflict resolution."""
    merged = base_metadata.copy()

    for key, value in additional_metadata.items():
        if key in merged:
            # Handle conflicts by prefixing with 'additional_'
            merged[f"additional_{key}"] = value
        else:
            merged[key] = value

    return merged


class ProgressTracker:
    """Simple progress tracker for long-running operations."""

    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = time.time()

    def update(self, increment: int = 1):
        """Update progress."""
        self.current += increment
        self._print_progress()

    def _print_progress(self):
        """Print progress bar."""
        if self.total == 0:
            return

        percent = (self.current / self.total) * 100
        elapsed = time.time() - self.start_time

        if self.current > 0:
            eta = (elapsed / self.current) * (self.total - self.current)
            eta_str = f"ETA: {eta:.1f}s"
        else:
            eta_str = "ETA: --"

        bar_length = 30
        filled_length = int(bar_length * self.current / self.total)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)

        print(f'\r{self.description}: |{bar}| {percent:.1f}% ({self.current}/{self.total}) {eta_str}', end='')

        if self.current >= self.total:
            print()  # New line when complete
