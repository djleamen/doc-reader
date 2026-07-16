"""
Unit tests for src.utils utility functions.

Covers the pure, dependency-free helpers: file hashing/metadata, text
processing, validation, and small data helpers. Functions that depend on
application settings (directory/backup helpers) are intentionally out of scope
to keep these tests fast and isolated.

Written by DJ Leamen (2025-2026)
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import (  # noqa: E402
    ProgressTracker,
    chunk_list,
    clean_text,
    extract_keywords,
    format_file_size,
    get_file_hash,
    get_file_info,
    merge_metadata,
    split_text_by_sentences,
    validate_document_format,
    validate_openai_api_key,
)


def test_get_file_hash_matches_known_md5(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_bytes(b"hello world")
    # Reference MD5 of "hello world".
    assert get_file_hash(str(file_path)) == "5eb63bbbe01eeed093cb22bb8f5acdc3"


def test_get_file_hash_is_stable_and_content_sensitive(tmp_path):
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_bytes(b"same")
    b.write_bytes(b"same")
    assert get_file_hash(str(a)) == get_file_hash(str(b))
    b.write_bytes(b"different")
    assert get_file_hash(str(a)) != get_file_hash(str(b))


def test_get_file_info_reports_metadata(tmp_path):
    file_path = tmp_path / "doc.txt"
    file_path.write_bytes(b"12345")
    info = get_file_info(str(file_path))
    assert info["name"] == "doc.txt"
    assert info["size"] == 5
    assert info["extension"] == ".txt"
    assert info["hash"] == get_file_hash(str(file_path))


def test_get_file_info_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        get_file_info(str(tmp_path / "nope.txt"))


@pytest.mark.parametrize(
    "size,expected",
    [
        (0, "0 B"),
        (512, "512.0 B"),
        (1024, "1.0 KB"),
        (1024 * 1024, "1.0 MB"),
        (1536 * 1024, "1.5 MB"),
    ],
)
def test_format_file_size(size, expected):
    assert format_file_size(size) == expected


def test_validate_document_format():
    assert validate_document_format("report.PDF", ["pdf", "txt"]) is True
    assert validate_document_format("image.png", ["pdf", "txt"]) is False


def test_clean_text_normalizes_whitespace_and_control_chars():
    assert clean_text("  hello   world  ") == "hello world"
    # split()/join collapses all runs of whitespace (incl. tabs/newlines) to a
    # single space.
    assert clean_text("keep\tnewline\nchars") == "keep newline chars"
    assert clean_text("bad\x00char") == "badchar"
    assert clean_text("") == ""


def test_split_text_by_sentences_respects_max_size():
    text = "One. Two. Three. Four."
    chunks = split_text_by_sentences(text, max_chunk_size=10)
    assert len(chunks) > 1
    assert all(len(chunk) <= 12 for chunk in chunks)
    # No content is lost across the chunk boundaries.
    assert "One." in " ".join(chunks)
    assert "Four." in " ".join(chunks)


def test_extract_keywords_filters_stopwords_and_limits():
    text = "Python python testing testing testing the and or code code"
    keywords = extract_keywords(text, max_keywords=2)
    assert "the" not in keywords
    assert len(keywords) <= 2
    assert "testing" in keywords


def test_validate_openai_api_key():
    assert validate_openai_api_key("sk-" + "a" * 30) is True
    assert validate_openai_api_key("sk-short") is False
    assert validate_openai_api_key("nope-" + "a" * 30) is False


def test_chunk_list():
    assert chunk_list([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]
    assert chunk_list([], 3) == []


def test_merge_metadata_prefixes_conflicts():
    merged = merge_metadata({"a": 1, "b": 2}, {"b": 3, "c": 4})
    assert merged["a"] == 1
    assert merged["b"] == 2  # original value preserved
    assert merged["additional_b"] == 3  # conflicting value prefixed
    assert merged["c"] == 4


def test_progress_tracker_counts_updates():
    tracker = ProgressTracker(total=3, description="Test")
    tracker.update()
    tracker.update(2)
    assert tracker.current == 3


def test_progress_tracker_zero_total_does_not_crash():
    tracker = ProgressTracker(total=0)
    tracker.update()  # should be a no-op, not a ZeroDivisionError
    assert tracker.current == 1
