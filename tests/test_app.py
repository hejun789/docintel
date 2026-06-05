"""Tests for allowed_file — the upload extension filter (a security boundary)."""
import pytest

from app import allowed_file


@pytest.mark.parametrize("filename", [
    "report.pdf",
    "notes.docx",
    "readme.md",
    "data.txt",
    "REPORT.PDF",   # case-insensitive
    "My.Report.pdf",  # dots in name
])
def test_allowed_extensions(filename):
    assert allowed_file(filename) is True


@pytest.mark.parametrize("filename", [
    "malware.exe",
    "script.sh",
    "image.png",
    "archive.zip",
    "noextension",
    ".env",
])
def test_disallowed_extensions(filename):
    assert allowed_file(filename) is False
