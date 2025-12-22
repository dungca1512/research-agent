"""Paper parsing tool for extracting content from PDFs."""

import io
import httpx
from typing import Optional
from langchain_core.tools import tool
from pypdf import PdfReader


def download_pdf(url: str, timeout: float = 30.0) -> Optional[bytes]:
    """
    Download a PDF from URL.
    
    Args:
        url: URL to the PDF file
        timeout: Request timeout in seconds
        
    Returns:
        PDF content as bytes or None if failed
    """
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            response = client.get(url)
            response.raise_for_status()
            return response.content
    except Exception as e:
        print(f"Failed to download PDF from {url}: {e}")
        return None


def extract_text_from_pdf(pdf_content: bytes, max_pages: int = 10) -> str:
    """
    Extract text content from PDF bytes.
    
    Args:
        pdf_content: PDF file content as bytes
        max_pages: Maximum number of pages to extract
        
    Returns:
        Extracted text content
    """
    try:
        pdf_file = io.BytesIO(pdf_content)
        reader = PdfReader(pdf_file)
        
        text_parts = []
        pages_to_read = min(len(reader.pages), max_pages)
        
        for i in range(pages_to_read):
            page = reader.pages[i]
            text = page.extract_text()
            if text:
                text_parts.append(f"--- Page {i + 1} ---\n{text}")
        
        return "\n\n".join(text_parts)
        
    except Exception as e:
        print(f"Failed to extract text from PDF: {e}")
        return ""


def parse_paper(url: str, max_pages: int = 10) -> dict:
    """
    Download and parse a paper from URL.
    
    Args:
        url: URL to the paper (PDF)
        max_pages: Maximum pages to extract
        
    Returns:
        Dictionary with parsed content
    """
    pdf_content = download_pdf(url)
    
    if not pdf_content:
        return {
            "success": False,
            "error": "Failed to download PDF",
            "url": url,
            "content": ""
        }
    
    text = extract_text_from_pdf(pdf_content, max_pages)
    
    if not text:
        return {
            "success": False,
            "error": "Failed to extract text from PDF",
            "url": url,
            "content": ""
        }
    
    return {
        "success": True,
        "url": url,
        "content": text,
        "char_count": len(text),
        "pages_extracted": min(max_pages, text.count("--- Page"))
    }


@tool
def paper_parser_tool(pdf_url: str) -> str:
    """
    Download and extract text content from a research paper PDF.
    Use this tool when you need to read the full content of a paper, not just its abstract.
    
    Args:
        pdf_url: Direct URL to the PDF file (e.g., from arXiv)
        
    Returns:
        The extracted text content from the paper
    """
    result = parse_paper(pdf_url)
    
    if not result["success"]:
        return f"Failed to parse paper: {result['error']}\nURL: {pdf_url}"
    
    output = f"## Paper Content\n"
    output += f"**Source:** {result['url']}\n"
    output += f"**Pages Extracted:** {result['pages_extracted']}\n\n"
    output += "---\n\n"
    output += result["content"][:8000]  # Limit output size
    
    if len(result["content"]) > 8000:
        output += "\n\n[... Content truncated for brevity ...]"
    
    return output
