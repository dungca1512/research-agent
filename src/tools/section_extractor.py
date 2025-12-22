"""Advanced PDF section extraction tools."""

import re
from typing import Optional
from src.tools.paper_parser import download_pdf, extract_text_from_pdf


def extract_section(
    content: str,
    section_name: str,
    next_sections: list[str] = None
) -> Optional[str]:
    """
    Extract a specific section from paper content.
    
    Args:
        content: Full paper text
        section_name: Section to extract (e.g., "Abstract", "Methodology")
        next_sections: Possible section headers that end this section
        
    Returns:
        Section content or None if not found
    """
    if next_sections is None:
        next_sections = [
            "Abstract", "Introduction", "Background", "Related Work",
            "Methodology", "Methods", "Method", "Approach",
            "Experiments", "Experimental", "Results", "Evaluation",
            "Discussion", "Conclusion", "Conclusions",
            "References", "Bibliography", "Acknowledgments"
        ]
    
    # Remove the current section from next_sections
    next_sections = [s for s in next_sections if s.lower() != section_name.lower()]
    
    # Build regex pattern
    section_pattern = rf"(?:^|\n)\s*(?:\d+\.?\s*)?{re.escape(section_name)}[\s:]*\n"
    next_pattern = "|".join([rf"(?:^|\n)\s*(?:\d+\.?\s*)?{re.escape(s)}[\s:]*\n" for s in next_sections])
    
    # Find section start
    match = re.search(section_pattern, content, re.IGNORECASE)
    if not match:
        return None
    
    start = match.end()
    
    # Find section end
    remaining = content[start:]
    end_match = re.search(next_pattern, remaining, re.IGNORECASE)
    
    if end_match:
        section_content = remaining[:end_match.start()]
    else:
        section_content = remaining[:3000]  # Limit if no end found
    
    return section_content.strip()


def extract_abstract(content: str) -> Optional[str]:
    """Extract abstract from paper."""
    return extract_section(content, "Abstract", ["Introduction", "1.", "Keywords"])


def extract_introduction(content: str) -> Optional[str]:
    """Extract introduction from paper."""
    return extract_section(content, "Introduction", ["Background", "Related Work", "Methodology", "2."])


def extract_methodology(content: str) -> Optional[str]:
    """Extract methodology/methods section from paper."""
    result = extract_section(content, "Methodology")
    if not result:
        result = extract_section(content, "Methods")
    if not result:
        result = extract_section(content, "Method")
    if not result:
        result = extract_section(content, "Approach")
    return result


def extract_results(content: str) -> Optional[str]:
    """Extract results/experiments section from paper."""
    result = extract_section(content, "Results")
    if not result:
        result = extract_section(content, "Experiments")
    if not result:
        result = extract_section(content, "Evaluation")
    return result


def extract_conclusion(content: str) -> Optional[str]:
    """Extract conclusion from paper."""
    result = extract_section(content, "Conclusion")
    if not result:
        result = extract_section(content, "Conclusions")
    return result


def extract_references_list(content: str) -> list[str]:
    """
    Extract references from paper.
    
    Returns:
        List of reference strings
    """
    refs_section = extract_section(content, "References", ["Appendix", "Acknowledgments"])
    if not refs_section:
        refs_section = extract_section(content, "Bibliography", ["Appendix"])
    
    if not refs_section:
        return []
    
    # Split by common patterns
    # Pattern 1: [1], [2], etc.
    refs = re.split(r'\n\s*\[\d+\]', refs_section)
    
    if len(refs) <= 1:
        # Pattern 2: 1., 2., etc.
        refs = re.split(r'\n\s*\d+\.', refs_section)
    
    # Clean up
    references = []
    for ref in refs:
        ref = ref.strip()
        if ref and len(ref) > 20:  # Filter out very short fragments
            references.append(ref)
    
    return references[:50]  # Limit to 50 references


async def extract_paper_sections(pdf_url: str) -> dict:
    """
    Extract all major sections from a paper PDF.
    
    Args:
        pdf_url: URL to PDF file
        
    Returns:
        Dict with extracted sections
    """
    pdf_content = download_pdf(pdf_url)
    if not pdf_content:
        return {"error": "Failed to download PDF"}
    
    text = extract_text_from_pdf(pdf_content, max_pages=20)
    if not text:
        return {"error": "Failed to extract text"}
    
    return {
        "abstract": extract_abstract(text),
        "introduction": extract_introduction(text),
        "methodology": extract_methodology(text),
        "results": extract_results(text),
        "conclusion": extract_conclusion(text),
        "references": extract_references_list(text),
        "full_text_length": len(text)
    }
