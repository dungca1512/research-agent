"""Structured paper extraction and comparison using Pydantic schemas + LLM."""

from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from src.config import get_config


# ─── Pydantic Schemas for Structured Extraction ─────────────────────────────

class PaperExtraction(BaseModel):
    """Structured extraction of a single paper's key aspects."""

    title: str = Field(description="Paper title")
    methodology: str = Field(
        description="Core methodology or approach used (1-3 sentences)"
    )
    datasets: list[str] = Field(
        default_factory=list,
        description="Datasets or benchmarks used for evaluation",
    )
    metrics: list[str] = Field(
        default_factory=list,
        description="Evaluation metrics reported (e.g., accuracy, F1, BLEU)",
    )
    key_results: list[str] = Field(
        default_factory=list,
        description="Main quantitative or qualitative results (2-5 bullet points)",
    )
    contributions: list[str] = Field(
        default_factory=list,
        description="Novel contributions claimed by the paper",
    )
    limitations: list[str] = Field(
        default_factory=list,
        description="Limitations acknowledged or apparent",
    )


class ComparisonMatrix(BaseModel):
    """Structured comparison of multiple papers."""

    papers: list[PaperExtraction] = Field(
        description="Extracted information for each paper"
    )
    common_themes: list[str] = Field(
        default_factory=list,
        description="Themes or approaches shared across papers",
    )
    key_differences: list[str] = Field(
        default_factory=list,
        description="Major differences between the papers",
    )
    complementary_strengths: list[str] = Field(
        default_factory=list,
        description="How papers complement each other",
    )
    recommendation: str = Field(
        default="",
        description="Brief recommendation on which paper suits what use case",
    )


# ─── Extraction Functions ───────────────────────────────────────────────────

def _get_llm() -> ChatGoogleGenerativeAI:
    config = get_config()
    return ChatGoogleGenerativeAI(
        model=config.llm_model,
        google_api_key=config.google_api_key,
        temperature=0.1,
    )


async def extract_paper_structured(
    title: str,
    content: str,
) -> dict:
    """
    Extract structured information from a paper using with_structured_output.

    Args:
        title: Paper title
        content: Paper abstract or full text (will be truncated to 6000 chars)

    Returns:
        PaperExtraction as dict
    """
    llm = _get_llm()
    structured_llm = llm.with_structured_output(PaperExtraction)

    prompt = f"""Analyze this academic paper and extract structured information.

Paper Title: {title}

Content:
{content[:6000]}

Extract the methodology, datasets, metrics, key results, contributions, and limitations.
Be concise and factual. Only include information explicitly stated or clearly implied."""

    result = await structured_llm.ainvoke([HumanMessage(content=prompt)])
    return result.model_dump()


async def compare_papers_structured(
    papers: list[dict],
) -> dict:
    """
    Compare multiple papers by extracting structured info then generating a matrix.

    Args:
        papers: List of dicts with 'title' and 'summary'/'content' keys

    Returns:
        ComparisonMatrix as dict
    """
    if len(papers) < 2:
        return {"error": "Need at least 2 papers to compare"}

    llm = _get_llm()
    structured_llm = llm.with_structured_output(ComparisonMatrix)

    papers_text = ""
    for i, p in enumerate(papers[:5], 1):
        authors = ", ".join(p.get("authors", [])[:3])
        if len(p.get("authors", [])) > 3:
            authors += " et al."
        content = p.get("summary", p.get("content", p.get("abstract", "")))
        papers_text += f"\n### Paper {i}: {p.get('title', 'N/A')}\n"
        papers_text += f"**Authors:** {authors}\n"
        papers_text += f"**Year:** {p.get('year', p.get('published', 'N/A'))}\n"
        papers_text += f"**Content:** {content[:1500]}\n"

    prompt = f"""Analyze and compare the following research papers.
For each paper, extract: methodology, datasets, metrics, key results, contributions, and limitations.
Then identify common themes, key differences, complementary strengths, and provide a recommendation.

{papers_text}

Be thorough but concise. Only include information explicitly stated or clearly implied in the content."""

    result = await structured_llm.ainvoke([HumanMessage(content=prompt)])
    return result.model_dump()


def format_comparison_markdown(matrix: dict) -> str:
    """Format a ComparisonMatrix dict as a readable Markdown report."""
    lines = ["# Paper Comparison Matrix\n"]

    # Per-paper table
    papers = matrix.get("papers", [])
    if papers:
        lines.append("## Individual Paper Analysis\n")
        for i, p in enumerate(papers, 1):
            lines.append(f"### {i}. {p.get('title', 'N/A')}\n")
            lines.append(f"**Methodology:** {p.get('methodology', 'N/A')}\n")

            if p.get("datasets"):
                lines.append(f"**Datasets:** {', '.join(p['datasets'])}\n")
            if p.get("metrics"):
                lines.append(f"**Metrics:** {', '.join(p['metrics'])}\n")

            if p.get("key_results"):
                lines.append("**Key Results:**")
                for r in p["key_results"]:
                    lines.append(f"- {r}")
                lines.append("")

            if p.get("contributions"):
                lines.append("**Contributions:**")
                for c in p["contributions"]:
                    lines.append(f"- {c}")
                lines.append("")

            if p.get("limitations"):
                lines.append("**Limitations:**")
                for lim in p["limitations"]:
                    lines.append(f"- {lim}")
                lines.append("")

    # Cross-paper analysis
    if matrix.get("common_themes"):
        lines.append("## Common Themes\n")
        for t in matrix["common_themes"]:
            lines.append(f"- {t}")
        lines.append("")

    if matrix.get("key_differences"):
        lines.append("## Key Differences\n")
        for d in matrix["key_differences"]:
            lines.append(f"- {d}")
        lines.append("")

    if matrix.get("complementary_strengths"):
        lines.append("## Complementary Strengths\n")
        for s in matrix["complementary_strengths"]:
            lines.append(f"- {s}")
        lines.append("")

    if matrix.get("recommendation"):
        lines.append("## Recommendation\n")
        lines.append(matrix["recommendation"])
        lines.append("")

    return "\n".join(lines)
