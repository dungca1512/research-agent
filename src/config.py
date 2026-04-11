"""Configuration management for Research Agent."""

import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()


class Config(BaseModel):
    """Application configuration."""

    # LLM Settings
    google_api_key: str = Field(default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""))
    openai_api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    llm_model: str = Field(default_factory=lambda: os.getenv("LLM_MODEL", "gemini-2.0-flash"))
    llm_temperature: float = Field(default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.1")))

    # Search Settings
    tavily_api_key: str = Field(default_factory=lambda: os.getenv("TAVILY_API_KEY", ""))
    serp_api_key: str = Field(default_factory=lambda: os.getenv("SERP_API_KEY", ""))

    # Agent Settings
    max_iterations: int = Field(default=5)
    max_search_results: int = Field(default=10)
    max_papers: int = Field(default=5)

    # A2A Endpoints
    search_agent_url: str = Field(default_factory=lambda: os.getenv("SEARCH_AGENT_URL", "http://localhost:8001"))
    paper_agent_url: str = Field(default_factory=lambda: os.getenv("PAPER_AGENT_URL", "http://localhost:8002"))
    synthesis_agent_url: str = Field(default_factory=lambda: os.getenv("SYNTHESIS_AGENT_URL", "http://localhost:8003"))
    tracker_agent_url: str = Field(default_factory=lambda: os.getenv("TRACKER_AGENT_URL", "http://localhost:8004"))
    qa_agent_url: str = Field(default_factory=lambda: os.getenv("QA_AGENT_URL", "http://localhost:8005"))

    # Paths
    project_root: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[1])
    output_dir: Path = Field(default=Path("output"))

    @property
    def has_google_api(self) -> bool:
        return bool(self.google_api_key)

    @property
    def has_openai_api(self) -> bool:
        return bool(self.openai_api_key)

    @property
    def has_tavily_api(self) -> bool:
        return bool(self.tavily_api_key)

    @property
    def output_path(self) -> Path:
        """Return the absolute output directory path."""
        if self.output_dir.is_absolute():
            return self.output_dir
        return self.project_root / self.output_dir

    def agent_health_targets(self) -> list[tuple[str, str]]:
        """Return the configured A2A agent names and URLs."""
        return [
            ("Search Agent", self.search_agent_url),
            ("Paper Agent", self.paper_agent_url),
            ("Synthesis Agent", self.synthesis_agent_url),
            ("Tracker Agent", self.tracker_agent_url),
            ("QA Agent", self.qa_agent_url),
        ]


# Global config instance
config = Config()


def get_config() -> Config:
    """Get the global configuration instance."""
    return config
