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
    
    # Agent Settings
    max_iterations: int = Field(default=5)
    max_search_results: int = Field(default=10)
    max_papers: int = Field(default=5)
    
    # Paths
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


# Global config instance
config = Config()


def get_config() -> Config:
    """Get the global configuration instance."""
    return config
