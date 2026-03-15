"""Configuration settings for genetics MCP server."""

import os
from dataclasses import dataclass, field
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    """Application settings loaded from environment."""

    # genetics API
    genetics_api_url: str = field(
        default_factory=lambda: os.environ.get(
            "GENETICS_API_URL", "http://localhost:2000/api"
        )
    )

    # BigQuery API (for direct SQL queries)
    bigquery_api_url: str | None = field(
        default_factory=lambda: os.environ.get("BIGQUERY_API_URL")
    )

    # LLM providers
    anthropic_api_key: str | None = field(
        default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY")
    )
    openai_api_key: str | None = field(
        default_factory=lambda: os.environ.get("OPENAI_API_KEY")
    )

    # web search
    tavily_api_key: str | None = field(
        default_factory=lambda: os.environ.get("TAVILY_API_KEY")
    )

    # literature search
    perplexity_api_key: str | None = field(
        default_factory=lambda: os.environ.get("PERPLEXITY_API_KEY")
    )
    literature_search_backend: str = field(
        default_factory=lambda: os.environ.get("LITERATURE_SEARCH_BACKEND", "europepmc")
    )

    # LLM defaults
    default_provider: str = "anthropic"
    default_model: str = "claude-sonnet-4-6"
    fast_model: str = "claude-haiku-4-5"
    max_tokens: int = 8192
    temperature: float = 0.3

    # MCP settings
    mcp_enabled: bool = True
    mcp_max_iterations: int = 25
    mcp_max_result_size: int = 50000

    # optional tools (disabled by default)
    enable_credible_sets_stats: bool = field(
        default_factory=lambda: os.environ.get(
            "ENABLE_CREDIBLE_SETS_STATS", "false"
        ).lower() in ("1", "true", "yes")
    )
    enable_phenotype_report: bool = field(
        default_factory=lambda: os.environ.get(
            "ENABLE_PHENOTYPE_REPORT", "false"
        ).lower() in ("1", "true", "yes")
    )

    # RAG MCP server (separate from always-on external servers)
    rag_mcp_server: str | None = field(
        default_factory=lambda: os.environ.get("RAG_MCP_SERVER")
    )

    # database paths
    llm_config_db: str = field(
        default_factory=lambda: os.environ.get(
            "LLM_CONFIG_DB", "/mnt/disks/data/llm_config.db"
        )
    )
    chat_history_db: str = field(
        default_factory=lambda: os.environ.get(
            "CHAT_HISTORY_DB", "/mnt/disks/data/chat_history.db"
        )
    )

    # attachment storage
    attachment_storage_path: str = field(
        default_factory=lambda: os.environ.get(
            "ATTACHMENT_STORAGE_PATH", "/mnt/disks/data/attachments"
        )
    )
    max_attachment_size: int = field(
        default_factory=lambda: int(os.environ.get("MAX_ATTACHMENT_SIZE", "52428800"))  # 50MB
    )


    @property
    def disabled_tools(self) -> set[str]:
        disabled = set()
        if not self.enable_credible_sets_stats:
            disabled.add("get_credible_sets_stats")
        if not self.enable_phenotype_report:
            disabled.add("get_phenotype_report")
        return disabled


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
