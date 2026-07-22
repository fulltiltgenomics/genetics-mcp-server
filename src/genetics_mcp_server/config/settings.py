"""Configuration settings for genetics MCP server."""

import os
import re
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

    # branding
    app_name: str = field(
        default_factory=lambda: os.environ.get("APP_NAME", "FinnGenie")
    )

    # LLM defaults
    default_provider: str = "anthropic"
    default_model: str = field(
        default_factory=lambda: os.environ.get("DEFAULT_MODEL", "claude-sonnet-4-6")
    )
    fast_model: str = "claude-haiku-4-5"
    max_tokens: int = 8192
    # temperature is off by default; many current models (Fable, Opus 4.7+)
    # reject it. set TEMPERATURE to opt in for models that still support it.
    temperature: float | None = field(
        default_factory=lambda: (
            float(os.environ["TEMPERATURE"])
            if os.environ.get("TEMPERATURE", "").strip()
            else None
        )
    )

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

    # myvariant.info API
    myvariant_api_url: str = field(
        default_factory=lambda: os.environ.get(
            "MYVARIANT_API_URL", "https://myvariant.info/v1"
        )
    )

    # UniProt REST API (protein entries, search, sequences)
    uniprot_api_url: str = field(
        default_factory=lambda: os.environ.get(
            "UNIPROT_API_URL", "https://rest.uniprot.org"
        )
    )
    # EBI Proteins API, used for protein-position-to-genome coordinate mapping
    ebi_proteins_api_url: str = field(
        default_factory=lambda: os.environ.get(
            "EBI_PROTEINS_API_URL", "https://www.ebi.ac.uk/proteins/api"
        )
    )
    # UniProt entries change at most weekly, so a long TTL is safe
    uniprot_cache_ttl: int = field(
        default_factory=lambda: int(os.environ.get("UNIPROT_CACHE_TTL", "86400"))
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

    # download storage (for tool result TSV files)
    download_storage_path: str = field(
        default_factory=lambda: os.environ.get(
            "DOWNLOAD_STORAGE_PATH", "/mnt/disks/data/downloads"
        )
    )
    download_ttl_seconds: int = field(
        default_factory=lambda: int(os.environ.get("DOWNLOAD_TTL_SECONDS", "2592000"))
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
    # cap on a single user message: typed text length (excludes attachments) and
    # number of attachment blocks (image/document) per message
    max_message_chars: int = field(
        default_factory=lambda: int(os.environ.get("MAX_MESSAGE_CHARS", "50000"))
    )
    max_attachments_per_message: int = field(
        default_factory=lambda: int(os.environ.get("MAX_ATTACHMENTS_PER_MESSAGE", "10"))
    )

    # admin page
    enable_admin_page: bool = field(
        default_factory=lambda: os.environ.get(
            "ENABLE_ADMIN_PAGE", "false"
        ).lower() in ("1", "true", "yes")
    )
    admin_users: str = field(
        default_factory=lambda: os.environ.get("ADMIN_USERS", "")
    )

    # bearer token auth: allowed email domains and specific emails
    # mirrors genetics-results-api/app/config/common.py parsing semantics
    allowed_email_domains: set[str] = field(
        default_factory=lambda: {
            d.strip() for d in os.environ.get("ALLOWED_EMAIL_DOMAINS", "finngen.fi").split(",") if d.strip()
        }
    )
    allowed_emails: set[str] = field(
        default_factory=lambda: {
            e.strip() for e in os.environ.get("ALLOWED_EMAILS", "").split(",") if e.strip()
        }
    )

    # CORS: the frontend sends credentialed requests, and browsers reject a
    # wildcard Access-Control-Allow-Origin on those, so origins must be explicit.
    # only relevant in dev — in prod the frontend and this API share an origin
    # behind the reverse proxy and no CORS preflight happens.
    cors_origins: list[str] = field(
        default_factory=lambda: [
            o.strip()
            for o in os.environ.get(
                "CORS_ORIGINS",
                "http://localhost:3000,http://127.0.0.1:3000",
            ).split(",")
            if o.strip()
        ]
    )

    # OAuth 2.1 resource server (optional): trust Keycloak-issued access tokens
    # as a fourth bearer-validation path. inert unless both issuer and resource
    # url are set. token validation reuses allowed_emails / allowed_email_domains.
    oauth_issuer: str | None = field(
        default_factory=lambda: os.environ.get("OAUTH_ISSUER") or None
    )
    oauth_resource_url: str | None = field(
        default_factory=lambda: os.environ.get("OAUTH_RESOURCE_URL") or None
    )
    oauth_jwks_uri: str | None = field(
        default_factory=lambda: os.environ.get("OAUTH_JWKS_URI") or None
    )

    @property
    def oauth_enabled(self) -> bool:
        return bool(self.oauth_issuer and self.oauth_resource_url)

    @property
    def resolved_oauth_jwks_uri(self) -> str | None:
        # keycloak exposes its JWKS at a fixed path relative to the realm issuer
        if self.oauth_jwks_uri:
            return self.oauth_jwks_uri
        if self.oauth_issuer:
            return f"{self.oauth_issuer.rstrip('/')}/protocol/openid-connect/certs"
        return None

    @property
    def admin_users_list(self) -> list[str]:
        if not self.admin_users:
            return []
        return [u.strip().lower() for u in self.admin_users.split(",") if u.strip()]

    # subagent settings
    enable_subagents: bool = field(
        default_factory=lambda: os.environ.get(
            "ENABLE_SUBAGENTS", "false"
        ).lower() in ("1", "true", "yes")
    )
    subagent_model: str = field(
        default_factory=lambda: os.environ.get("SUBAGENT_MODEL", "")
    )
    subagent_max_tokens: int = 4096
    subagent_max_iterations: int = 10
    subagent_timeout: int = field(
        default_factory=lambda: int(os.environ.get("SUBAGENT_TIMEOUT", "120"))
    )
    subagent_allowed_paths: str = field(
        default_factory=lambda: os.environ.get("SUBAGENT_ALLOWED_PATHS", "")
    )
    enable_script_execution: bool = field(
        default_factory=lambda: os.environ.get(
            "ENABLE_SCRIPT_EXECUTION", "false"
        ).lower() in ("1", "true", "yes")
    )
    subagent_script_timeout: int = field(
        default_factory=lambda: int(os.environ.get("SUBAGENT_SCRIPT_TIMEOUT", "30"))
    )

    @property
    def subagent_allowed_paths_list(self) -> list[str]:
        if not self.subagent_allowed_paths:
            return []
        return [p.strip() for p in self.subagent_allowed_paths.split(",") if p.strip()]

    @property
    def disabled_tools(self) -> set[str]:
        disabled = set()
        if not self.enable_credible_sets_stats:
            disabled.add("get_credible_sets_stats")
        if not self.enable_phenotype_report:
            disabled.add("get_phenotype_report")
        if not self.enable_subagents:
            disabled.add("launch_subagents")
        return disabled


# Claude Opus deprecated the temperature parameter starting with 4.7;
# assume every Opus from that version onward (4.7+, 5.x, …) rejects it.
# Claude Fable models don't support temperature at all.
_OPUS_TEMPERATURE_FLOOR = (4, 7)
_OPUS_VERSION_RE = re.compile(r"claude-opus-(\d+)-(\d+)")
_FABLE_RE = re.compile(r"claude-fable-")


def model_rejects_temperature(model: str) -> bool:
    """Check if a model doesn't support the temperature parameter."""
    if _FABLE_RE.search(model):
        return True
    match = _OPUS_VERSION_RE.search(model)
    if match:
        version = (int(match.group(1)), int(match.group(2)))
        return version >= _OPUS_TEMPERATURE_FLOOR
    return False


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
