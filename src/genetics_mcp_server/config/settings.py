"""Configuration settings for genetics MCP server."""

import logging
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


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

    # shared secret for service-to-service calls (bearer form of auth_required, and the
    # /tokens/validate caller). read here rather than at each use site so the four consumers
    # cannot disagree, and so no value can be snapshotted before config/settings.py's load_dotenv()
    internal_api_secret: str = field(
        default_factory=lambda: os.environ.get("INTERNAL_API_SECRET", "")
    )

    # auth-gateway's provenance secret (genetics-results-suite-4h6.84). Held by auth-gateway
    # and chat-backend ONLY — not by mcp-server, not by results-api, not by the sandbox — so
    # its presence on a request is a fact about who relayed it that no other holder of
    # internal_api_secret can forge by choosing a different header. Separate from
    # internal_api_secret for exactly the reason sandbox_token_signing_key is: a distinct
    # blast radius and independent rotation.
    gateway_identity_secret: str = field(
        default_factory=lambda: os.environ.get("GATEWAY_IDENTITY_SECRET", "")
    )

    # signing key for the per-execution sandbox tokens (docs/code-execution-security.md §4).
    # Deliberately NOT internal_api_secret: separate key, separate blast radius, independent
    # rotation. Only chat-backend (mint) and db-api/results-api (verify) hold it; the sandbox
    # holds neither this nor internal_api_secret.
    sandbox_token_signing_key: str = field(
        default_factory=lambda: os.environ.get("SANDBOX_TOKEN_SIGNING_KEY", "")
    )

    # base URL of the sandbox supervisor (docs/code-execution-security.md §2, the HTTP
    # contract). ONE value: the in-cluster Service in production, the local Docker container
    # in dev — the contract is identical in both, so nothing downstream may branch on which
    # one this points at. The default is the dev container; the cluster sets it explicitly.
    sandbox_url: str = field(
        default_factory=lambda: os.environ.get("SANDBOX_URL", "http://127.0.0.1:8080")
    )

    # whether a sandbox supervisor is actually deployed at sandbox_url. A DEPLOYMENT FACT,
    # not a preference, and false until the deploy that creates the sandbox sets it — the
    # same env var name db-api and results-api already gate their token verification on, so
    # one value describes the sandbox's existence across the three services that care.
    #
    # While it is false `run_analysis` is absent from `disabled_tools`' complement, so no
    # resolved tool list contains it (genetics-results-suite-4h6.56). Shipping it enabled
    # ahead of the sandbox is the expensive failure, not the safe one: the system prompt
    # tells the model to PREFER it, and an unreachable sandbox classifies as
    # SandboxUnavailable with retryable: True, so every chat turn is steered into a tool
    # that always fails and invites a retry. Removing it from the list also removes the
    # prompt's guidance for it, because the prompt is assembled from the tool list in force
    # (genetics-results-suite-4h6.69) — there is deliberately no prompt edit to remember.
    sandbox_enabled: bool = field(
        default_factory=lambda: os.environ.get(
            "SANDBOX_ENABLED", "false"
        ).lower() in ("1", "true", "yes")
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
        default_factory=lambda: os.environ.get("LITERATURE_SEARCH_BACKEND", "perplexity")
    )

    # branding
    app_name: str = field(
        default_factory=lambda: os.environ.get("APP_NAME", "FinnGenie")
    )

    # LLM defaults
    default_provider: str = "anthropic"
    default_model: str = field(
        default_factory=lambda: os.environ.get("DEFAULT_MODEL", "claude-opus-5")
    )
    fast_model: str = "claude-haiku-4-5"
    # caps thinking + visible text together, and is a ceiling rather than a
    # reservation (only generated tokens are billed). Kept well inside the
    # 5-minute per-iteration timeout in llm_service; turns that need more room
    # are continued across iterations instead of being given a larger single cap.
    max_tokens: int = field(
        default_factory=lambda: int(os.environ.get("MAX_TOKENS", "16384"))
    )
    # how many times a turn cut short by max_tokens may be resumed before the
    # truncation is reported to the user. Bounds a model that always runs long.
    max_continuations: int = field(
        default_factory=lambda: int(os.environ.get("MAX_CONTINUATIONS", "3"))
    )
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
    # caps on the request as a whole. The per-message caps above only ever saw the newest user
    # message, so a client-sent assistant turn and every replayed history turn were unbounded
    # (genetics-results-suite-e0u). Deliberately generous: replayed tool results are legitimately
    # far larger than a typed message, so these bound the payload rather than police it. Text
    # only — inline image data is counted by max_attachments_per_message, not by length
    max_request_chars: int = field(
        default_factory=lambda: int(os.environ.get("MAX_REQUEST_CHARS", "2000000"))
    )
    max_messages_per_request: int = field(
        default_factory=lambda: int(os.environ.get("MAX_MESSAGES_PER_REQUEST", "500"))
    )

    # auth gate. The single source of truth for REQUIRE_AUTH: auth.dependencies used to snapshot
    # it into a module global at import time while chat_api re-read os.environ per request, so a
    # test moving one left the other behind and /chat/v1/auth could report an admin that
    # admin_required then refused (genetics-results-suite-pol).
    require_auth: bool = field(
        default_factory=lambda: os.environ.get(
            "REQUIRE_AUTH", ""
        ).lower() in ("1", "true", "yes")
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

    # whether the deployment supplied an allow-list at all, as opposed to inheriting the
    # finngen.fi default above. auth.core fails open on the proxied-identity path when this is
    # False: chat-backend only started reading these in genetics-results-suite-th2, so a pod
    # that has not yet picked up the bearer-auth-allowed ConfigMap must not refuse every user
    # of a non-finngen deployment. Enforcement is defence in depth; the trusted-proxy marker
    # is what actually closes the hole.
    allow_list_configured: bool = field(
        default_factory=lambda: bool(
            os.environ.get("ALLOWED_EMAIL_DOMAINS") or os.environ.get("ALLOWED_EMAILS")
        )
    )

    # OAuth client id(s) a Google Identity Token must be addressed to (its `aud` claim).
    # google.oauth2.id_token.verify_oauth2_token skips audience verification entirely when
    # this is None, which means ANY Google-signed id_token belonging to an allow-listed email
    # is accepted — including one minted for an unrelated third-party app the user signed
    # into. Set GOOGLE_TOKEN_AUDIENCE to the client id issued for programmatic access here.
    google_token_audience: list[str] = field(
        default_factory=lambda: [
            a.strip()
            for a in os.environ.get("GOOGLE_TOKEN_AUDIENCE", "").split(",")
            if a.strip()
        ]
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
        if not self.sandbox_enabled:
            # only run_analysis: list_capabilities and read_artifact are inert without a
            # sandbox rather than broken by it, and neither is a tool the prompt prefers
            disabled.add("run_analysis")
        return disabled


# Claude Opus deprecated the temperature parameter starting with 4.7;
# assume every Opus from that version onward (4.7+, 5.x, …) rejects it.
# Claude Fable models don't support temperature at all.
_OPUS_TEMPERATURE_FLOOR = (4, 7)
# minor version is optional: Opus 5 and later ship as "claude-opus-5", not "claude-opus-5-0"
_OPUS_VERSION_RE = re.compile(r"claude-opus-(\d+)(?:-(\d+))?")
_SONNET_VERSION_RE = re.compile(r"claude-sonnet-(\d+)(?:-(\d+))?")
_FABLE_RE = re.compile(r"claude-fable-")
_MYTHOS_RE = re.compile(r"claude-mythos-")

# adaptive thinking arrived with the 4.6 generation. Earlier models only accept
# the (now removed) budget_tokens form, so sending them adaptive is a 400.
_ADAPTIVE_THINKING_FLOOR = (4, 6)


def model_rejects_temperature(model: str) -> bool:
    """Check if a model doesn't support the temperature parameter."""
    if _FABLE_RE.search(model):
        return True
    match = _OPUS_VERSION_RE.search(model)
    if match:
        version = (int(match.group(1)), int(match.group(2) or 0))
        return version >= _OPUS_TEMPERATURE_FLOOR
    return False


def model_supports_adaptive_thinking(model: str) -> bool:
    """Check if a model accepts `thinking={"type": "adaptive"}`."""
    if _FABLE_RE.search(model) or _MYTHOS_RE.search(model):
        return True
    for pattern in (_OPUS_VERSION_RE, _SONNET_VERSION_RE):
        match = pattern.search(model)
        if match:
            version = (int(match.group(1)), int(match.group(2) or 0))
            return version >= _ADAPTIVE_THINKING_FLOOR
    return False


def model_rejects_disabled_thinking(model: str) -> bool:
    """Check if a model rejects `thinking={"type": "disabled"}`.

    Fable and Mythos always think, so asking them to stop is a 400 — the only
    way to keep a call thinking-free there is to omit the parameter entirely
    (which still thinks; those models just cannot be turned off). Everything
    else accepts `disabled`, though on Opus 5 only at effort `high` or below —
    callers that raise effort above that must not disable thinking.
    """
    return bool(_FABLE_RE.search(model) or _MYTHOS_RE.search(model))


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


def warn_unless_gateway_identity_secret(component: str) -> None:
    """Log loudly when the gateway provenance secret is missing. Deliberately NOT fatal.

    genetics-results-suite-4h6.84 requires the unset case to FAIL CLOSED, and it does —
    structurally, at dispatch: `auth.core._expected_gateway_marker` answers None, so
    `is_gateway_caller` answers False for every request and `run_analysis` refuses. That
    refusal is unconditional and needs no startup check to hold; this function only makes the
    misconfiguration legible instead of presenting as "code execution mysteriously stopped".

    A startup refusal (`require_internal_api_secret`'s shape) was considered and rejected on
    blast radius. That secret is load-bearing for EVERY outbound call this process makes, so
    a pod without it is useless and crash-looping is the honest outcome. This one gates ONE
    tool: crashing chat-backend over it would turn a code-execution outage into a total chat
    outage, and it would do so in exactly the window where the gateway leads the backend in a
    rollout. The operator-facing gate lives where it can act before anything is applied —
    scripts/deploy.sh refuses to apply when the `gateway-identity-secret` key of
    genetics-secrets is absent or empty, and k8s/deployments/chat-backend.yaml declares the
    secretKeyRef non-optional so the kubelet holds the pod at CreateContainerConfigError
    rather than starting it blind.
    """
    settings = get_settings()
    if not settings.require_auth:
        return
    secret = settings.gateway_identity_secret
    if not secret.strip():
        logger.error(
            "GATEWAY_IDENTITY_SECRET is unset or empty in %s, so no request can be shown to "
            "have come from auth-gateway and code execution (run_analysis) will refuse every "
            "dispatch. In the cluster this is the gateway-identity-secret key of the "
            "genetics-secrets Secret, mounted by auth-gateway and chat-backend only.",
            component,
        )
    elif not secret.isascii():
        logger.error(
            "GATEWAY_IDENTITY_SECRET contains non-ASCII characters in %s. HTTP clients "
            "disagree on how to encode a non-ASCII header value, so it is treated as unset "
            "and code execution will refuse every dispatch. Rotate it to an ASCII value.",
            component,
        )


def require_internal_api_secret(component: str) -> str:
    """The internal credential, or a startup failure that names the variable.

    genetics-results-suite-618: every call this process makes to results-api and db-api goes
    through one client whose header is built as "bearer if the secret is set, nothing if it is
    not". With the variable unset that fallback is silent, it happens at request time rather
    than at startup, and it is invisible at the far end — on a route that resolves no principal
    an anonymous caller's usage log row is identical to an internal one's (measured: 246/246
    NULL user_email on GET /api/v1/rsid/variants over 90 days). results-api now refuses a
    credential-less request on every data-path route (`ANONYMOUS_SURFACE_MINIMAL`,
    genetics-results-suite-rhh), so the same misconfiguration would present as an unexplained
    401 on every tool call with nothing local saying why.

    Deployed entrypoints call this so the pod fails immediately and says which variable is
    missing. Deliberately NOT enforced at import, in `Settings`, or in `ToolExecutor`: a local
    run against an unauthenticated results-api needs no secret (README documents it as
    optional), and the sandbox image holds no internal credential BY DESIGN — see
    `_PrunedInstallSettings` in tools/executor.py. The contract being enforced is
    genetics-results-suite-4h6.9's "a deployed service never falls back to no credential", not
    "this variable is always set".
    """
    secret = get_settings().internal_api_secret
    if not secret.strip():
        raise RuntimeError(
            f"INTERNAL_API_SECRET is unset or empty, so {component} would send every call to "
            "results-api and db-api with no Authorization header at all — anonymously, and "
            "invisibly in their logs. Set INTERNAL_API_SECRET (in the cluster: the "
            "internal-api-secret key of the genetics-secrets Secret). Only the mcp-server "
            "Deployment marks that secretKeyRef optional: true, so a missing key there leaves "
            "the variable unset and the pod starts — which is how you get here. chat-backend's "
            "does NOT, so a missing key stops that pod at CreateContainerConfigError instead."
        )
    # genetics-results-suite-ctq: the secret must be ASCII, and this is where that becomes real
    # rather than documented. Measured off a real socket, HTTP clients disagree on how to put a
    # non-ASCII header value on the wire — node fetch/undici and python-requests send latin-1,
    # aiohttp sends utf-8, httpx 0.28 (this process's own client) refuses to send one at all —
    # so no server-side codec recovers the same secret from every caller and `is_internal_caller`
    # would be well defined for none of them. Checked here rather than in `Settings` on purpose:
    # the same reasoning as above applies, an unset secret is legitimate for a local run and for
    # the sandbox image, so only the deployed entrypoints enforce anything. The failure mode is
    # the good one — the pod fails readiness and the rollout stalls with the old pods still
    # serving, instead of every internal call 401ing at request time with no local signal.
    if not secret.isascii():
        raise RuntimeError(
            "INTERNAL_API_SECRET contains non-ASCII characters. HTTP clients disagree on how "
            "to encode a non-ASCII header value (node/undici and python-requests send latin-1, "
            "aiohttp sends utf-8, httpx refuses to send one at all), so no server-side decoding "
            "recovers the same secret from every caller. Set INTERNAL_API_SECRET to an ASCII "
            "value — scripts/create-secrets.sh generates one with `openssl rand -base64 32`."
        )
    return secret
