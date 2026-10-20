"""Runtime access grants: admit an address for a bounded time without a deploy.

Three gates decide who gets in, and each has its own copy of the allow-list: Keycloak's
first-broker-login authenticator (realm attribute `allowedEmails`), oauth2-proxy
(`--authenticated-emails-file`, a ConfigMap-mounted file it watches) and the backends
(`ALLOWED_EMAILS` env, plus the `access_grants` table through `auth.core._email_allowed`).
`deploy.sh` renders the first two from terraform, so a grant written straight into them would
be lost on the next deploy. The table is therefore the record, and the reconciler below
re-derives the two other copies from it — on every grant and revocation, on a timer, and at
startup — so a deploy that resets them is repaired within one interval, and expiry needs no
scheduled job of its own.

The static list the reconciler merges with is `ALLOWED_EMAILS`: the same `${OAUTH_ALLOWED_EMAILS}`
deploy.sh writes into the oauth2-proxy file and the realm attribute, so what it writes back
is a superset of what a deploy would have written.
"""

import asyncio
import logging
import re
import smtplib
from dataclasses import dataclass
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path

import httpx

from genetics_mcp_server.config import get_settings
from genetics_mcp_server.db import get_llm_config_db

logger = logging.getLogger(__name__)

# consumer domains whose accounts are known to sign in through the respective IdP. Any other
# address may still be a Google, Apple or Microsoft account — all three let a person register an
# existing address as their login — so a miss here is advice to the admin, never a refusal.
GOOGLE_DOMAINS = frozenset({"gmail.com", "googlemail.com"})
APPLE_DOMAINS = frozenset({"icloud.com", "me.com", "mac.com", "privaterelay.appleid.com"})
MICROSOFT_DOMAINS = frozenset({"outlook.com", "hotmail.com", "live.com", "msn.com"})
# a domain's mail host tells which IdP its users most likely have: Google Workspace, iCloud+
# custom domains and Microsoft 365 all publish distinctive MX records; Microsoft's consumer
# domains outside the set above (hotmail.co.uk, outlook.fr, ...) sit under olc.protection
GOOGLE_MX_SUFFIXES = ("google.com.", "googlemail.com.")
APPLE_MX_SUFFIXES = ("mail.icloud.com.",)
MICROSOFT_CONSUMER_MX_SUFFIXES = ("olc.protection.outlook.com.",)
MICROSOFT_365_MX_SUFFIXES = ("mail.protection.outlook.com.",)
DOH_URL = "https://dns.google/resolve"

# good enough to refuse a typo before it is mailed; the IdP decides what an address really is
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

RECONCILE_INTERVAL_S = 30
SA_DIR = Path("/var/run/secrets/kubernetes.io/serviceaccount")
CONFIGMAP_KEY = "allowed-emails.txt"
REALM_ATTRIBUTE = "allowedEmails"


def normalize_email(email: str) -> str:
    return email.strip().lower()


def is_valid_email(email: str) -> bool:
    return bool(EMAIL_RE.match(email))


def parse_expires_at(value: str) -> datetime | None:
    """The configured deadline as an aware UTC datetime, or None when unset or unparsable."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        logger.error("ACCESS_GRANT_EXPIRES_AT is not an ISO 8601 timestamp: %r", value)
        return None
    if parsed.tzinfo is None:
        # a naive deadline is ambiguous by exactly the amount that matters at a conference
        logger.error("ACCESS_GRANT_EXPIRES_AT must carry a UTC offset: %r", value)
        return None
    return parsed.astimezone(timezone.utc)


async def classify_provider(email: str, client: httpx.AsyncClient | None = None) -> str:
    """Which identity provider the address most likely belongs to.

    "google" / "apple" / "microsoft" for the consumer domains, "google-workspace" or
    "microsoft-365" when the domain's MX points at Google or Microsoft 365 (an iCloud+ custom
    domain reports "apple", a regional Outlook domain "microsoft"), "unknown" otherwise —
    including when the lookup itself fails, because the answer only steers a confirmation prompt.
    """
    domain = normalize_email(email).rsplit("@", 1)[-1]
    if domain in GOOGLE_DOMAINS:
        return "google"
    if domain in APPLE_DOMAINS:
        return "apple"
    if domain in MICROSOFT_DOMAINS:
        return "microsoft"
    try:
        own = client is None
        client = client or httpx.AsyncClient(timeout=4.0)
        try:
            resp = await client.get(DOH_URL, params={"name": domain, "type": "MX"})
            resp.raise_for_status()
            answers = resp.json().get("Answer") or []
        finally:
            if own:
                await client.aclose()
    except Exception as exc:
        logger.warning("MX lookup for %s failed: %s", domain, exc)
        return "unknown"
    hosts = [str(a.get("data", "")).rsplit(" ", 1)[-1].lower() for a in answers]
    if any(h.endswith(GOOGLE_MX_SUFFIXES) for h in hosts):
        return "google-workspace"
    if any(h.endswith(APPLE_MX_SUFFIXES) for h in hosts):
        return "apple"
    if any(h.endswith(MICROSOFT_CONSUMER_MX_SUFFIXES) for h in hosts):
        return "microsoft"
    if any(h.endswith(MICROSOFT_365_MX_SUFFIXES) for h in hosts):
        return "microsoft-365"
    return "unknown"


# ---------------------------------------------------------------------------
# notification e-mail
# ---------------------------------------------------------------------------


def email_configured() -> bool:
    s = get_settings()
    return bool(s.smtp_host and s.access_grant_email_from and s.access_grant_email_body_file)


def send_notification(email: str) -> None:
    """Mail the granted address. Raises on any failure; the caller records the text."""
    s = get_settings()
    body = Path(s.access_grant_email_body_file).read_text(encoding="utf-8")
    msg = EmailMessage()
    msg["From"] = s.access_grant_email_from
    msg["To"] = email
    msg["Subject"] = s.access_grant_email_subject or "Access granted"
    if s.access_grant_email_reply_to:
        msg["Reply-To"] = s.access_grant_email_reply_to
    # replace, not format: the body is prose an operator wrote, and a brace in it is just a brace
    msg.set_content(body.replace("{email}", email))
    with smtplib.SMTP(s.smtp_host, s.smtp_port, timeout=20) as smtp:
        smtp.starttls()
        if s.smtp_username:
            smtp.login(s.smtp_username, s.smtp_password)
        smtp.send_message(msg)


async def notify(email: str) -> str | None:
    """Send the notification and record the outcome on the grant. Returns the error text, if any."""
    if not email_configured():
        error = "e-mail sending is not configured on this deployment"
    else:
        try:
            await asyncio.to_thread(send_notification, email)
            error = None
        except Exception as exc:
            logger.warning("access-grant e-mail to %s failed: %s", email, exc)
            error = f"{type(exc).__name__}: {exc}"
    get_llm_config_db().record_access_grant_email(email, error)
    return error


# ---------------------------------------------------------------------------
# reconciliation targets
# ---------------------------------------------------------------------------


class ConfigMapTarget:
    """Keeps oauth2-proxy's authenticated-emails file equal to the static list plus active grants.

    Talks to the API server directly with the pod's projected token: a client library would be
    a heavy dependency for one GET and one merge-patch. The token is re-read per call because
    projected tokens rotate.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    @staticmethod
    def available() -> bool:
        return (SA_DIR / "token").exists()

    def _client(self) -> httpx.AsyncClient:
        import os

        host = os.environ.get("KUBERNETES_SERVICE_HOST", "kubernetes.default.svc")
        port = os.environ.get("KUBERNETES_SERVICE_PORT", "443")
        token = (SA_DIR / "token").read_text().strip()
        return httpx.AsyncClient(
            base_url=f"https://{host}:{port}",
            headers={"Authorization": f"Bearer {token}"},
            verify=str(SA_DIR / "ca.crt"),
            timeout=10.0,
        )

    async def sync(self, desired: set[str]) -> bool:
        """Patch the file when it differs. True when a patch was written."""
        namespace = (SA_DIR / "namespace").read_text().strip()
        path = f"/api/v1/namespaces/{namespace}/configmaps/{self.name}"
        async with self._client() as client:
            resp = await client.get(path)
            resp.raise_for_status()
            current_text = (resp.json().get("data") or {}).get(CONFIGMAP_KEY, "")
            current = {line.strip() for line in current_text.splitlines() if line.strip()}
            if current == desired:
                return False
            # deploy.sh writes one address per line; keep that shape so a diff against what a
            # deploy would render stays readable
            body = {"data": {CONFIGMAP_KEY: "\n".join(sorted(desired))}}
            resp = await client.patch(
                path, json=body, headers={"Content-Type": "application/merge-patch+json"}
            )
            resp.raise_for_status()
        logger.info("configmap %s: allow-list now holds %d addresses", self.name, len(desired))
        return True


class KeycloakTarget:
    """Keeps the realm's `allowedEmails` attribute in step, and disables accounts of ended grants.

    The first-broker-login authenticator only runs at a user's first login, so once a grant
    has produced an account the attribute no longer gates that person; disabling the account
    is what ends their Keycloak session path. Only accounts named by a grant row are ever
    touched. Writes go GET -> modify -> PUT of the full representation, the way kcadm does,
    because a realm PUT drops attributes absent from the body.
    """

    def __init__(self, base_url: str, realm: str, client_id: str, client_secret: str) -> None:
        self.base_url = base_url
        self.realm = realm
        self.client_id = client_id
        self.client_secret = client_secret

    async def _token(self, client: httpx.AsyncClient) -> str:
        resp = await client.post(
            f"/realms/{self.realm}/protocol/openid-connect/token",
            data={
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
            },
        )
        resp.raise_for_status()
        return resp.json()["access_token"]

    async def sync(
        self, desired: set[str], ended: list[str], reactivated: list[str]
    ) -> list[str]:
        """Apply the attribute and account states. Returns the ended addresses actually disabled."""
        admin = f"/admin/realms/{self.realm}"
        async with httpx.AsyncClient(base_url=self.base_url, timeout=15.0) as client:
            client.headers["Authorization"] = f"Bearer {await self._token(client)}"

            resp = await client.get(admin)
            resp.raise_for_status()
            realm = resp.json()
            attributes = dict(realm.get("attributes") or {})
            current = {
                e.strip().lower()
                for e in re.split(r"[,;\s]+", attributes.get(REALM_ATTRIBUTE, ""))
                if e.strip()
            }
            if current != desired:
                attributes[REALM_ATTRIBUTE] = ",".join(sorted(desired))
                realm["attributes"] = attributes
                resp = await client.put(admin, json=realm)
                resp.raise_for_status()
                logger.info("keycloak realm %s: %s now holds %d addresses",
                            self.realm, REALM_ATTRIBUTE, len(desired))

            disabled: list[str] = []
            for email in ended:
                if await self._set_enabled(client, admin, email, False):
                    disabled.append(email)
            for email in reactivated:
                await self._set_enabled(client, admin, email, True)
        return disabled

    async def _set_enabled(
        self, client: httpx.AsyncClient, admin: str, email: str, enabled: bool
    ) -> bool:
        """Set the account's enabled flag. True when the address has an account (changed or not)."""
        resp = await client.get(f"{admin}/users", params={"email": email, "exact": "true"})
        resp.raise_for_status()
        users = [u for u in resp.json() if (u.get("email") or "").lower() == email]
        if not users:
            return False
        user = users[0]
        if bool(user.get("enabled", True)) != enabled:
            user["enabled"] = enabled
            resp = await client.put(f"{admin}/users/{user['id']}", json=user)
            resp.raise_for_status()
            logger.info("keycloak user %s %s", email, "enabled" if enabled else "disabled")
        return True


# ---------------------------------------------------------------------------
# the reconciler
# ---------------------------------------------------------------------------


@dataclass
class ReconcileResult:
    desired: set[str]
    configmap_patched: bool
    keycloak_disabled: list[str]


class AccessGrantReconciler:
    def __init__(self) -> None:
        self._kick = asyncio.Event()
        self._task: asyncio.Task | None = None
        settings = get_settings()
        self.configmap = (
            ConfigMapTarget(settings.access_grant_configmap)
            if settings.access_grant_configmap and ConfigMapTarget.available()
            else None
        )
        if settings.access_grant_configmap and self.configmap is None:
            logger.warning(
                "ACCESS_GRANT_CONFIGMAP is set but no service-account token is mounted; "
                "oauth2-proxy will not learn about grants"
            )
        self.keycloak = (
            KeycloakTarget(
                settings.keycloak_admin_url,
                settings.keycloak_realm,
                settings.keycloak_access_grants_client_id,
                settings.keycloak_access_grants_client_secret,
            )
            if settings.keycloak_admin_url and settings.keycloak_access_grants_client_secret
            else None
        )

    def kick(self) -> None:
        """Run a pass soon rather than at the next tick."""
        self._kick.set()

    def desired_emails(self, now: datetime) -> set[str]:
        settings = get_settings()
        static = {normalize_email(e) for e in settings.allowed_emails if e.strip()}
        return static | get_llm_config_db().active_access_grant_emails(now)

    async def run_once(self) -> ReconcileResult:
        now = datetime.now(timezone.utc)
        db = get_llm_config_db()
        desired = self.desired_emails(now)
        grants = db.list_access_grants()
        ended = [g.email for g in grants if not g.is_active(now) and g.deactivated_at is None]
        # a re-granted address has had its deactivation mark cleared; re-enable the account
        # the previous expiry disabled. Harmless for first-time grants, whose account does not
        # exist yet
        reactivated = [g.email for g in grants if g.is_active(now)]

        patched = False
        if self.configmap is not None:
            try:
                patched = await self.configmap.sync(desired)
            except Exception as exc:
                logger.error("access-grant configmap sync failed: %s", exc)

        disabled: list[str] = []
        if self.keycloak is not None:
            try:
                disabled = await self.keycloak.sync(desired, ended, reactivated)
            except Exception as exc:
                logger.error("access-grant keycloak sync failed: %s", exc)
                ended = []
        # with no Keycloak target there is nothing to withdraw, so the mark just records that
        # the grant's end was seen
        for email in ended if self.keycloak is None else disabled:
            db.mark_access_grant_deactivated(email)
        # an ended grant whose address never produced an account has nothing to disable; mark
        # it too, or every pass re-queries Keycloak for it
        if self.keycloak is not None:
            for email in set(ended) - set(disabled):
                db.mark_access_grant_deactivated(email)
        return ReconcileResult(desired, patched, disabled)

    async def _loop(self) -> None:
        while True:
            try:
                await self.run_once()
            except Exception as exc:
                logger.error("access-grant reconcile failed: %s", exc)
            try:
                await asyncio.wait_for(self._kick.wait(), timeout=RECONCILE_INTERVAL_S)
            except asyncio.TimeoutError:
                pass
            self._kick.clear()

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._loop())

    def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            self._task = None


_reconciler: AccessGrantReconciler | None = None


def get_reconciler() -> AccessGrantReconciler | None:
    """The process's reconciler, or None when the feature is off."""
    global _reconciler
    if _reconciler is None and get_settings().access_grants_enabled:
        _reconciler = AccessGrantReconciler()
    return _reconciler


def reset_reconciler() -> None:
    """Drop the cached instance so the next get_reconciler builds from current settings (tests)."""
    global _reconciler
    if _reconciler is not None:
        _reconciler.stop()
    _reconciler = None
