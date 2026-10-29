"""On-demand user provisioning for partner apps whose users reach /mcp through the RFC 7523
jwt-bearer grant (routers/broker_provisioning.py exposes it).

Keycloak's grant resolves an assertion's `sub` only through an existing federated-identity link
on the partner's identity provider and never creates one, so a partner vouching for someone we
have never seen gets `invalid_grant "User not found"`. This is what the partner calls at that
point: create the realm user when missing, link them to the partner's provider, after the same
e-mail allow-list the token paths apply. The caller proves it is the partner with a
client-credentials token of its own Keycloak client, and the client id doubles as the provider
alias — the convention keycloak-register-brainzzz.sh in genetics-results-suite follows.

Trust, stated plainly: a client on BROKER_PROVISIONING_CLIENTS can make any allow-listed address
reachable through its own assertions. That is what "no GeneGenie login" means, and the allow-list
is the boundary that stays ours. A user token the partner obtained through the grant is refused
here on purpose: it names a person, not the partner.
"""

import asyncio
import logging
import re
from dataclasses import dataclass

import httpx
import jwt

from genetics_mcp_server.access_grants import KeycloakTarget

logger = logging.getLogger(__name__)

# one @, no whitespace: the admin API would store anything, and a stray space in an address
# would create a user nobody can ever match
_EMAIL = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")


def configured(settings) -> bool:
    return bool(
        settings.broker_provisioning_clients
        and settings.keycloak_admin_url
        and settings.keycloak_access_grants_client_secret
        and settings.oauth_issuer
    )


def normalise_email(raw: str) -> str | None:
    email = raw.strip().lower()
    return email if _EMAIL.match(email) else None


def _jwks_uri(settings) -> str:
    # the realm's keys are the same at every URL; the in-cluster one needs no ingress hairpin
    if settings.oauth_jwks_uri:
        return settings.oauth_jwks_uri
    if settings.keycloak_admin_url:
        return f"{settings.keycloak_admin_url}/realms/{settings.keycloak_realm}/protocol/openid-connect/certs"
    return settings.resolved_oauth_jwks_uri


def _signing_key(jwks_uri: str, token: str):
    return jwt.PyJWKClient(jwks_uri).get_signing_key_from_jwt(token).key


def caller_client_id(token: str, settings) -> str | None:
    """The Keycloak client whose client-credentials token this is, or None.

    Verifies signature, issuer and expiry. The audience is deliberately not checked: the client's
    audience mapper stamps the /mcp resource onto every token it issues, and what matters here is
    not where the token may be presented but who it was issued to (`azp`) and that it is the
    client's service account rather than one of its users (`preferred_username`).
    """
    try:
        key = _signing_key(_jwks_uri(settings), token)
        payload = jwt.decode(
            token,
            key,
            algorithms=["RS256"],
            issuer=settings.oauth_issuer,
            options={"verify_aud": False, "verify_exp": True},
        )
    except Exception as e:  # noqa: BLE001 - any failure is a 401, never a 500
        logger.debug("provisioning token refused: %s", e)
        return None
    azp = payload.get("azp")
    if not azp or payload.get("preferred_username") != f"service-account-{azp}":
        return None
    return azp


class AccountDisabled(Exception):
    pass


@dataclass
class ProvisionResult:
    email: str
    created: bool
    newly_linked: bool


class BrokerProvisioner(KeycloakTarget):
    """Create-if-missing and link-if-unlinked, with the access-grants service account."""

    async def ensure_linked(self, email: str, alias: str) -> ProvisionResult:
        admin = f"/admin/realms/{self.realm}"
        async with httpx.AsyncClient(base_url=self.base_url, timeout=15.0) as client:
            client.headers["Authorization"] = f"Bearer {await self._token(client)}"

            resp = await client.get(f"{admin}/users", params={"email": email, "exact": "true"})
            resp.raise_for_status()
            users = [u for u in resp.json() if (u.get("email") or "").lower() == email]
            created = False
            if users:
                user = users[0]
            else:
                resp = await client.post(
                    f"{admin}/users",
                    json={
                        "username": email,
                        "email": email,
                        "enabled": True,
                        "emailVerified": True,
                    },
                )
                resp.raise_for_status()
                user = {"id": resp.headers["Location"].rstrip("/").rsplit("/", 1)[-1]}
                created = True
            # an account the access-grant reconciler disabled stays disabled: the partner's
            # word is not a re-grant
            if not user.get("enabled", True):
                raise AccountDisabled(email)

            resp = await client.get(f"{admin}/users/{user['id']}/federated-identity")
            resp.raise_for_status()
            newly_linked = False
            if not any(f.get("identityProvider") == alias for f in resp.json()):
                resp = await client.post(
                    f"{admin}/users/{user['id']}/federated-identity/{alias}",
                    json={"identityProvider": alias, "userId": email, "userName": email},
                )
                resp.raise_for_status()
                newly_linked = True
        logger.info(
            "provisioned %s for %s (created=%s, newly_linked=%s)",
            email,
            alias,
            created,
            newly_linked,
        )
        return ProvisionResult(email, created, newly_linked)


def provisioner(settings) -> BrokerProvisioner:
    return BrokerProvisioner(
        settings.keycloak_admin_url,
        settings.keycloak_realm,
        settings.keycloak_access_grants_client_id,
        settings.keycloak_access_grants_client_secret,
    )


async def verify_caller(token: str, settings) -> str | None:
    # the JWKS fetch inside is blocking network I/O
    return await asyncio.to_thread(caller_client_id, token, settings)
