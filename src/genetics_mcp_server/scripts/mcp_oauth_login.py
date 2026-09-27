#!/usr/bin/env python3
"""Log a service identity in to an OAuth-protected MCP server and print its credentials.

Servers such as C3PO expose no client-credentials grant, so chat-backend cannot obtain a
token by itself: someone has to complete a browser login once. This runs the OAuth 2.1
authorization-code flow with PKCE — discovery from the server's protected-resource metadata,
dynamic client registration, a URL to sign in at, then the code exchange — and prints the
JSON that `EXTERNAL_MCP_SERVERS`'s `oauth=<ENV VAR>` option expects in that variable:
`token_endpoint`, `client_id` and `refresh_token`. The access token is used once, to list the
server's tools so the entry's `tools=` allow-list can be chosen, and then dropped; only the
refresh token is durable.

The browser is usually not on the machine running this (a dev VM), so the redirect goes to a
loopback address nothing listens on and the operator pastes the address the browser lands on
back here; the code is in its query string. The device-code grant would avoid the paste, but
WorkOS AuthKit, C3PO's issuer, refuses it for a dynamically registered client ("Activation
denied" on the activation page, measured 2026-09-27).

The account that signs in is the identity every chat user then acts as on that server, so
sign in with the account that should own that.

Usage:
    python -m genetics_mcp_server.scripts.mcp_oauth_login https://mcp.c3po.bio --path / --out c3po.json
"""

import argparse
import base64
import hashlib
import json
import logging
import secrets
import sys
from urllib.parse import parse_qs, urlencode, urlparse

import httpx

from genetics_mcp_server.mcp_proxy import MCPProxyClient

# the URL to open is the only thing the operator needs to see
logging.getLogger("httpx").setLevel(logging.WARNING)

# a loopback redirect is what a public native client registers; port 1 so that nothing on the
# operator's machine could be listening and the browser reliably fails to load it, leaving
# the code visible in the address bar
REDIRECT_URI = "http://127.0.0.1:1/callback"


def _get_json(client: httpx.Client, url: str) -> dict:
    response = client.get(url)
    response.raise_for_status()
    return response.json()


def discover(client: httpx.Client, server_url: str) -> tuple[dict, str]:
    """The authorization server's metadata and the resource identifier, found through the
    MCP server's own protected-resource metadata."""
    resource = _get_json(client, f"{server_url.rstrip('/')}/.well-known/oauth-protected-resource")
    issuers = resource.get("authorization_servers") or []
    if not issuers:
        sys.exit(f"{server_url} names no authorization server in its protected-resource metadata")
    issuer = issuers[0].rstrip("/")
    resource_id = resource.get("resource") or server_url
    for suffix in ("/.well-known/oauth-authorization-server", "/.well-known/openid-configuration"):
        try:
            return _get_json(client, f"{issuer}{suffix}"), resource_id
        except httpx.HTTPStatusError:
            continue
    sys.exit(f"{issuer} publishes no authorization-server metadata")


def register_client(client: httpx.Client, metadata: dict, client_name: str) -> str:
    endpoint = metadata.get("registration_endpoint")
    if not endpoint:
        sys.exit("the authorization server offers no dynamic client registration; pass --client-id")
    response = client.post(
        endpoint,
        json={
            "client_name": client_name,
            "redirect_uris": [REDIRECT_URI],
            "grant_types": ["authorization_code", "refresh_token"],
            "response_types": ["code"],
            "token_endpoint_auth_method": "none",
        },
    )
    if response.status_code >= 400:
        sys.exit(f"client registration failed: HTTP {response.status_code} {response.text[:300]}")
    return response.json()["client_id"]


def _pkce() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(64)
    digest = hashlib.sha256(verifier.encode()).digest()
    return verifier, base64.urlsafe_b64encode(digest).rstrip(b"=").decode()


def _read_code(expected_state: str) -> str:
    """The authorization code, from the redirect address the operator pastes back."""
    print(
        "Paste the address the browser landed on (it starts with the redirect URI): ",
        end="",
        file=sys.stderr,
        flush=True,
    )
    pasted = sys.stdin.readline().strip()
    if not pasted:
        sys.exit("nothing pasted")
    query = parse_qs(urlparse(pasted).query) if "?" in pasted else {"code": [pasted]}
    if query.get("error"):
        sys.exit(f"login refused: {query['error'][0]}: {query.get('error_description', [''])[0]}")
    if "state" in query and query["state"][0] != expected_state:
        sys.exit("the pasted address carries a different state than this run issued; run again")
    if not query.get("code"):
        sys.exit("no code in the pasted address")
    return query["code"][0]


def code_login(client: httpx.Client, metadata: dict, client_id: str, scope: str, resource: str) -> dict:
    verifier, challenge = _pkce()
    state = secrets.token_urlsafe(16)
    params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": REDIRECT_URI,
        "scope": scope,
        "state": state,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        # RFC 8707: the token is bound to this MCP server, as the MCP spec requires
        "resource": resource,
    }
    print(
        f"\nOpen this in a browser and sign in:\n\n{metadata['authorization_endpoint']}?{urlencode(params)}\n\n"
        f"The browser will then fail to load {REDIRECT_URI}; that is expected.\n",
        file=sys.stderr,
    )
    code = _read_code(state)
    response = client.post(
        metadata["token_endpoint"],
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": REDIRECT_URI,
            "client_id": client_id,
            "code_verifier": verifier,
            "resource": resource,
        },
    )
    if response.status_code != 200:
        sys.exit(f"code exchange failed: HTTP {response.status_code} {response.text[:300]}")
    return response.json()


def list_tools(server_url: str, path: str, access_token: str) -> list[dict]:
    proxy = MCPProxyClient(base_url=server_url, timeout=60.0, auth_token=access_token, endpoint_path=path)
    return proxy.list_tools_sync()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("server_url", help="the MCP server, e.g. https://mcp.c3po.bio")
    parser.add_argument("--path", default="/mcp", help="the JSON-RPC endpoint under the URL (default /mcp; C3PO answers at /)")
    parser.add_argument("--client-name", default="genetics-chat-backend")
    parser.add_argument(
        "--client-id", help=f"skip registration and use this public client id (it must have {REDIRECT_URI} registered)"
    )
    parser.add_argument("--scope", default="openid offline_access", help="offline_access is what yields a refresh token")
    parser.add_argument("--out", help="write the credential JSON here instead of stdout")
    parser.add_argument("--no-list-tools", action="store_true", help="skip the tools/list after login")
    args = parser.parse_args()

    with httpx.Client(timeout=30.0) as client:
        metadata, resource = discover(client, args.server_url)
        client_id = args.client_id or register_client(client, metadata, args.client_name)
        tokens = code_login(client, metadata, client_id, args.scope, resource)

    if not tokens.get("refresh_token"):
        sys.exit("the login yielded no refresh token; the scope must include offline_access")

    credentials = {
        "server_url": args.server_url,
        "issuer": metadata.get("issuer"),
        "token_endpoint": metadata["token_endpoint"],
        "client_id": client_id,
        "refresh_token": tokens["refresh_token"],
    }
    if not args.no_list_tools:
        tools = list_tools(args.server_url, args.path, tokens["access_token"])
        print(f"\n{len(tools)} tools on {args.server_url}:", file=sys.stderr)
        for tool in tools:
            description = (tool.get("description") or "").strip().split("\n")[0]
            print(f"  {tool.get('name')}: {description[:110]}", file=sys.stderr)
        print(file=sys.stderr)

    rendered = json.dumps(credentials, separators=(",", ":"))
    if args.out:
        with open(args.out, "w") as f:
            f.write(rendered + "\n")
        print(f"credentials written to {args.out}", file=sys.stderr)
    else:
        print(rendered)


if __name__ == "__main__":
    main()
