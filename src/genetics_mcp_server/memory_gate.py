"""The single place that decides whether a caller gets cross-session chat memory.

The chat turn (chat_api._resolve_user_memory) and the memory dialog's endpoint
(routers/chat_history.get_project_memory) must agree on what "on" means, so the setting is read
here and nowhere else: a second reader is a second definition of the opt-in, and the two
would drift the moment either side gained a default.
"""

import hashlib

from genetics_mcp_server.auth.core import SERVICE_IDENTITY
from genetics_mcp_server.db import get_llm_config_db

# written through the generic PUT /chat/v1/llm-config/user/settings/{key} route; memory
# has no settings route of its own
MEMORY_SETTING_KEY = "chat_memory"
MEMORY_SETTING_ON = "on"

# how far back the digest looks INSIDE one project. Pinned sessions filed in the same
# project come back regardless of this window. It stays a recency window rather than the
# whole project because the premise measurement's window-miss share was borderline at the
# epic's own ~10% line, and at that measurement only 3 of 149 users held more than 20
# sessions at all, so a larger cap bought nothing measurable. Raise it when a user's
# project outgrows the window and the digest starts missing the session they mean.
MEMORY_PROJECT_SESSION_CAP = 20

_ANONYMOUS = "anonymous"


def is_identifiable_user(user: str | None) -> bool:
    """Whether `user` names a person rather than a service or an unauthenticated caller.

    `anonymous` is what auth_required returns with REQUIRE_AUTH off: it is shared by every
    local caller, so treating it as a person would pool strangers' sessions into one memory.
    """
    return bool(user) and user not in (SERVICE_IDENTITY, _ANONYMOUS)


def memory_setting_on(user: str) -> bool:
    """Whether this user opted in. Absent or any other value means no."""
    setting = get_llm_config_db().get_user_setting(user, MEMORY_SETTING_KEY)
    return setting is not None and setting.setting_value == MEMORY_SETTING_ON


def memory_gate_open(user: str | None, *, gateway_asserted: bool, secret: bool) -> bool:
    """Whether a chat turn may be seeded with this user's memory.

    Each condition is independently sufficient to withhold it. `gateway_asserted` is the
    read_artifact rule: the digest is private content, so the identity it is keyed to has
    to be one auth-gateway just authenticated, not one any INTERNAL_API_SECRET holder
    typed into a header. Instruction sets do not need it — the caller supplies the id of a
    set they must already own — but memory is resolved from the identity alone.
    """
    if secret or not gateway_asserted or not is_identifiable_user(user):
        return False
    return memory_setting_on(user)


def _log_hash(value: str) -> str:
    return hashlib.sha256(value.strip().lower().encode()).hexdigest()[:12]


def user_log_hash(user: str) -> str:
    """A stable pseudonym, so the memory log line can name a user without the address.

    Other logs here do carry the raw address; this one is different because it reports on
    a read across the user's whole history, and it is emitted on every first turn.
    """
    return _log_hash(user)


def project_log_hash(project_id: str) -> str:
    """The same pseudonym scheme for a project id, so one log line can correlate a user's
    turns within a project without naming either the person or the project."""
    return _log_hash(project_id)
