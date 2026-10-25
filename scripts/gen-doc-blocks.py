#!/usr/bin/env python3
"""Regenerate the enumerated blocks of docs/project-spec.md from the code.

Everything this script owns is a list the code already holds: the instruction-set and
project-memory routes, the chat_turn_metrics columns, the database-and-storage env vars and
their defaults, and the package's module tree. Those are the parts of a document that rot
silently — the prose around them stays plausible while the list stops matching the thing it
describes — so they are derived rather than transcribed.

A block is delimited in the doc by `<!-- BEGIN GENERATED: name -->` / `<!-- END GENERATED:
name -->`; a block with no marker, or a marker with no block, is an error, so the two cannot
drift apart silently either. Same convention as genetics-results-suite's
scripts/gen-doc-blocks.py.

    scripts/gen-doc-blocks.py            rewrite the blocks in place
    scripts/gen-doc-blocks.py --check    exit 1 if any block is stale (the test gate)

Exit 0 = up to date (or written), 1 = stale under --check, 2 = a source could not be read.

Sources are read with `ast`, `sqlite3` and `git ls-files`, never imported: an import would
read the environment (Settings' defaults are whatever the shell exports, and the test run
exports several), and a doc block must not depend on who ran the generator.
"""

import argparse
import ast
import os
import re
import sqlite3
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = "src/genetics_mcp_server"
SPEC = "docs/project-spec.md"


def harness(msg):
    print(f"HARNESS: {msg}", file=sys.stderr)
    raise SystemExit(2)


def parse(rel):
    path = os.path.join(ROOT, rel)
    try:
        with open(path) as fh:
            return ast.parse(fh.read(), filename=path)
    except OSError as exc:
        harness(f"cannot read {rel}: {exc}")


def const(node):
    return node.value if isinstance(node, ast.Constant) else None


def first_paragraph(doc):
    if not doc:
        return ""
    return " ".join(doc.strip().split("\n\n")[0].split())


# ---------------------------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------------------------

_METHODS = ("get", "post", "put", "delete", "patch")


def _router_prefixes():
    """{router module name: mount prefix}, from routers/__init__.py and chat_api.py."""
    alias_to_module = {}
    for node in parse(f"{PKG}/routers/__init__.py").body:
        if isinstance(node, ast.ImportFrom) and node.level == 1:
            for a in node.names:
                if a.name == "router" and a.asname:
                    alias_to_module[a.asname] = node.module
    prefixes = {}
    for node in ast.walk(parse(f"{PKG}/chat_api.py")):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "include_router"
            and node.args
            and isinstance(node.args[0], ast.Name)
        ):
            module = alias_to_module.get(node.args[0].id)
            prefix = next((const(k.value) for k in node.keywords if k.arg == "prefix"), "")
            if module:
                prefixes[module] = prefix or ""
    return prefixes


def _models(tree):
    """{class name: [(field, required)]} for the module's pydantic request models."""
    models = {}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        fields = []
        for stmt in node.body:
            if not (isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)):
                continue
            v = stmt.value
            if v is None:
                required = True
            elif isinstance(v, ast.Call) and getattr(v.func, "id", None) == "Field":
                default = (
                    v.args[0]
                    if v.args
                    else next((k.value for k in v.keywords if k.arg == "default"), None)
                )
                required = default is None or const(default) is Ellipsis
            else:
                required = False
            fields.append((stmt.target.id, required))
        models[node.name] = fields
    return models


def _routes(module):
    rel = f"{PKG}/routers/{module}.py"
    prefix = _router_prefixes().get(module)
    if prefix is None:
        harness(f"{rel}'s router is not mounted by chat_api.py's include_router")
    tree = parse(rel)
    models = _models(tree)
    routes = []
    for fn in tree.body:
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for dec in fn.decorator_list:
            if not (
                isinstance(dec, ast.Call)
                and isinstance(dec.func, ast.Attribute)
                and isinstance(dec.func.value, ast.Name)
                and dec.func.value.id == "router"
                and dec.func.attr in _METHODS
            ):
                continue
            kw = {k.arg: const(k.value) for k in dec.keywords}
            route = {
                "method": dec.func.attr.upper(),
                "path": prefix + const(dec.args[0]),
                "name": fn.name,
                "status": kw.get("status_code"),
                "doc": first_paragraph(ast.get_docstring(fn)),
                "body": None,
                "query": [],
                "deps": set(),
                "raises": set(),
            }
            args = fn.args.args
            defaults = [None] * (len(args) - len(fn.args.defaults)) + fn.args.defaults
            for arg, default in zip(args, defaults):
                call = default if isinstance(default, ast.Call) else None
                callee = getattr(call.func, "id", None) if call else None
                if callee == "Depends":
                    route["deps"].add(ast.unparse(call.args[0]))
                elif callee == "Query":
                    bounds = {k.arg: const(k.value) for k in call.keywords}
                    route["query"].append(
                        (
                            arg.arg,
                            const(call.args[0]) if call.args else None,
                            bounds.get("ge"),
                            bounds.get("le"),
                        )
                    )
                elif isinstance(arg.annotation, ast.Name) and arg.annotation.id in models:
                    route["body"] = models[arg.annotation.id]
            for node in ast.walk(fn):
                if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "HTTPException":
                    code = next(
                        (const(k.value) for k in node.keywords if k.arg == "status_code"),
                        const(node.args[0]) if node.args else None,
                    )
                    if isinstance(code, int):
                        route["raises"].add(code)
            routes.append(route)
    return routes


def _render_routes(routes, what):
    if not routes:
        harness(f"no route matched {what} — the selection in gen-doc-blocks.py has no subject")
    lines = []
    for r in routes:
        parts = [f"- `{r['method']} {r['path']}` (`{r['name']}`)"]
        if r["doc"]:
            parts.append(f" — {r['doc']}")
        extra = []
        if r["body"]:
            fields = ", ".join(f if req else f"{f}?" for f, req in r["body"])
            extra.append(f"body `{{{fields}}}`")
        for name, default, ge, le in r["query"]:
            if isinstance(default, bool):
                default = str(default).lower()
            q = f"`?{name}={default}`" if default is not None else f"`?{name}`"
            if ge is not None and le is not None:
                q += f" ({ge}–{le})"
            extra.append(q)
        if r["status"]:
            extra.append(f"answers **{r['status']}**")
        if r["raises"]:
            extra.append("raises " + ", ".join(str(c) for c in sorted(r["raises"])))
        if extra:
            parts.append(" " + "; ".join(extra) + ".")
        lines.append("".join(parts))
    deps = {d for r in routes for d in r["deps"]}
    if len(deps) == 1 and all(r["deps"] for r in routes):
        lines += [
            "",
            f"Every route above takes `Depends({deps.pop()})`. The status codes are "
            "the ones the handler raises itself, not those of the helpers it calls.",
        ]
    else:
        lines += [
            "",
            "Dependencies differ between these routes: "
            + "; ".join(
                f"`{r['name']}` " + (", ".join(f"`{d}`" for d in sorted(r["deps"])) or "none")
                for r in routes
            )
            + ".",
        ]
    return "\n".join(lines)


def block_instruction_set_endpoints():
    prefix = "/llm-config/user/instruction-sets"
    return _render_routes(
        [r for r in _routes("llm_config") if prefix in r["path"]],
        f"{prefix} in routers/llm_config.py",
    )


# the memory feature's routes: anything addressing a project, plus the per-session filing and
# pin toggles. A new route under /projects joins the block without anyone editing this.
_MEMORY_ROUTE = re.compile(r"/projects\b|/sessions/\{session_id\}/(project|pin)$")


def block_project_endpoints():
    return _render_routes(
        [r for r in _routes("chat_history") if _MEMORY_ROUTE.search(r["path"])],
        "the project and pin routes in routers/chat_history.py",
    )


# ---------------------------------------------------------------------------------------
# chat_turn_metrics
# ---------------------------------------------------------------------------------------


def block_turn_metrics_columns():
    """The table as a fresh database creates it, through SQLite rather than a regex."""
    rel = f"{PKG}/db/chat_history_db.py"
    ddl = None
    for node in parse(rel).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "_TURN_METRICS_COLUMNS" for t in node.targets
        ):
            ddl = const(node.value)
    if not isinstance(ddl, str):
        harness(f"{rel} has no literal _TURN_METRICS_COLUMNS")
    con = sqlite3.connect(":memory:")
    con.execute(f"CREATE TABLE chat_turn_metrics ({ddl})")
    cols = con.execute("PRAGMA table_info(chat_turn_metrics)").fetchall()
    rows = ["| column | type | constraint |", "|---|---|---|"]
    for _, name, typ, notnull, default, pk in cols:
        c = ["primary key"] if pk else ["not null"] if notnull else ["nullable"]
        if default is not None:
            c.append(f"default `{default}`")
        rows.append(f"| `{name}` | {typ} | {', '.join(c)} |")
    return "\n".join(rows)


# ---------------------------------------------------------------------------------------
# env vars
# ---------------------------------------------------------------------------------------


def _env_defaults():
    """{env var: {literal defaults}} for every `os.environ.get` in config/settings.py."""
    rel = f"{PKG}/config/settings.py"
    found = {}
    for node in ast.walk(parse(rel)):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and ast.unparse(node.func.value) == "os.environ"
            and node.args
            and isinstance(const(node.args[0]), str)
        ):
            continue
        name = node.args[0].value
        found.setdefault(name, set()).add(const(node.args[1]) if len(node.args) > 1 else None)
    return found


def _human(value, unit):
    n = int(value)
    if unit == "bytes" and n % (1024 * 1024) == 0:
        return f" ({n // (1024 * 1024)} MiB)"
    if unit == "seconds" and n % 86400 == 0:
        return f" ({n // 86400} days)"
    return ""


# the grouping and the descriptions are the doc's own judgement; what the code decides — that
# the variable exists and what it defaults to — is read from Settings.
DATABASE_ENV = [
    ("LLM_CONFIG_DB", "Path to LLM config SQLite DB", None),
    ("CHAT_HISTORY_DB", "Path to chat history SQLite DB", None),
    (
        "TURN_DRAIN_TIMEOUT_S",
        "How long shutdown waits for chat turns whose clients are gone "
        "before cancelling them; keep it inside the pod's termination grace period together with "
        "uvicorn's own drain",
        None,
    ),
    ("ATTACHMENT_STORAGE_PATH", "Path for file attachment storage", None),
    (
        "MAX_ATTACHMENT_SIZE",
        "Max attachment size in bytes; data files are further capped at "
        "`sandbox_client.MAX_INPUT_BYTES` (not configurable)",
        "bytes",
    ),
    (
        "MAX_MESSAGE_CHARS",
        "Max typed-text characters in a single user message (excludes attachments)",
        None,
    ),
    (
        "MAX_REQUEST_CHARS",
        "Cap on total text across **all** messages in one chat request, "
        "images excluded. Deliberately generous — replayed tool results are legitimately far "
        "larger than a typed message — so it bounds the payload rather than policing use",
        None,
    ),
    (
        "HISTORY_PRUNE_TRIGGER_CHARS",
        "Serialised size of the replayed history past which the "
        "oldest tool results are cleared from the model's view (see \"Replayed tool results are "
        'cleared past a budget"). About 300K tokens of JSON. `0` disables the pass',
        None,
    ),
    (
        "HISTORY_PRUNE_TARGET_CHARS",
        "What a clearing pass brings the history under. The gap "
        "below the trigger is how much growth each prompt-cache rewrite buys",
        None,
    ),
    (
        "HISTORY_PRUNE_KEEP_TURNS",
        "The newest assistant turns whose tool results are never "
        "cleared, since the model is answering a follow-up to them",
        None,
    ),
    ("MAX_MESSAGES_PER_REQUEST", "Cap on the number of messages in one chat request", None),
    (
        "MAX_ATTACHMENTS_PER_MESSAGE",
        "Max attachment blocks (image/document/data-file reference) per message",
        None,
    ),
    (
        "MAX_FILE_BLOCK_BYTES",
        "Cap in UTF-8 bytes on one `[File: ...]` reference block, in "
        "any turn; mirrored by the browser",
        None,
    ),
    ("DOWNLOAD_STORAGE_PATH", "Path for tool result download files", None),
    ("DOWNLOAD_TTL_SECONDS", "TTL for download files in seconds", "seconds"),
]


def block_database_env():
    defaults = _env_defaults()
    missing = [name for name, _, _ in DATABASE_ENV if name not in defaults]
    if missing:
        harness(
            f"config/settings.py no longer reads {missing}; drop or rename the rows in "
            f"gen-doc-blocks.py's DATABASE_ENV"
        )
    rows = ["| Variable | Description | Default |", "|----------|-------------|---------|"]
    for name, desc, unit in DATABASE_ENV:
        if len(defaults[name]) > 1:
            harness(
                f"config/settings.py reads {name} with several defaults: "
                f"{sorted(map(repr, defaults[name]))}"
            )
        (d,) = defaults[name]
        shown = "-" if d in (None, "") else f"`{d}`{_human(d, unit) if unit else ''}"
        rows.append(f"| `{name}` | {desc} | {shown} |")
    return "\n".join(rows)


# ---------------------------------------------------------------------------------------
# module tree
# ---------------------------------------------------------------------------------------


def _tracked(prefix):
    try:
        out = subprocess.run(
            ["git", "-C", ROOT, "ls-files", "--", prefix],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        harness(f"git ls-files failed: {exc}")
    files = [f[len(prefix) + 1 :] for f in out.splitlines()]
    if not files:
        harness(f"git ls-files lists nothing under {prefix}")
    return files


def _describe(rel):
    try:
        return first_paragraph(ast.get_docstring(parse(f"{PKG}/{rel}")))
    except SyntaxError as exc:
        harness(f"cannot parse {rel}: {exc}")


def block_module_tree():
    """Every tracked file of the package; a module's comment is its docstring's summary.

    Non-Python files collapse to one line per extension and directory once there is more than
    one of them, so a regenerated schema_docs/ (the suite owns its contents) does not make
    this block stale.
    """
    tree = {}
    for rel in _tracked(PKG):
        node = tree
        *dirs, name = rel.split("/")
        for d in dirs:
            node = node.setdefault(d + "/", {})
        node[name] = rel

    lines = []

    def walk(node, indent):
        py = sorted(
            (k for k, v in node.items() if isinstance(v, str) and k.endswith(".py")),
            key=lambda k: (k != "__init__.py", k),
        )
        other = {}
        for k, v in node.items():
            if isinstance(v, str) and not k.endswith(".py"):
                other.setdefault(os.path.splitext(k)[1] or k, []).append(k)
        others = sorted(v[0] if len(v) == 1 else f"*{ext}" for ext, v in other.items())
        dirs = sorted(k for k, v in node.items() if isinstance(v, dict))
        entries = (
            [(k, _describe(node[k])) for k in py]
            + [(k, "") for k in others]
            + [(k, None) for k in dirs]
        )
        for i, (name, desc) in enumerate(entries):
            last = i == len(entries) - 1
            lines.append((indent + ("└── " if last else "├── ") + name, desc or ""))
            if desc is None:
                walk(node[name], indent + ("    " if last else "│   "))

    walk(tree, "")
    width = max(len(text) for text, _ in lines) + 1
    out = ["```", "src/genetics_mcp_server/"]
    out += [f"{text.ljust(width)}# {desc}" if desc else text for text, desc in lines]
    out.append("```")
    return "\n".join(out)


BLOCKS = {
    "instruction-set-endpoints": (SPEC, block_instruction_set_endpoints),
    "project-endpoints": (SPEC, block_project_endpoints),
    "turn-metrics-columns": (SPEC, block_turn_metrics_columns),
    "database-env": (SPEC, block_database_env),
    "module-tree": (SPEC, block_module_tree),
}

MARKER = re.compile(
    r"(<!-- BEGIN GENERATED: (?P<name>[a-z-]+) -->\n)(?P<body>.*?)(<!-- END GENERATED: (?P=name) -->)",
    re.S,
)


def render(doc, text):
    seen = set()

    def repl(m):
        name = m.group("name")
        seen.add(name)
        if name not in BLOCKS or BLOCKS[name][0] != doc:
            harness(f"{doc}: unknown generated block {name!r}")
        return m.group(1) + "\n" + BLOCKS[name][1]().rstrip() + "\n\n" + m.group(4)

    out = MARKER.sub(repl, text)
    missing = {n for n, (d, _) in BLOCKS.items() if d == doc} - seen
    if missing:
        harness(f"{doc}: no marker for generated block(s) {sorted(missing)}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--check", action="store_true", help="exit 1 if any generated block is stale; write nothing"
    )
    args = ap.parse_args()

    stale = []
    for doc in sorted({d for d, _ in BLOCKS.values()}):
        path = os.path.join(ROOT, doc)
        with open(path) as fh:
            current = fh.read()
        fresh = render(doc, current)
        n = sum(1 for d, _ in BLOCKS.values() if d == doc)
        if current == fresh:
            print(f"{doc}: {n} generated block(s) up to date")
            continue
        if args.check:
            stale.append(doc)
            continue
        with open(path, "w") as fh:
            fh.write(fresh)
        print(f"{doc}: {n} generated block(s) rewritten")
    if stale:
        print("STALE, run scripts/gen-doc-blocks.py: " + ", ".join(stale), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
