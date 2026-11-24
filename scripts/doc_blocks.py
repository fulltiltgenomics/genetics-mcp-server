"""The marked-block mechanism the gen-doc-blocks.py generators share.

A generated block sits between `<!-- BEGIN GENERATED: name -->` and `<!-- END GENERATED: name -->`
in a doc. Which blocks exist and what each derives from is each generator's own business;
finding the markers and refusing a doc whose markers and blocks disagree is the same
everywhere, and a generator that got that part subtly different would pass a stale doc.

One file for every repo that has a generator: genetics-results-suite's copy is the source
and scripts/vendor.py writes the others, so edit it there.
"""

import re

MARKER = re.compile(
    r"(<!-- BEGIN GENERATED: (?P<name>[a-z-]+) -->\n)(?P<body>.*?)(<!-- END GENERATED: (?P=name) -->)",
    re.S,
)


def render(doc, text, blocks, fail, skip=frozenset()):
    """`text` with every marked block regenerated.

    `blocks` maps a block name to (doc it belongs to, function returning its body). `fail`
    takes a message and must not return. A name in `skip` must still be a known block of
    this doc, and its current body is kept as it is.
    """
    seen = set()

    def repl(m):
        name = m.group("name")
        seen.add(name)
        if name not in blocks or blocks[name][0] != doc:
            fail(f"{doc}: unknown generated block {name!r}")
        if name in skip:
            return m.group(0)
        return m.group(1) + "\n" + blocks[name][1]().rstrip() + "\n\n" + m.group(4)

    out = MARKER.sub(repl, text)
    missing = {n for n, (d, _) in blocks.items() if d == doc} - seen
    if missing:
        fail(f"{doc}: no marker for generated block(s) {sorted(missing)}")
    return out
