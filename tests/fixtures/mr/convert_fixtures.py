"""Derive the MR replication fixtures in this directory from the published workbooks.

    uv run python -I tests/fixtures/mr/convert_fixtures.py <workbook-dir>

<workbook-dir> holds the three downloads in their per-paper subdirectories, as README.md
lists them. The workbooks are untrusted downloads, so this runs under `python -I`, evaluates
nothing (a formula cell contributes only its cached value), and refuses any workbook whose
sha256 differs from the one pinned below: a fixture is only a replication anchor if the bytes
it came from are the bytes the README names.

The sheet XML is read directly with the standard library rather than through a spreadsheet
library, because the cell's stored type is what decides how it is written, and fastexcel
loses it twice: a column mixing header text with numbers is inferred as text, and its
float-to-text cast rounds to nine decimals (a p-value of 5.8e-71 became 0); and an error cell
such as #N/A comes back null, indistinguishable from a blank.

Headers are matched by content, never by position, and a mismatch is an error: a sheet laid
out differently from the one inspected is a different table, and guessing would put the
wrong numbers under the right names.
"""

import argparse
import hashlib
import os
import posixpath
import pyexpat
import re
import sys
import xml.etree.ElementTree as ET
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))

ZHANG = (
    "zhang2025_natcommun/41467_2025_57879_MOESM3_ESM.xlsx",
    "78f66fa2c9e01cffaf2f0dad5a47fe27418397a915d125e6e5708debc7c60449",
)
ZHOU = (
    "zhou2021_natmed/41591_2021_1281_MOESM2_ESM.xlsx",
    "dff7cbfbf5f01988032c87a2698498bcef35d92294280542ffd806a7ee030e6c",
)
ZHENG = (
    "zheng2020_natgenet/41588_2020_682_MOESM3_ESM.xlsx",
    "0812c04ca38150e67d6eae1bde3863331ffd61fc0f7634e1120780edd8328a14",
)

# "1.97(1.52,2.55)"; the workbook writes no spaces, but tolerating them costs nothing
OR_CI = re.compile(r"^\s*([0-9.eE+-]+)\s*\(\s*([0-9.eE+-]+)\s*,\s*([0-9.eE+-]+)\s*\)\s*$")


class FixtureError(Exception):
    pass


MAIN = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
REL = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
PKG_REL = "{http://schemas.openxmlformats.org/package/2006/relationships}"
CELL_REF = re.compile(r"^([A-Z]+)([0-9]+)$")


def open_workbook(root, spec):
    rel, want = spec
    path = os.path.join(root, rel)
    with open(path, "rb") as fh:
        got = hashlib.sha256(fh.read()).hexdigest()
    if got != want:
        raise FixtureError(f"{rel}: sha256 {got} is not the pinned {want}")
    return zipfile.ZipFile(path)


def xml_of(book, member):
    return ET.fromstring(book.read(member))


def shared_strings(book):
    if "xl/sharedStrings.xml" not in book.namelist():
        return []
    # a rich-text string is its runs' text concatenated; phonetic guides (rPh) are not text
    return [
        "".join(t.text or "" for t in [*si.findall(f"{MAIN}t"), *si.findall(f"{MAIN}r/{MAIN}t")])
        for si in xml_of(book, "xl/sharedStrings.xml").findall(f"{MAIN}si")
    ]


def number_text(raw):
    """A numeric cell as written here: an integral value as an integer, anything else as the
    shortest text that round-trips to the stored double."""
    value = float(raw)
    if value.is_integer() and abs(value) < 2**53:
        return str(int(value))
    return repr(value)


def cell_text(cell, strings):
    kind = cell.get("t", "n")
    if kind == "inlineStr":
        return "".join(t.text or "" for t in cell.iter(f"{MAIN}t"))
    v = cell.find(f"{MAIN}v")
    if v is None or v.text is None:
        return ""
    if kind == "n":
        return number_text(v.text)
    if kind == "s":
        return strings[int(v.text)]
    if kind in ("str", "e", "d"):
        # an error cell keeps its literal (#N/A), so it never reads as a blank
        return v.text
    if kind == "b":
        return "TRUE" if v.text == "1" else "FALSE"
    raise FixtureError(f"cell {cell.get('r')}: unknown cell type {kind!r}")


def column_index(letters):
    n = 0
    for ch in letters:
        n = n * 26 + ord(ch) - 64
    return n - 1


def rows_of(book, sheet):
    """Every row of `sheet` from row 1, as equal-width tuples of strings; "" is a blank cell."""
    rels = {
        r.get("Id"): r.get("Target")
        for r in xml_of(book, "xl/_rels/workbook.xml.rels").findall(f"{PKG_REL}Relationship")
    }
    target = next(
        (
            rels[s.get(f"{REL}id")]
            for s in xml_of(book, "xl/workbook.xml").iter(f"{MAIN}sheet")
            if s.get("name") == sheet
        ),
        None,
    )
    if target is None:
        raise FixtureError(f"no sheet named {sheet!r}")
    member = target.lstrip("/") if target.startswith("/") else posixpath.join("xl", target)
    strings = shared_strings(book)
    cells = {}
    for row in xml_of(book, member).iter(f"{MAIN}row"):
        for cell in row.findall(f"{MAIN}c"):
            match = CELL_REF.match(cell.get("r", ""))
            if not match:
                raise FixtureError(f"{sheet}: cell without an A1 reference")
            key = (int(match.group(2)) - 1, column_index(match.group(1)))
            cells[key] = cell_text(cell, strings)
    if not cells:
        return []
    height = max(r for r, _ in cells) + 1
    width = max(c for _, c in cells) + 1
    return [tuple(cells.get((r, c), "") for c in range(width)) for r in range(height)]


def is_empty(row):
    return all(v.strip() == "" for v in row)


def find_row(rows, prefix, start=0, sheet=""):
    for i in range(start, len(rows)):
        if rows[i][: len(prefix)] == tuple(prefix):
            return i
    raise FixtureError(f"{sheet}: no row starting {prefix!r}")


def data_until_break(rows, start):
    """Rows from `start` to the first empty row or footnote. A footnote is a row with text in
    the first cell only, which no data row of these sheets has."""
    out = []
    for row in rows[start:]:
        if is_empty(row) or all(v == "" for v in row[1:]):
            break
        out.append(row)
    return out


def parse_or(text, where):
    if text == "":
        return "", "", ""
    match = OR_CI.match(text)
    if not match:
        raise FixtureError(f"{where}: unparsable OR (95% CI) {text!r}")
    return match.groups()


# some free-text cells wrap with an embedded newline (Zheng's protein names); a TSV cannot
# carry one, and the text is a label, not a number, so the wrap becomes a space
LINE_BREAK = re.compile(r"[\t\r\n]+")


def write_tsv(name, header, rows):
    folded = 0
    clean = []
    for row in [header, *rows]:
        if len(row) != len(header):
            raise FixtureError(f"{name}: row width {len(row)} != header width {len(header)}")
        folded += sum(1 for v in row if LINE_BREAK.search(v))
        clean.append([LINE_BREAK.sub(" ", v) for v in row])
    path = os.path.join(HERE, name)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for row in clean:
            fh.write("\t".join(row) + "\n")
    print(f"{name}: {len(rows)} rows" + (f", {folded} line-wrapped cells folded" if folded else ""))
    return path


def zhang_data8(book):
    sheet = "Supplementary Data 8"
    rows = rows_of(book, sheet)
    source = (
        "Protein",
        "SNP",
        "chr.exposure",
        "pos.exposure",
        "effect_allele.exposure",
        "other_allele.exposure",
        "eaf.exposure",
        "beta.exposure",
        "se.exposure",
        "pval.exposure",
        "samplesize.exposure",
    )
    h = find_row(rows, source, sheet=sheet)
    header = ("protein", "rsid", "chr", "pos_grch38", "ea", "oa", "eaf", "beta", "se", "p", "n")
    body = [row[: len(source)] for row in data_until_break(rows, h + 1)]
    return write_tsv("zhang2025_data8_instruments.tsv", header, body)


def zhang_data9(book):
    sheet = "Supplementary Data 9"
    rows = rows_of(book, sheet)
    h = find_row(rows, ("Proteins", "Significance in the observational study"), sheet=sheet)
    outcomes = []
    for offset in (0, 10, 20):
        outcome = rows[h][offset + 2]
        expect_top = (outcome, "", "IIBDGC", "", "FinnGen", "", "Combined", "", "")
        expect_sub = ("OR(95%CI)", "P-value") * 3 + ("FDR",)
        if (
            rows[h][offset : offset + 2] != ("Proteins", "Significance in the observational study")
            or rows[h + 1][offset + 2 : offset + 7 : 2] != expect_top[2:7:2]
            or rows[h + 2][offset + 2 : offset + 9] != expect_sub
        ):
            raise FixtureError(
                f"{sheet}: column block at offset {offset} is not laid out as inspected"
            )
        outcomes.append((outcome.lower(), offset))
    header = ["protein"]
    for o, _ in outcomes:
        header.append(f"{o}_observational")
        for study in ("iibdgc", "finngen", "combined"):
            header += [
                f"{o}_{study}_or_ci",
                f"{o}_{study}_or",
                f"{o}_{study}_ci_low",
                f"{o}_{study}_ci_high",
                f"{o}_{study}_p",
            ]
        header.append(f"{o}_fdr")
    body = []
    for n, row in enumerate(data_until_break(rows, h + 3)):
        protein = row[0]
        out = [protein]
        for o, offset in outcomes:
            if row[offset] != protein:
                raise FixtureError(
                    f"{sheet}: row {n}: the {o} block names {row[offset]!r}, not {protein!r}"
                )
            out.append(row[offset + 1])
            for k in range(3):
                text, p = row[offset + 2 + 2 * k], row[offset + 3 + 2 * k]
                out += [text, *parse_or(text, f"{sheet} {protein} {o}"), p]
            out.append(row[offset + 8])
        body.append(out)
    return write_tsv("zhang2025_data9_mr.tsv", header, body)


def zhang_data13(book):
    """Data 13, the deCODE replication: three side-by-side blocks like Data 9's, except that
    each block lists its own proteins (IBD, CD and UC differ in length and in order), so the
    blocks are read one at a time and stacked long, one row per outcome and protein."""
    sheet = "Supplementary Data 13"
    rows = rows_of(book, sheet)
    h = find_row(rows, ("Proteins",), sheet=sheet)
    studies = ("iibdgc", "finngen", "combined")
    header = ["outcome", "protein"]
    for study in studies:
        header += [
            f"{study}_or_ci",
            f"{study}_or",
            f"{study}_ci_low",
            f"{study}_ci_high",
            f"{study}_p",
        ]
    body = []
    for offset in (0, 8, 16):
        outcome = rows[h][offset + 1]
        if (
            rows[h][offset] != "Proteins"
            or outcome not in ("IBD", "CD", "UC")
            or rows[h + 1][offset + 1 : offset + 6 : 2] != ("IIBDGC", "FinnGen", "Combined")
            or rows[h + 2][offset + 1 : offset + 7] != ("OR(95%CI)", "P-value") * 3
        ):
            raise FixtureError(
                f"{sheet}: column block at offset {offset} is not laid out as inspected"
            )
        block = [row[offset : offset + 7] for row in rows[h + 3 :]]
        data = data_until_break(block, 0)
        # past the break a block may hold its footnote and nothing else
        if any(not is_empty(row) and any(row[1:]) for row in block[len(data) :]):
            raise FixtureError(f"{sheet}: the {outcome} block has data after its first break")
        if len({row[0] for row in data}) != len(data):
            raise FixtureError(f"{sheet}: the {outcome} block names a protein twice")
        for row in data:
            out = [outcome.lower(), row[0]]
            for k in range(3):
                text, p = row[1 + 2 * k], row[2 + 2 * k]
                out += [text, *parse_or(text, f"{sheet} {row[0]} {outcome}"), p]
            body.append(out)
    return write_tsv("zhang2025_data13_decode_mr.tsv", header, body)


def zhou_st2(book):
    sheet = "Supplementary Table 2"
    rows = rows_of(book, sheet)
    source = (
        "SOMAmers",
        "Method",
        "#SNP",
        "Beta",
        "Se",
        "P value",
        "P FDR",
        "OR",
        "95% CI lower bound",
        "95% CI upper bound",
    )
    h = find_row(rows, source, sheet=sheet)
    header = ("somamer", "method", "n_snp", "beta", "se", "p", "p_fdr", "or", "ci_low", "ci_high")
    body = [row[: len(source)] for row in data_until_break(rows, h + 1)]
    oas1 = [r for r in body if r[0] == "OAS1.10361.25.3"]
    # the anchor the investigation quotes; a different number means a different sheet
    if len(oas1) != 1 or not oas1[0][7].startswith("0.544"):
        raise FixtureError(f"{sheet}: OAS1.10361.25.3 is not the single row with OR 0.544: {oas1}")
    return write_tsv("zhou2021_st2_somamer_a2.tsv", header, body)


def dedupe(names):
    seen, out = {}, []
    for name in names:
        seen[name] = seen.get(name, 0) + 1
        out.append(name if seen[name] == 1 else f"{name}_{seen[name]}")
    return tuple(out)


def zheng_s1(book):
    sheet = "TableS1.Instruments"
    rows = rows_of(book, sheet)
    h = find_row(rows, ("Exposure", "Exposure_name", "Entrez_id"), sheet=sheet)
    width = max(i for i, v in enumerate(rows[h]) if v != "") + 1
    header = dedupe(rows[h][:width])
    body = [row[:width] for row in data_until_break(rows, h + 1)]
    return write_tsv("zheng2020_s1_instruments.tsv", header, body)


def zheng_s9(book):
    sheet = "TableS9.multi-cis-top"
    rows = rows_of(book, sheet)
    source = (
        "Exposure",
        "N_instrument",
        "Outcome_id",
        "beta_trans",
        "Outcome",
        "Method",
        "Multi_cis_beta",
        "Multi_cis_se",
        "Multi_cis_P",
        "Single_cis_beta",
        "Single_cis_P",
        "Q",
        "Q_pval",
    )
    body, start = [], 0
    for table in ("9A", "9B"):
        # the caption cell carries the whole caption, so it is matched on its prefix
        title = next(
            (
                i
                for i in range(start, len(rows))
                if rows[i][0].startswith(f"Supplementary Table {table}.")
            ),
            None,
        )
        if title is None:
            raise FixtureError(f"{sheet}: no caption for table {table}")
        h = find_row(rows, source, title, sheet)
        block = data_until_break(rows, h + 1)
        body += [(table, *row[: len(source)]) for row in block]
        start = h + 1 + len(block)
    return write_tsv("zheng2020_s9_multicis.tsv", ("table", *source), body)


def check_expat():
    # expat guards against entity-expansion bombs from 2.4.0 on; older, an untrusted
    # workbook's XML could exhaust memory before anything here sees it
    if pyexpat.version_info < (2, 4, 0):
        raise FixtureError(f"expat {pyexpat.EXPAT_VERSION} predates entity-expansion limits")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("workbook_dir")
    args = parser.parse_args()
    root = os.path.abspath(args.workbook_dir)
    try:
        check_expat()
        zhang = open_workbook(root, ZHANG)
        outputs = [
            zhang_data8(zhang),
            zhang_data9(zhang),
            zhang_data13(zhang),
            zhou_st2(open_workbook(root, ZHOU)),
        ]
        zheng = open_workbook(root, ZHENG)
        outputs += [zheng_s1(zheng), zheng_s9(zheng)]
    except (FixtureError, OSError, zipfile.BadZipFile, ET.ParseError) as exc:
        print(f"convert_fixtures: {exc}", file=sys.stderr)
        return 1
    with open(os.path.join(HERE, "SHA256SUMS"), "w", encoding="utf-8", newline="\n") as fh:
        for path in sorted(outputs):
            with open(path, "rb") as f:
                fh.write(f"{hashlib.sha256(f.read()).hexdigest()}  {os.path.basename(path)}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
