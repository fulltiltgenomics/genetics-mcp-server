"""Fetch the cis input tables and their signed-r matrices from a running stack, once.

Exposure: Zhang 2025 Data 8 cis instruments (../zhang2025_data8_instruments.tsv) for PROTEINS.
Outcome: the IIBDGC IBD rows of resource ibd_gwas. LD: finngen_r12 signed r, one pair call per
instrument pair. Harmonisation is scripts/mr-proving-ground.py's rule: effect allele == the
outcome row's alt keeps beta, == its ref flips beta and eaf, anything else drops the instrument.

Run from the genetics-mcp-server checkout with the stack env (README.md has the command).
"""

import csv
import os
import sys

from genetics_mcp_server import sdk as genetics

HERE = os.path.dirname(os.path.abspath(__file__))
PROTEINS = ["IL12B", "CD6", "IL18", "RSPO3", "CD40", "CXCL10", "FAS", "CD274", "CD55", "ANGPTL3"]
OUTCOME_RESOURCE = "ibd_gwas"
OUTCOME_TRAIT = "Inflammatory bowel disease"
PANEL = "finngen_r12"


def fmt(x):
    return "" if x is None else repr(float(x))


def main():
    with open(os.path.join(HERE, "..", "zhang2025_data8_instruments.tsv"), newline="") as fh:
        instruments = [r for r in csv.DictReader(fh, delimiter="\t") if r["protein"] in PROTEINS]
    missing = set(PROTEINS) - {r["protein"] for r in instruments}
    if missing:
        sys.exit(f"no Data 8 instruments for {sorted(missing)}")

    resolved = genetics.search(rsids=sorted({r["rsid"] for r in instruments})).to_dicts()
    by_rsid = {r["rsid"]: r["variants"] or [] for r in resolved}
    for inst in instruments:
        alleles = {inst["ea"], inst["oa"]}
        hits = [v for v in by_rsid.get(inst["rsid"], []) if set(v.split(":")[2:]) == alleles]
        base = f"{inst['chr']}:{inst['pos_grch38']}"
        inst["candidates"] = hits or [f"{base}:{inst['oa']}:{inst['ea']}", f"{base}:{inst['ea']}:{inst['oa']}"]

    pheno = {r["trait_name"]: r for r in genetics.phenotypes(resource=OUTCOME_RESOURCE).to_dicts()}
    code = pheno[OUTCOME_TRAIT]["trait_original"]
    variants = sorted({v for i in instruments for v in i["candidates"]})
    rows = genetics.summary_stats(phenotypes=[code], variants=variants, resource=OUTCOME_RESOURCE).to_dicts()
    index = {f"{r['chr']}:{r['pos']}:{r['ref']}:{r['alt']}": r for r in rows if r["phenotype"] == code}
    print(f"outcome {OUTCOME_RESOURCE}/{code}: {len(index)} rows for {len(variants)} candidate ids")

    table, kept = [], {p: [] for p in PROTEINS}
    for inst in instruments:
        row = next((index[v] for v in inst["candidates"] if v in index), None)
        if row is None:
            print(f"DROP {inst['protein']} {inst['rsid']}: no outcome row")
            continue
        beta, eaf = float(inst["beta"]), float(inst["eaf"])
        if (inst["ea"], inst["oa"]) == (row["alt"], row["ref"]):
            pass
        elif (inst["ea"], inst["oa"]) == (row["ref"], row["alt"]):
            beta, eaf = -beta, 1 - eaf
        else:
            print(f"DROP {inst['protein']} {inst['rsid']}: alleles {inst['ea']}/{inst['oa']} vs row {row['ref']}/{row['alt']}")
            continue
        variant = f"{row['chr']}:{row['pos']}:{row['ref']}:{row['alt']}"
        if row.get("af") is not None and abs(eaf - row["af"]) > 0.2:
            print(f"NOTE {inst['protein']} {variant}: af_x {eaf:.3f} vs af_y {row['af']:.3f}")
        kept[inst["protein"]].append(variant)
        table.append({
            "protein": inst["protein"], "variant": variant, "rsid": inst["rsid"],
            "beta_x": fmt(beta), "se_x": fmt(inst["se"]), "beta_y": fmt(row["beta"]), "se_y": fmt(row["se"]),
            "af_x": fmt(eaf), "af_y": fmt(row.get("af")), "n_x": inst["n"],
            "n_y": int(row["n_cases"]) + int(row["n_controls"]),
            "ncase_y": int(row["n_cases"]), "ncontrol_y": int(row["n_controls"]),
        })
    with open(os.path.join(HERE, "input_cis_ibd.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(table)

    for protein, vs in kept.items():
        in_panel, r = [], {}
        for v in vs:
            try:
                genetics.ld(v, v, panel=PANEL, fields="r", r2_threshold=0)
                in_panel.append(v)
            except Exception as exc:
                print(f"LD DROP {protein} {v}: {type(exc).__name__}: {str(exc)[:200]}")
        for i, a in enumerate(in_panel):
            for b in in_panel[i + 1:]:
                res = genetics.ld(a, b, panel=PANEL, fields="r", r2_threshold=0).to_dicts()
                val = res[0].get("r") if res and res[0].get("in_ld") else None
                r[(a, b)] = r[(b, a)] = 0.0 if val is None else float(val)
        with open(os.path.join(HERE, f"ld_{protein}.csv"), "w", newline="") as fh:
            w = csv.writer(fh, lineterminator="\n")
            w.writerow(["variant"] + in_panel)
            for a in in_panel:
                w.writerow([a] + [repr(1.0) if a == b else repr(r[(a, b)]) for b in in_panel])
        print(f"{protein}: {len(vs)} harmonised, {len(in_panel)} in {PANEL}")


if __name__ == "__main__":
    main()
