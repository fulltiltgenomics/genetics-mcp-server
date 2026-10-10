"""Synthetic polygenic exposure-outcome table with a known causal effect and a seeded
pleiotropic subset. Standard library only, so any Python 3 reproduces it byte for byte.

Model, per variant j: a_j ~ U(0.05, 0.5), se_x = 1/sqrt(2 a (1-a) n_x) (SD units),
true bx_j = s_j * U(0.03, 0.12) with a random sign s_j, beta_x = bx_j + N(0, se_x);
true by_j = THETA * bx_j + alpha_j, alpha_j ~ N(ALPHA_MEAN, ALPHA_SD) on the pleiotropic
subset (directional, independent of bx so InSIDE holds) and 0 elsewhere; beta_y = by_j + N(0, se_y).
"""

import csv
import math
import os
import random

SEED = 20261010
K, K_PLEIO = 50, 10
THETA = 0.3
ALPHA_MEAN, ALPHA_SD = 0.02, 0.01
N_X, N_Y = 30000, 50000


def main():
    rng = random.Random(SEED)
    pleio = set(rng.sample(range(K), K_PLEIO))
    rows = []
    for j in range(K):
        af = rng.uniform(0.05, 0.5)
        het = 2 * af * (1 - af)
        se_x, se_y = 1 / math.sqrt(het * N_X), 1 / math.sqrt(het * N_Y)
        bx = rng.choice((-1, 1)) * rng.uniform(0.03, 0.12)
        alpha = rng.gauss(ALPHA_MEAN, ALPHA_SD) if j in pleio else 0.0
        beta_x = bx + rng.gauss(0, se_x)
        beta_y = THETA * bx + alpha + rng.gauss(0, se_y)
        g = lambda x: f"{x:.10g}"  # noqa: E731
        rows.append({"variant": f"syn{j + 1:02d}", "beta_x": g(beta_x), "se_x": g(se_x),
                     "beta_y": g(beta_y), "se_y": g(se_y), "af_x": g(af), "af_y": g(af),
                     "n_x": N_X, "n_y": N_Y, "pleiotropic": int(j in pleio), "alpha_true": g(alpha)})
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "input_synthetic.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
