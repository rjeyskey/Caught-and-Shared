"""Script 01 — Embedding Blending + per-category α on MovieLens (thin adapter).

Realizes the LensOperator seam (ADR-0002). The heavy lifting lives in
src/cs_core/pipeline.py; this script is a thin CLI wrapper:

  python scripts/01_embedding_blending_movielens.py [--alpha 0.2,0.45,0.7] [--debug-users N]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from cs_core.pipeline import run_blend_experiment  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=REPO / "ml-latest-small")
    ap.add_argument("--out", type=Path, default=REPO / "data/test_results/blend_movielens.json")
    ap.add_argument("--alpha", type=str, default="0.2,0.45,0.7")
    ap.add_argument("--n-test-users", type=int, default=120)
    ap.add_argument("--debug-users", type=int, default=None, help="Use first N warm users (replaces old 01b)")
    args = ap.parse_args()

    alphas = [float(x) for x in args.alpha.split(",")]
    results = run_blend_experiment(
        data_dir=args.data,
        alpha_values=alphas,
        n_test_users=args.n_test_users,
        debug_users=args.debug_users,
    )

    by_alpha = defaultdict(list)
    for r in results:
        by_alpha[r.alpha].append(r)

    summary = {}
    for a, rows in by_alpha.items():
        if not rows:
            summary[f"alpha_{a}"] = {"n_users": 0}
            continue

        def avg(key):
            return float(sum(r.deltas()[key] for r in rows)) / len(rows)

        summary[f"alpha_{a}"] = {
            "n_users": len(rows),
            "ndcg10_delta_mean": avg("delta_ndcg@10"),
            "recall10_delta_mean": avg("delta_recall@10"),
            "coverage_delta_mean": avg("delta_category_coverage"),
            "novelty_delta_mean": avg("delta_novelty"),
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "alphas": alphas}, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"WROTE {args.out}")


if __name__ == "__main__":
    main()