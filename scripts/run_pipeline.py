from __future__ import annotations
import sys
from pathlib import Path
import argparse
import json

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from gs_match_ai.ingest import load_l2_csv
from gs_match_ai.stats import compute_match_stats
from gs_match_ai.candidates import select_candidate_points

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default=str(ROOT / "data" / "processed"))
    args = ap.parse_args()

    df = load_l2_csv(args.csv)
    evidence = compute_match_stats(df)
    candidates = select_candidate_points(df)

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "evidence_packet.json").write_text(json.dumps(evidence, indent=2, ensure_ascii=False))
    (outdir / "candidate_points.json").write_text(json.dumps(candidates, indent=2, ensure_ascii=False))
    print(f"Wrote artifacts to {outdir}")

if __name__ == "__main__":
    main()
