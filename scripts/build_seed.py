"""Build compact seed data for the preset locations (data/seed/).

Seed files are committed to the repo so a deployed app (e.g. Streamlit
Community Cloud, whose disk is wiped on restart) can show presets instantly
without calling the Open-Meteo API.

Usage (from the repo root):
    python scripts/build_seed.py                 # 1991-2025, all presets
    python scripts/build_seed.py --start 1981    # extend further back

Existing seed blocks are skipped, so after a new year completes you only
need to bump LAST_COMPLETE_YEAR in weatherplanner/data.py and re-run.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from weatherplanner import data  # noqa: E402
from weatherplanner.locations import PRESETS  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=1991)
    ap.add_argument("--end", type=int, default=data.LAST_COMPLETE_YEAR)
    args = ap.parse_args()

    data.SEED_DIR.mkdir(parents=True, exist_ok=True)
    total = 0
    for p in PRESETS:
        lat, lon = data.snap(p.lat), data.snap(p.lon)
        for y0, y1 in data._blocks(args.start, args.end):
            out = data.SEED_DIR / data.block_name(lat, lon, y0, y1)
            if not out.exists():
                # loads from data/cache if present, otherwise downloads
                block = data.load_hourly(p.lat, p.lon, y0, y1, seed_dir=Path("__none__"))
                block = block[(block.index.year >= y0) & (block.index.year <= y1)]
                if block.isna().any().any():
                    raise ValueError(f"Missing values in {out.name}; not writing seed")
                data.to_seed(block).to_parquet(out, compression="zstd", compression_level=19)
            total += out.stat().st_size
        print(f"{p.name:28s} done", flush=True)
    print(f"Seed size: {total / 1e6:.1f} MB in {data.SEED_DIR}")


if __name__ == "__main__":
    main()
