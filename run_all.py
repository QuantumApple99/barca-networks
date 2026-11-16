"""Run the full pipeline: stage 1 raw -> processed, stage 2 metrics, stage 3 O1 results and figures."""
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


def main():
    runpy.run_path(str(ROOT / "scripts" / "build_processed.py"), run_name="__main__")
    from barca import metrics
    metrics.run_stage2()
    runpy.run_path(str(ROOT / "notebooks" / "02_O1_style.py"), run_name="__main__")


if __name__ == "__main__":
    main()
