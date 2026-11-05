"""Run the full pipeline. Only stage 1 (raw -> processed) exists so far."""
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    runpy.run_path(str(ROOT / "scripts" / "build_processed.py"), run_name="__main__")


if __name__ == "__main__":
    main()
