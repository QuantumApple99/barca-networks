import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from barca import parse  # noqa: E402


@pytest.fixture(scope="session")
def tables():
    res = [parse.parse_match(m) for m in (69276, 69299, 267183)]
    return parse.assemble(res)
