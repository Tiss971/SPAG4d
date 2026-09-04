import os
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(autouse=True, scope="session")
def _deterministic_env():
    os.environ["SPAG_DETERMINISTIC"] = "1"
    yield
