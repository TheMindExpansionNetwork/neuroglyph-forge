"""Pytest hooks for adaptive engine signals."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def pytest_sessionfinish(session, exitstatus):
    marker = ROOT / "tests" / ".last_pytest_ok"
    marker.write_text("1" if exitstatus == 0 else "0", encoding="utf-8")