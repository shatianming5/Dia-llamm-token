from __future__ import annotations

from typing import List

from ..schemas import FactSlot


def extract_slots_from_report(report_text: str) -> List[FactSlot]:
    """Extract structured slot tuples from a free-form report text."""
    raise NotImplementedError

