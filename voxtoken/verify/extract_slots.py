from __future__ import annotations

from typing import List

from ..schemas import FactSlot


def extract_slots_from_report(report_text: str) -> List[FactSlot]:
    """Extract structured slot tuples from a free-form report text."""
    slots: List[FactSlot] = []
    for line in report_text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.lower().startswith("no findings"):
            continue

        finding = s
        side = "U"
        location = "U"
        size_bin = "U"
        certainty = "U"

        open_i = s.find("(")
        close_i = s.find(")")
        if open_i >= 0:
            finding = s[:open_i].strip()
            inside = s[open_i + 1 : close_i if close_i > open_i else len(s)]
            parts = [p.strip() for p in inside.split(",") if p.strip()]
            kv = {}
            for p in parts:
                if "=" not in p:
                    continue
                k, v = p.split("=", 1)
                kv[k.strip()] = v.strip().strip(".")

            side = str(kv.get("side", side))
            location = str(kv.get("location", location))
            size_bin = str(kv.get("size", kv.get("size_bin", size_bin)))
            certainty = str(kv.get("certainty", certainty))

        slots.append(
            FactSlot(
                finding_type=str(finding) if finding else "U",
                side=side,
                location=location,
                size_bin=size_bin,
                certainty=certainty,
                supported_token_ids=[],
            )
        )

    return slots
