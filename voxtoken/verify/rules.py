from __future__ import annotations

from typing import List

from ..schemas import Citation, Issue, ReportPlan


def check_missing_slots(plan: ReportPlan, report_text: str) -> List[Issue]:
    raise NotImplementedError


def check_inconsistency(plan: ReportPlan, report_text: str) -> List[Issue]:
    raise NotImplementedError


def check_overclaim(plan: ReportPlan, report_text: str) -> List[Issue]:
    raise NotImplementedError


def check_unsupported(report_text: str, citations: List[Citation], plan: ReportPlan) -> List[Issue]:
    raise NotImplementedError

