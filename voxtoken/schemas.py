from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

BoxMM = Tuple[float, float, float, float, float, float]  # x0,x1,y0,y1,z0,z1


@dataclass
class Token:
    token_id: int
    level: int
    omega_box_mm: BoxMM
    parent_id: Optional[int] = None
    children_ids: List[int] = field(default_factory=list)

    # representation
    code: Optional[int] = None  # VQ token id (discrete); None for continuous MVP
    embed_ref: Optional[int] = None  # index into a tensor bank on disk (np.memmap/pt)


@dataclass
class Citation:
    sent_id: int
    cited_token_ids: List[int]


@dataclass
class EvidenceNode:
    eid: str
    finding_type: str  # e.g., nodule/effusion/atelectasis/normal...
    attrs: Dict[str, Any]  # side, location, size_bin, certainty, negation...
    supported_token_ids: List[int]
    optional_mask_ref: Optional[str] = None  # path to mask in npy/nii.gz
    optional_measure: Optional[Dict[str, float]] = None  # diameter/volume...


IssueType = Literal["missing_slot", "inconsistency", "overclaim", "unsupported"]


@dataclass
class Issue:
    type: IssueType
    span: Tuple[int, int]  # char span in report, or (sent_id, token_span)
    reason: str
    related_eids: List[str] = field(default_factory=list)
    related_tokens: List[int] = field(default_factory=list)


@dataclass
class TraceStep:
    k: int
    budget_total: int
    budget_used: int
    split_token_ids: List[int]
    added_token_ids: List[int]
    verifier_score_before: float
    verifier_score_after: float
    latency_ms: Dict[str, float]  # tokenize/generate/verify/total


@dataclass
class FactSlot:
    finding_type: str
    side: str
    location: str
    size_bin: str
    certainty: str
    supported_token_ids: List[int]


@dataclass
class ReportPlan:
    facts: List[FactSlot]
    impression: List[FactSlot]  # or derived summary facts


@dataclass
class TokenFeatures:
    token_id: int
    level: int
    recon_error: float
    evidence_entropy: float
    citation_pressure: float
    history_splits: int
    center_x_mm: float = 0.0
    center_y_mm: float = 0.0
    center_z_mm: float = 0.0
    mean_intensity: float = 0.0
    max_intensity: float = 0.0


__all__ = [
    "BoxMM",
    "Citation",
    "EvidenceNode",
    "FactSlot",
    "Issue",
    "IssueType",
    "ReportPlan",
    "Token",
    "TokenFeatures",
    "TraceStep",
]
