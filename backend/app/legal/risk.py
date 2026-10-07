"""Clause risk rubric (DRAFT: needs review by a qualified lawyer before any real use).

Each CUAD clause category carries a base risk and a suggested edit; regex red-flags found in the
clause text can raise the level and are quoted back as the reason. This is a transparent rule layer
on top of the clause classifier, not legal advice. Edit RUBRIC to change policy; unit-tested.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

LEVELS = ["low", "medium", "high"]

# category -> (base level, why it matters, suggested alternative)
RUBRIC = {
    "Uncapped Liability": ("high", "Liability is not limited, so exposure can exceed contract value.", "Cap liability at fees paid in the preceding 12 months, with narrow carve-outs."),
    "Liquidated Damages": ("medium", "Fixed damages may be disproportionate; under s.74 Indian Contract Act courts allow only reasonable compensation.", "Tie the amount to a genuine pre-estimate of loss and add a cap."),
    "Non-Compete": ("high", "Post-termination restraints are void under s.27 Indian Contract Act; in-term restraints must be reasonable.", "Limit to the term of the agreement, narrow scope and geography."),
    "Exclusivity": ("medium", "Locks a party to one counterparty and limits commercial flexibility.", "Add performance thresholds and an exit right if the counterparty underperforms."),
    "Termination For Convenience": ("medium", "One-sided exit rights create uncertainty.", "Make the right mutual with reasonable notice (30-90 days)."),
    "Change Of Control": ("medium", "A change of ownership may trigger termination or consent requirements.", "Limit to competitor acquirers and require notice rather than consent."),
    "Anti-Assignment": ("low", "Restricts transfer of rights; usually standard.", "Allow assignment to affiliates and successors on merger."),
    "Ip Ownership Assignment": ("high", "Transfers IP ownership; confirm scope, consideration and moral-rights position.", "Limit to deliverables created under the contract; retain background IP."),
    "Irrevocable Or Perpetual License": ("medium", "Licence survives termination.", "Make the licence terminable for uncured material breach."),
    "Most Favored Nation": ("medium", "Pricing parity obligation constrains future deals.", "Limit to comparable volumes and a defined period."),
    "Audit Rights": ("low", "Counterparty may inspect records.", "Limit frequency, require notice and confidentiality."),
    "Cap On Liability": ("low", "Limits liability; check carve-outs for fraud, IP and confidentiality.", "Confirm the cap amount is proportionate."),
    "Governing Law": ("low", "Check the chosen law and forum fit the parties.", "For Indian parties prefer Indian law with a defined arbitration seat."),
    "Covenant Not To Sue": ("medium", "Waives the right to bring claims.", "Narrow to specified subject matter."),
    "No-Solicit Of Employees": ("low", "Restricts hiring.", "Time-limit to 12 months and carve out general advertisements."),
    "No-Solicit Of Customers": ("medium", "Restricts dealing with customers.", "Limit to customers actually served by the party."),
    "Minimum Commitment": ("medium", "Binds a party to minimum purchase/spend.", "Add reduction rights if the counterparty cannot supply."),
    "Source Code Escrow": ("low", "Escrow arrangement; confirm release conditions.", "Define release events explicitly."),
}
DEFAULT = ("low", "No specific risk rule for this clause type.", "")

# regex red flags: (pattern, raises to level, label)
FLAGS = [
    (re.compile(r"\bunlimited\s+liabilit|\bwithout\s+(?:any\s+)?limit", re.I), "high", "unlimited liability wording"),
    (re.compile(r"\bsole\s+(?:and\s+absolute\s+)?discretion", re.I), "medium", "sole-discretion wording"),
    (re.compile(r"\bwithout\s+(?:prior\s+)?notice\b", re.I), "medium", "no-notice right"),
    (re.compile(r"\birrevocabl[ey]\b|\bin\s+perpetuity\b|\bperpetual\b", re.I), "medium", "irrevocable/perpetual wording"),
    (re.compile(r"\bindemnif", re.I), "medium", "indemnity obligation"),
    (re.compile(r"\bauto(?:matic(?:ally)?)?[-\s]renew", re.I), "medium", "automatic renewal"),
    (re.compile(r"\bshall\s+not\s+be\s+liable\b", re.I), "low", "liability exclusion"),
    (re.compile(r"\bstamp\s+duty\b", re.I), "low", "stamp duty allocation (check Indian Stamp Act / state law)"),
    (re.compile(r"\bseat\s+of\s+arbitration\b|\bvenue\b", re.I), "low", "arbitration seat/venue"),
]


@dataclass
class Risk:
    level: str
    reason: str
    suggestion: str
    flags: List[str]


def _max(a: str, b: str) -> str:
    return a if LEVELS.index(a) >= LEVELS.index(b) else b


def assess(category: Optional[str], text: str) -> Risk:
    level, why, fix = RUBRIC.get(category or "", DEFAULT)
    flags: List[str] = []
    for rx, lvl, label in FLAGS:
        if rx.search(text):
            flags.append(label)
            level = _max(level, lvl)
    reason = why + (f" Red flags: {', '.join(flags)}." if flags else "")
    return Risk(level, reason, fix, flags)
