"""Clause screening rubric: "issues to check", not advice.

Each CUAD clause category carries a base priority, why it matters under Indian law, a common negotiating position and, where the
issue depends on whose side you are on, a perspective note. Regex flags found in the clause text add points to check and can raise
the priority. Document-level checks cover things whose absence is the issue (stamp/registration reference, force majeure, ...).

PROVENANCE: the content of this file follows review comments supplied by the project owner on 2026-10-15 (author not stated).
The statute and case citations in them have NOT been independently verified by the developer, and a qualified lawyer's sign-off is
still required. The classifier behind the categories was trained on US commercial contracts (CUAD); Indian-specific checks
(s.28, stamp duty, arbitrator appointment, force majeure, s.12A, ...) have no trained model and run on regex only.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

LEVELS = ["low", "medium", "high"]
RULES_REVIEWED = "2026-10-15"
RULES_NOTE = (
    "Informational screening, not legal advice. Rules follow owner-supplied review comments dated 2026-10-15 "
    "(author not stated; citations not independently verified). Statutes change; check currency before relying on any point."
)

# category -> (base level, why it matters, common negotiating position, perspective note)
RUBRIC = {
    "Uncapped Liability": (
        "high", "Liability is not limited, so exposure can exceed the contract value.",
        "A cap at fees paid in the preceding 12 months is a common commercial norm, not a legal rule. Carve-outs for fraud and wilful misconduct are required, not optional: liability for fraud cannot be excluded (Contract Act ss.17, 23).",
        "The supplier or service provider carries the exposure; the customer benefits."),
    "Cap On Liability": (
        "low", "Limits liability. Fraud and wilful misconduct cannot be excluded (Contract Act ss.17, 23); confidentiality and IP breaches are usually carved out as well.",
        "Check the cap amount is proportionate and that the carve-outs are listed.", ""),
    "Liquidated Damages": (
        "medium", "Under s.74 the named sum is the ceiling: the court awards reasonable compensation up to it, whether or not actual loss is proved (Fateh Chand v Balkishan Dass, 1963). The claimant must still show a legal injury (Kailash Nath v DDA, 2015). The English penalty / liquidated-damages distinction does not apply.",
        "Draft the amount to be proportionate to foreseeable loss.",
        "The paying party can resist an excessive sum; the receiving party should not assume the full sum is recoverable."),
    "Non-Compete": (
        "medium", "Post-termination restraints are generally unenforceable (s.27; Percept D'Mark v Zaheer Khan, 2006); in-term restraints are generally valid (Niranjan Shankar Golikari, 1967). Exceptions: sale of goodwill (statutory exception to s.27) and partnership carve-outs (Partnership Act ss.11(2), 36(2), 54): reasonable non-competes in a business sale or share purchase are commonly enforced. 'Unenforceable' does not void the rest of the contract.",
        "Limit to the term of the agreement unless tied to a sale of goodwill; narrow the scope and geography.",
        "Restrained party: a post-term restraint is likely unenforceable against you. Beneficiary: do not rely on it after termination; use confidentiality, IP and in-term protections."),
    "Exclusivity": (
        "medium", "Locks a party to one counterparty. Exclusive dealing and supply can be scrutinised under the Competition Act s.3(4) (rule of reason) and s.4 (dominant parties). Post-term exclusivity falls under s.27.",
        "Add performance thresholds and an exit right if the counterparty underperforms.", ""),
    "Termination For Convenience": (
        "medium", "One-sided exit rights create uncertainty. Public-body counterparties are also subject to Article 14 (arbitrariness).",
        "Make the right mutual with reasonable notice (30-90 days) where possible.", "Check who holds the right: it favours that party."),
    "Change Of Control": (
        "medium", "A change of ownership may trigger termination or consent requirements. For listed or foreign-owned parties, the Companies Act, SEBI Takeover Regulations and FEMA may also apply.",
        "Limit to competitor acquirers and require notice rather than consent.", ""),
    "Anti-Assignment": (
        "low", "The benefit of a contract is generally assignable unless personal, but the burden is not: assigning obligations needs novation or consent (s.62).",
        "Allow assignment of benefits to affiliates and successors; state that assignment of obligations requires novation or consent.", ""),
    "Ip Ownership Assignment": (
        "high", "Copyright Act s.19: an assignment must be in writing and specify rights, duration and territory. If duration is silent it defaults to 5 years, if territory is silent to India, and rights not exercised within 1 year can lapse (s.19(4)) unless the contract says otherwise. Author's special rights (s.57) are not assignable. Patents Act s.68 requires a written, registered assignment.",
        "Limit to deliverables created under the contract; state rights, duration and territory expressly; retain background IP.", ""),
    "Irrevocable Or Perpetual License": (
        "medium", "Copyright Act s.19 defaults apply (5 years, India) unless the licence says otherwise: a licence meant to be perpetual must say so expressly.",
        "State the duration and territory expressly; make it terminable for uncured material breach.", ""),
    "Most Favored Nation": (
        "medium", "Pricing parity obligation constrains future deals; price-parity clauses have been examined by the CCI under the Competition Act.",
        "Limit to comparable volumes and a defined period.", ""),
    "Audit Rights": (
        "low", "Counterparty may inspect records. Personal data in those records is subject to the DPDP Act 2023.",
        "Limit frequency, require notice and confidentiality.", ""),
    "Governing Law": (
        "low", "Parties to a contract with a foreign element may choose foreign law; foreign law for two Indian parties is contested. Check s.28 and the jurisdiction clause (exclusive jurisdiction is valid only where the chosen court is otherwise competent: ABC Laminart).",
        "Indian law with a defined arbitration seat is a common choice for Indian parties (a commercial preference, not a legal requirement).", ""),
    "Covenant Not To Sue": (
        "medium", "An absolute restraint on enforcing rights through legal proceedings is void (s.28); narrow, defined releases are valid.",
        "Narrow to specified subject matter and claims.", ""),
    "No-Solicit Of Employees": (
        "low", "Post-term employee non-solicits face the same s.27 problem as non-competes; in-term ones are generally fine.",
        "Keep to the term of the agreement; carve out general advertisements.", ""),
    "No-Solicit Of Customers": (
        "medium", "Post-term customer non-solicits face the same s.27 problem as non-competes; in-term ones are generally fine.",
        "Limit to the term of the agreement and to customers actually served.", ""),
    "Minimum Commitment": (
        "medium", "Binds a party to a minimum purchase or spend; a shortfall payment may be tested under s.74.",
        "Add reduction rights if the counterparty cannot supply.", ""),
    "Source Code Escrow": ("low", "Escrow arrangement; confirm the release conditions.", "Define release events explicitly.", ""),
}
DEFAULT = ("low", "No specific rule for this clause type.", "", "")

# regex flags: (pattern, raises to level, label, point to check)
FLAGS = [
    (re.compile(r"\bunlimited\s+liabilit|\bwithout\s+(?:any\s+)?limit", re.I), "high", "unlimited liability wording",
     "Liability for fraud cannot be excluded or limited; check the carve-outs."),
    (re.compile(r"\bsole\s+(?:and\s+absolute\s+)?discretion", re.I), "medium", "sole-discretion wording", "One-sided discretion: check for good-faith or reasonableness limits."),
    (re.compile(r"\bwithout\s+(?:prior\s+)?notice\b", re.I), "medium", "no-notice right", "A right exercisable without notice."),
    (re.compile(r"\birrevocabl[ey]\b|\bin\s+perpetuity\b|\bperpetual\b", re.I), "medium", "irrevocable/perpetual wording",
     "May be a false positive (for example 'irrevocable letter of credit'); for licences see the Copyright Act s.19 defaults."),
    (re.compile(r"\bindemnif", re.I), "medium", "indemnity obligation", "Risk depends on direction: check whether it is one-way or mutual."),
    (re.compile(r"\bauto(?:matic(?:ally)?)?[-\s]renew", re.I), "medium", "automatic renewal", "Check the notice window to stop renewal."),
    (re.compile(r"\bshall\s+not\s+be\s+liable\b", re.I), "low", "liability exclusion", "Cannot exclude liability for fraud."),
    (re.compile(r"\bstamp\s+duty\b", re.I), "low", "stamp duty reference",
     "An unstamped or under-stamped instrument is inadmissible in evidence until duty and penalty are paid (Stamp Act s.35 and state Acts); curable after the 2023 seven-judge Bench ruling. For leases, Registration Act ss.17 and 49 also apply."),
    (re.compile(r"\bseat\s+(?:or\s+venue\s+)?of\s+(?:the\s+)?arbitration|\bvenue\s+of\s+(?:the\s+)?arbitration", re.I), "low", "arbitration seat/venue",
     "Check the seat is defined and consistent with the governing law."),
    (re.compile(r"\b(?:sole\s+)?arbitrator\s+(?:shall\s+be\s+|will\s+be\s+)?(?:appointed|nominated)\s+(?:solely\s+)?by\s+(?:the\s+)?(?:company|employer|licensor|bank|lender|purchaser|buyer|client|first\s+party|party\s+a)\b", re.I),
     "high", "unilateral arbitrator appointment", "A unilateral appointment is invalid under s.12(5) and the Seventh Schedule of the Arbitration Act (TRF v Energo; Perkins Eastman v HSCC)."),
    (re.compile(r"\b(?:no\s+(?:suit|claim|action|proceeding)s?\s+(?:shall\s+)?(?:be\s+)?(?:filed|brought|instituted|commenced)\s+(?:after|beyond|later\s+than)|waives?\s+(?:its\s+|any\s+)?right\s+to\s+(?:sue|initiate\s+(?:legal\s+)?proceedings|approach\s+(?:any\s+)?court))", re.I),
     "high", "restricts legal proceedings / shortens limitation", "Clauses shortening the limitation period or restricting legal proceedings are void (s.28)."),
    (re.compile(r"\bforfeit(?:ed|ure|s)?\b[^.]{0,80}\b(?:deposit|advance|earnest|security)|\b(?:deposit|advance|earnest|security)\b[^.]{0,80}\bforfeit", re.I),
     "medium", "forfeiture of deposit or advance", "s.74 applies: only reasonable compensation up to the stated sum can be retained."),
    (re.compile(r"\bforce\s+majeure\b", re.I), "low", "force majeure clause", "Check the listed events, notice duty and termination right; absent a clause, s.56 frustration is narrow (Energy Watchdog v CERC)."),
    (re.compile(r"\btime\s+is\s+(?:of\s+)?the\s+essence\b", re.I), "low", "time of the essence", "s.55: late performance can let the other party rescind."),
    (re.compile(r"\bpenal\s+interest\b|\binterest\s+at\s+the\s+rate\b|\blate\s+payment\s+(?:charge|interest)", re.I), "low", "interest on late payment", "Check the rate against the Interest Act 1978 and s.74."),
    (re.compile(r"\bpersonal\s+(?:data|information)\b|\bdata\s+protection\b", re.I), "medium", "personal data / data protection", "The DPDP Act 2023 applies to personal data; check purpose, consent, retention and breach notification."),
    (re.compile(r"\bmediation\b", re.I), "low", "mediation clause", "Commercial Courts Act s.12A makes pre-institution mediation mandatory for most commercial suits."),
]


@dataclass
class Risk:
    level: str
    reason: str
    suggestion: str
    flags: List[str]
    perspective: str = ""
    notes: List[str] = field(default_factory=list)


def _max(a: str, b: str) -> str:
    return a if LEVELS.index(a) >= LEVELS.index(b) else b


def assess(category: Optional[str], text: str) -> Risk:
    level, why, position, perspective = RUBRIC.get(category or "", DEFAULT)
    flags: List[str] = []
    notes: List[str] = []
    for rx, lvl, label, note in FLAGS:
        if rx.search(text):
            flags.append(label)
            notes.append(f"{label}: {note}")
            level = _max(level, lvl)
    return Risk(level, why, position, flags, perspective, notes)


def document_checks(full_text: str) -> List[str]:
    """Points to check that arise from something being ABSENT from the whole document (call with the document's full text)."""
    t = full_text.lower()
    notes: List[str] = []
    if not re.search(r"stamp|registr", t):
        notes.append("No stamp duty or registration reference found: check the instrument is duly stamped (and registered where required, e.g. leases: Registration Act ss.17, 49). An unstamped or under-stamped instrument is inadmissible in evidence until duty and penalty are paid (Stamp Act s.35 and state Acts); curable after the 2023 seven-judge Bench ruling.")
    if not re.search(r"force\s+majeure", t):
        notes.append("No force majeure clause found: without one, courts do not readily excuse performance (Energy Watchdog v CERC); s.56 frustration is narrow.")
    if re.search(r"jurisdiction|courts?\s+at\s+", t) and not re.search(r"mediation", t):
        notes.append("A court-jurisdiction clause but no mediation step: Commercial Courts Act s.12A makes pre-institution mediation mandatory for most commercial suits.")
    return notes
