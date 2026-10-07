"""Synthetic legal-document *content* — text, clause tables and citations.

All content below is written from scratch for this project in realistic legal
drafting style. It paraphrases well-known general legal concepts (contract
formation, termination, confidentiality, tort liability, statutory rights)
rather than reproducing any real, copyrighted, or jurisdiction-specific
statute/judgment verbatim. It exists to give the RAG pipeline a large, varied,
genuinely multimodal legal corpus to index — not as a source of real legal
authority (see the disclaimer baked into every generated document and into
the system prompt).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class Contract:
    title: str
    doc_type: str  # e.g. "Non-Disclosure Agreement"
    parties: List[Tuple[str, str]]  # (role, name)
    effective_date: str
    clauses: List[Tuple[str, str]]  # (heading, body)
    key_terms: List[Tuple[str, str]]  # table rows (term, value)


@dataclass
class CaseJudgment:
    case_name: str
    citation: str
    court: str
    date: str
    judge: str
    outcome: str
    facts: str
    issues: List[str]
    holding: str
    reasoning: str


@dataclass
class StatuteSection:
    act_name: str
    section_no: str
    heading: str
    body: str


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------
_STANDARD_CLAUSES = [
    ("Definitions", "In this Agreement, unless the context otherwise requires, capitalised "
     "terms shall have the meanings assigned to them in this Agreement, and \"Confidential "
     "Information\" means any non-public information disclosed by one Party to the other, "
     "whether orally, in writing, or in any other form, that is designated as confidential "
     "or that a reasonable person would understand to be confidential given the nature of "
     "the information and the circumstances of disclosure."),
    ("Term and Termination", "This Agreement shall commence on the Effective Date and shall "
     "continue in force until terminated in accordance with this Clause. Either Party may "
     "terminate this Agreement for convenience upon thirty (30) days' prior written notice "
     "to the other Party. Either Party may terminate this Agreement immediately upon written "
     "notice if the other Party commits a material breach of this Agreement that remains "
     "uncured for fifteen (15) days after written notice of such breach."),
    ("Confidentiality", "Each Party agrees to hold the other Party's Confidential Information "
     "in strict confidence and not to disclose such Confidential Information to any third "
     "party without the prior written consent of the disclosing Party, except to employees, "
     "agents or advisors who need to know such information and who are bound by "
     "confidentiality obligations no less protective than those set out herein. This "
     "obligation shall survive termination of this Agreement for a period of three (3) years."),
    ("Indemnification", "Each Party (the \"Indemnifying Party\") shall indemnify, defend and "
     "hold harmless the other Party from and against any and all claims, damages, losses, "
     "liabilities, costs and expenses (including reasonable attorneys' fees) arising out of "
     "or relating to the Indemnifying Party's breach of any representation, warranty or "
     "covenant contained in this Agreement, or the Indemnifying Party's gross negligence or "
     "wilful misconduct."),
    ("Limitation of Liability", "EXCEPT FOR BREACHES OF CONFIDENTIALITY OR INDEMNIFICATION "
     "OBLIGATIONS, IN NO EVENT SHALL EITHER PARTY'S AGGREGATE LIABILITY ARISING OUT OF OR "
     "RELATED TO THIS AGREEMENT EXCEED THE TOTAL FEES PAID OR PAYABLE UNDER THIS AGREEMENT IN "
     "THE TWELVE (12) MONTHS PRECEDING THE EVENT GIVING RISE TO THE CLAIM. NEITHER PARTY "
     "SHALL BE LIABLE FOR ANY INDIRECT, INCIDENTAL, SPECIAL OR CONSEQUENTIAL DAMAGES."),
    ("Governing Law and Dispute Resolution", "This Agreement shall be governed by and "
     "construed in accordance with the laws of the jurisdiction stated in the key-terms table "
     "below, without regard to its conflict-of-laws principles. Any dispute arising out of or "
     "in connection with this Agreement shall first be referred to good-faith negotiation "
     "between the Parties, failing which it shall be resolved by binding arbitration before a "
     "sole arbitrator, and the seat of arbitration shall be as stated in the key-terms table."),
    ("Force Majeure", "Neither Party shall be liable for any failure or delay in performance "
     "under this Agreement to the extent such failure or delay is caused by circumstances "
     "beyond its reasonable control, including acts of God, natural disaster, war, "
     "governmental action, or failure of third-party infrastructure, provided that the "
     "affected Party gives prompt notice and uses reasonable efforts to mitigate the effect."),
    ("Entire Agreement and Amendment", "This Agreement, including any schedules and annexures "
     "attached hereto, constitutes the entire agreement between the Parties with respect to "
     "its subject matter and supersedes all prior discussions, negotiations and agreements, "
     "whether oral or written. No amendment or modification of this Agreement shall be "
     "effective unless in writing and signed by authorised representatives of both Parties."),
    ("Assignment", "Neither Party may assign or transfer this Agreement, in whole or in part, "
     "without the prior written consent of the other Party, except that either Party may "
     "assign this Agreement without consent to a successor entity in connection with a "
     "merger, acquisition, or sale of substantially all of its assets, subject to prompt "
     "written notice to the other Party."),
    ("Severability", "If any provision of this Agreement is held to be invalid, illegal or "
     "unenforceable by a court or arbitral tribunal of competent jurisdiction, the remaining "
     "provisions shall continue in full force and effect, and the invalid provision shall be "
     "deemed modified to the minimum extent necessary to make it enforceable."),
]

_DOC_TYPE_SPECIFIC = {
    "Non-Disclosure Agreement": [
        ("Purpose", "The Parties wish to explore a potential business relationship (the "
         "\"Purpose\") in connection with which each Party may disclose certain Confidential "
         "Information to the other Party. This Agreement sets out the terms on which such "
         "Confidential Information will be protected."),
        ("Permitted Use", "The Receiving Party shall use the Disclosing Party's Confidential "
         "Information solely for the Purpose and shall not use such information for its own "
         "benefit or the benefit of any third party."),
    ],
    "Lease Agreement": [
        ("Demised Premises", "The Lessor hereby leases to the Lessee, and the Lessee hereby "
         "takes on lease from the Lessor, the premises described in the key-terms table below "
         "(the \"Premises\"), together with the fixtures and fittings installed therein, for "
         "residential/commercial use as specified in the key-terms table."),
        ("Rent and Security Deposit", "The Lessee shall pay to the Lessor the monthly rent "
         "specified in the key-terms table, payable in advance on or before the 5th day of "
         "each calendar month. The Lessee shall deposit with the Lessor the security deposit "
         "specified in the key-terms table, refundable within thirty (30) days of vacating "
         "the Premises, less any deductions for damage beyond normal wear and tear."),
        ("Maintenance and Repairs", "The Lessee shall maintain the Premises in good and "
         "tenantable condition and shall be responsible for minor repairs. The Lessor shall "
         "be responsible for structural repairs and major maintenance not caused by the "
         "Lessee's negligence."),
    ],
    "Employment Agreement": [
        ("Position and Duties", "The Employee is engaged in the position specified in the "
         "key-terms table and shall perform the duties customarily associated with that "
         "position, and such other duties as may reasonably be assigned by the Employer from "
         "time to time."),
        ("Compensation and Benefits", "The Employer shall pay the Employee the annual "
         "compensation specified in the key-terms table, payable in accordance with the "
         "Employer's standard payroll practices, subject to applicable withholding. The "
         "Employee shall be eligible for the benefits described in the Employee Handbook."),
        ("Non-Compete and Non-Solicitation", "During the term of employment and for a period "
         "of twelve (12) months thereafter, the Employee shall not directly or indirectly "
         "solicit any client or employee of the Employer for a competing business, to the "
         "extent enforceable under applicable law."),
    ],
    "Sale Deed": [
        ("Sale Consideration", "In consideration of the total sale price specified in the "
         "key-terms table, paid and received by the Seller in the manner recorded in the "
         "key-terms table, the Seller hereby sells, conveys, transfers and assigns to the "
         "Buyer the property described in the Schedule, free from all encumbrances."),
        ("Title and Encumbrances", "The Seller represents and warrants that the Seller has "
         "clear, marketable and absolute title to the property, that the property is free "
         "from any mortgage, charge, lien or other encumbrance, and that the Seller has full "
         "right and authority to sell the property."),
        ("Delivery of Possession", "The Seller shall deliver vacant, physical possession of "
         "the property to the Buyer simultaneously with the execution and registration of "
         "this Sale Deed, together with all original title documents."),
    ],
    "Loan Agreement": [
        ("Loan Amount and Disbursement", "The Lender agrees to lend to the Borrower the "
         "principal amount specified in the key-terms table, to be disbursed in the manner "
         "and on the schedule set out in Schedule A, subject to the Borrower satisfying the "
         "conditions precedent set out in Clause 2."),
        ("Interest and Repayment", "The Borrower shall repay the loan together with interest "
         "at the rate specified in the key-terms table, in the number of equal monthly "
         "instalments specified in the key-terms table, commencing on the date specified "
         "therein. Any overdue amount shall accrue default interest at an additional two "
         "percent (2%) per annum."),
        ("Events of Default", "It shall be an event of default if the Borrower fails to pay "
         "any instalment within fifteen (15) days of its due date, becomes insolvent, or "
         "breaches any material covenant under this Agreement, upon which the Lender may "
         "declare the entire outstanding amount immediately due and payable."),
    ],
    "Partnership Deed": [
        ("Nature and Name of Business", "The Partners hereby agree to carry on the business "
         "described in the key-terms table in partnership under the firm name specified "
         "therein, and shall share in the profits and losses of the business in the "
         "proportions set out in the key-terms table."),
        ("Capital Contribution", "Each Partner shall contribute to the capital of the firm "
         "the amount specified against their name in the key-terms table. No Partner shall "
         "withdraw any part of their capital contribution without the written consent of all "
         "other Partners."),
        ("Management and Voting", "The business of the firm shall be managed jointly by all "
         "Partners. Decisions in the ordinary course of business shall require a simple "
         "majority vote; decisions to admit a new partner, dissolve the firm, or amend this "
         "Deed shall require the unanimous consent of all Partners."),
    ],
    "Service Agreement": [
        ("Scope of Services", "The Service Provider shall perform the services described in "
         "Schedule A (the \"Services\") in a professional and workmanlike manner consistent "
         "with generally accepted industry standards, and in accordance with the timelines "
         "set out in the key-terms table."),
        ("Fees and Payment Terms", "The Client shall pay the Service Provider the fees "
         "specified in the key-terms table within thirty (30) days of receipt of a valid "
         "invoice. Late payments shall accrue interest at one and a half percent (1.5%) per "
         "month."),
        ("Intellectual Property", "All work product created by the Service Provider "
         "specifically for the Client under this Agreement shall, upon full payment, be "
         "assigned to and become the exclusive property of the Client, save for the Service "
         "Provider's pre-existing tools and methodologies, which remain the Service "
         "Provider's property and are licensed to the Client for use with the deliverables."),
    ],
    "Power of Attorney": [
        ("Grant of Authority", "The Principal hereby appoints the Attorney as the Principal's "
         "true and lawful attorney, to act in the Principal's name and on the Principal's "
         "behalf, with respect to the matters described in the key-terms table, with full "
         "power to do all acts, deeds and things as the Attorney may consider necessary or "
         "expedient in connection therewith."),
        ("Duration and Revocation", "This Power of Attorney shall remain in force until "
         "revoked in writing by the Principal, or until the date specified in the key-terms "
         "table, whichever is earlier. The Principal reserves the right to revoke this Power "
         "of Attorney at any time by written notice to the Attorney."),
    ],
}

_ORGS = [
    "Nova Fintech Pvt. Ltd.", "Bluepeak Technologies Inc.", "Meridian Retail Group",
    "Sunrise Realty Partners", "Vantage Consulting LLP", "Crestview Manufacturing Co.",
    "Orion Healthcare Systems", "Bayline Logistics Ltd.", "Silverline Textiles",
    "Ridgeway Capital Partners", "GreenFields Agro Corp.", "Falcon Software Labs",
    "Harborview Hospitality Group", "Pinnacle Energy Ltd.", "Aster Biosciences Inc.",
]
_PEOPLE = [
    "Aditya Rao", "Meera Nair", "Rohan Kapoor", "Ananya Iyer", "Vikram Sethi",
    "Priya Menon", "Karan Malhotra", "Sneha Bhatt", "Arjun Desai", "Ishita Sharma",
    "Nikhil Verma", "Divya Krishnan", "Sameer Joshi", "Tanvi Kulkarni", "Rahul Chatterjee",
]
_CITIES = ["Delhi", "Mumbai", "Bengaluru", "Chennai", "Hyderabad", "Pune", "Kolkata", "Ahmedabad"]


def generate_contracts(n_per_type: int = 3) -> List[Contract]:
    contracts: List[Contract] = []
    doc_types = list(_DOC_TYPE_SPECIFIC.keys())
    idx = 0
    for doc_type in doc_types:
        for k in range(n_per_type):
            org = _ORGS[idx % len(_ORGS)]
            person = _PEOPLE[idx % len(_PEOPLE)]
            city = _CITIES[idx % len(_CITIES)]
            year = 2021 + (idx % 5)
            date = f"{(idx % 28) + 1:02d}-{(idx % 12) + 1:02d}-{year}"
            title = f"{doc_type} No. {1000 + idx}"

            if doc_type in ("Employment Agreement",):
                parties = [("Employer", org), ("Employee", person)]
            elif doc_type in ("Lease Agreement",):
                parties = [("Lessor", org), ("Lessee", person)]
            elif doc_type in ("Sale Deed",):
                parties = [("Seller", org), ("Buyer", person)]
            elif doc_type in ("Loan Agreement",):
                parties = [("Lender", org), ("Borrower", person)]
            elif doc_type in ("Partnership Deed",):
                parties = [("Partner 1", _PEOPLE[idx % len(_PEOPLE)]),
                           ("Partner 2", _PEOPLE[(idx + 1) % len(_PEOPLE)]),
                           ("Partner 3", _PEOPLE[(idx + 2) % len(_PEOPLE)])]
            elif doc_type in ("Power of Attorney",):
                parties = [("Principal", person), ("Attorney", _PEOPLE[(idx + 3) % len(_PEOPLE)])]
            else:
                parties = [("Party A", org), ("Party B", person)]

            clauses = list(_DOC_TYPE_SPECIFIC[doc_type]) + list(_STANDARD_CLAUSES)

            key_terms = [("Governing Law", f"Laws of India, courts at {city}"),
                         ("Arbitration Seat", city),
                         ("Effective Date", date)]
            if doc_type == "Lease Agreement":
                key_terms += [("Monthly Rent", f"Rs. {(idx % 9 + 1) * 15000:,}"),
                              ("Security Deposit", f"Rs. {(idx % 9 + 1) * 45000:,}"),
                              ("Premises", f"Unit {100 + idx}, {city}")]
            elif doc_type == "Employment Agreement":
                key_terms += [("Position", ["Software Engineer", "Legal Counsel", "Financial Analyst",
                                             "Operations Manager", "Product Designer"][idx % 5]),
                              ("Annual Compensation", f"Rs. {(idx % 6 + 4) * 3_00_000:,}")]
            elif doc_type == "Sale Deed":
                key_terms += [("Sale Price", f"Rs. {(idx % 9 + 3) * 12_00_000:,}"),
                              ("Property", f"Plot No. {200 + idx}, {city}")]
            elif doc_type == "Loan Agreement":
                key_terms += [("Principal Amount", f"Rs. {(idx % 9 + 2) * 5_00_000:,}"),
                              ("Interest Rate", f"{9 + (idx % 6)}% per annum"),
                              ("Instalments", f"{(idx % 4 + 3) * 12}")]
            elif doc_type == "Partnership Deed":
                key_terms += [("Firm Name", f"{org.split()[0]} & Associates"),
                              ("Profit Share", "Equal (1/3 each Partner)")]
            elif doc_type == "Power of Attorney":
                key_terms += [("Scope", ["Sale of immovable property", "Banking transactions",
                                          "Litigation representation", "Business management"][idx % 4])]
            elif doc_type == "Non-Disclosure Agreement":
                key_terms += [("Term of Confidentiality", "3 years from disclosure")]
            elif doc_type == "Service Agreement":
                key_terms += [("Total Fees", f"Rs. {(idx % 9 + 1) * 2_50_000:,}"),
                              ("Project Duration", f"{(idx % 6 + 3)} months")]

            contracts.append(Contract(title=title, doc_type=doc_type, parties=parties,
                                       effective_date=date, clauses=clauses, key_terms=key_terms))
            idx += 1
    return contracts


# ---------------------------------------------------------------------------
# Case judgments
# ---------------------------------------------------------------------------
_CASE_TEMPLATES = [
    ("breach of contract", "The plaintiff alleged that the defendant failed to deliver goods "
     "conforming to the agreed specifications within the contractually stipulated time, "
     "causing the plaintiff to source substitute goods at a higher price and suffer "
     "consequential losses.",
     "Whether the delay in delivery amounted to a fundamental breach entitling the plaintiff "
     "to terminate the contract and claim damages, and whether a limitation-of-liability "
     "clause in the contract barred recovery of consequential losses.",
     "The Court held that a time-of-the-essence clause, where expressly agreed, makes timely "
     "delivery a condition rather than a warranty; breach of a condition entitles the "
     "innocent party to treat the contract as repudiated. However, a validly incorporated "
     "limitation-of-liability clause, being a freely negotiated allocation of commercial "
     "risk between parties of equal bargaining power, was enforced to cap damages, excluding "
     "purely consequential losses not within the reasonable contemplation of the parties at "
     "the time of contracting."),
    ("employment termination", "The employee was terminated without notice on allegations of "
     "misconduct. The employee contended that no opportunity of being heard was granted "
     "before termination, in violation of the principles of natural justice and the terms of "
     "the employment contract.",
     "Whether termination without a prior inquiry violated principles of natural justice, and "
     "whether the employee was entitled to reinstatement or only to compensation in lieu "
     "thereof.",
     "The Court held that even a contract of employment terminable at will must, where "
     "misconduct is the stated ground, be preceded by a fair inquiry affording the employee "
     "an opportunity to respond to the allegations; failure to do so renders the termination "
     "procedurally unfair. On the facts, since the employer-employee relationship had broken "
     "down irretrievably, the Court declined reinstatement and instead awarded compensation "
     "equivalent to twelve months' salary in lieu of reinstatement."),
    ("landlord-tenant dispute", "The landlord sought eviction of the tenant on the ground of "
     "non-payment of rent for six consecutive months, while the tenant contended that rent "
     "had been withheld due to the landlord's failure to carry out essential structural "
     "repairs.",
     "Whether the tenant was justified in withholding rent on account of the landlord's "
     "failure to repair, and whether such withholding could be treated as a defence to an "
     "eviction petition premised on non-payment of rent.",
     "The Court held that in the absence of an express contractual right to withhold rent for "
     "disrepair, the tenant's remedy lay in seeking specific performance of the repair "
     "covenant or damages, not in unilateral non-payment of rent; nevertheless, on account of "
     "the landlord's admitted and prolonged default in undertaking repairs, the Court ordered "
     "payment of arrears in instalments in lieu of immediate eviction, conditional upon the "
     "landlord completing the repairs within sixty days."),
    ("negligence and tortious liability", "The plaintiff suffered injury on the defendant's "
     "commercial premises due to an unmarked wet floor. The defendant argued that a "
     "cautionary sign, though not clearly visible, had in fact been placed near the area.",
     "Whether the defendant, as occupier of the premises, owed and breached a duty of care to "
     "a lawful visitor, and whether the placement of an inadequately visible warning sign "
     "discharged that duty.",
     "The Court held that an occupier owes a duty to take reasonable care to ensure that "
     "lawful visitors are reasonably safe in using the premises for the purposes for which "
     "they are invited, which includes taking reasonably effective steps to warn of known "
     "hazards. A warning sign that is not reasonably conspicuous to an ordinary visitor does "
     "not discharge this duty. The defendant was held liable, with contributory negligence "
     "of twenty percent attributed to the plaintiff for not exercising ordinary care."),
    ("intellectual property infringement", "The plaintiff, owner of a registered trademark, "
     "alleged that the defendant's use of a deceptively similar mark on competing goods was "
     "likely to cause confusion among consumers as to the origin of the goods.",
     "Whether the marks were deceptively similar such that an ordinary consumer of average "
     "intelligence and imperfect recollection would likely be confused, having regard to the "
     "overall structure, phonetic similarity and the nature of the goods.",
     "The Court applied the test of overall impression rather than a side-by-side comparison "
     "of the marks, holding that phonetic and structural similarity, combined with identical "
     "trade channels and target consumers, created a real likelihood of confusion. An "
     "injunction was granted restraining further use of the impugned mark, together with an "
     "account of profits attributable to the infringing use."),
    ("consumer protection", "The complainant purchased a home appliance that developed a "
     "recurring defect within the warranty period; despite repeated service requests, the "
     "manufacturer's authorised service centre failed to permanently cure the defect.",
     "Whether the recurring, uncured defect amounted to a manufacturing defect constituting "
     "a deficiency in service and an unfair trade practice, entitling the complainant to "
     "replacement or refund rather than repeated repair.",
     "The Court held that where a product suffers from a persistent defect that repeated "
     "repair attempts fail to cure, the consumer's confidence in the product's fitness for "
     "purpose is undermined, and the consumer is not obliged to accept indefinite repair as "
     "an adequate remedy. The manufacturer was directed to replace the product with a new "
     "unit or refund the full purchase price with interest, at the complainant's election, "
     "together with compensation for harassment and litigation costs."),
    ("company law - oppression and mismanagement", "A minority shareholder alleged that the "
     "majority shareholders had conducted the affairs of the company in a manner "
     "oppressive to minority interests, including by diverting business opportunities to a "
     "related entity controlled by the majority.",
     "Whether the diversion of corporate opportunity to a related entity, without full "
     "disclosure to or consent of the minority shareholder, amounted to oppression and "
     "mismanagement under company law, and what relief was appropriate.",
     "The Court held that directors and majority shareholders owe fiduciary duties to the "
     "company and must not divert corporate opportunities for personal or related-party "
     "benefit without full disclosure and appropriate consent; the burden lies on those in "
     "control to justify related-party transactions as being on arm's-length terms. Finding "
     "the diversion unjustified, the Court ordered disgorgement of profits to the company and "
     "granted the minority shareholder an exit at fair value determined by an independent "
     "valuer."),
    ("arbitration and enforcement of award", "The award-debtor resisted enforcement of a "
     "foreign arbitral award on the ground that the arbitral tribunal had decided issues "
     "beyond the scope of the arbitration agreement.",
     "Whether the tribunal exceeded its jurisdiction by deciding matters not falling within "
     "the scope of the arbitration clause, and whether such excess, if any, justified refusal "
     "to enforce the entire award or only the excess portion.",
     "The Court held that where an award deals with matters both within and beyond the scope "
     "of the submission to arbitration, and the matters within scope can be separated from "
     "those beyond scope, only the excess portion is liable to be set aside or refused "
     "enforcement, while the remainder of the award, being severable, is enforced. Public "
     "policy objections to enforcement were held to require a high threshold, not met on the "
     "facts."),
]

_COURTS = ["High Court", "District Court", "Commercial Court", "Consumer Disputes Redressal Commission",
           "Company Law Tribunal", "Arbitration Appellate Bench"]
_JUDGES = ["Hon'ble Justice R. Sharma", "Hon'ble Justice A. Fernandes", "Hon'ble Justice S. Krishnan",
           "Hon'ble Justice N. Kapadia", "Hon'ble Justice P. Bose", "Hon'ble Justice M. Reddy"]


def generate_cases(n: int = 24) -> List[CaseJudgment]:
    cases: List[CaseJudgment] = []
    for i in range(n):
        topic, facts, issue, holding = _CASE_TEMPLATES[i % len(_CASE_TEMPLATES)]
        p1, p2 = _ORGS[i % len(_ORGS)], _PEOPLE[i % len(_PEOPLE)]
        year = 2018 + (i % 8)
        case_no = f"{200 + i}"
        court = _COURTS[i % len(_COURTS)]
        case_name = f"{p2.split()[-1]} v. {p1.split()[0]}"
        citation = f"({year}) LGL {case_no} ({court.split()[0]})"
        outcome = ["Appeal allowed", "Appeal dismissed", "Petition partly allowed",
                   "Suit decreed", "Suit dismissed", "Application allowed with costs"][i % 6]
        cases.append(CaseJudgment(
            case_name=f"{case_name} — {topic.title()}",
            citation=citation,
            court=court,
            date=f"{(i % 28) + 1:02d}-{(i % 12) + 1:02d}-{year}",
            judge=_JUDGES[i % len(_JUDGES)],
            outcome=outcome,
            facts=facts,
            issues=[issue],
            holding=holding,
            reasoning=holding,
        ))
    return cases


# ---------------------------------------------------------------------------
# Statute-style sections (synthetic / paraphrased, for indexing demo only)
# ---------------------------------------------------------------------------
def generate_statute_sections() -> List[StatuteSection]:
    sections: List[Tuple[str, str, str, str]] = [
        ("General Contract Code", "10", "What agreements are contracts",
         "All agreements are contracts if they are made by the free consent of parties "
         "competent to contract, for a lawful consideration and with a lawful object, and are "
         "not hereby expressly declared to be void. Nothing herein contained shall affect any "
         "law relating to the form in which a contract must be made where such a requirement "
         "is prescribed by another enactment."),
        ("General Contract Code", "23", "Lawful and unlawful consideration and object",
         "The consideration or object of an agreement is lawful, unless it is forbidden by "
         "law, or is of such a nature that, if permitted, it would defeat the provisions of "
         "any law, or is fraudulent, or involves or implies injury to a person or property of "
         "another, or the court regards it as immoral or opposed to public policy. In each of "
         "these cases, the consideration or object of an agreement is said to be unlawful, "
         "and every agreement of which the object or consideration is unlawful is void."),
        ("General Contract Code", "56", "Discharge of contract by impossibility of performance",
         "An agreement to do an act impossible in itself is void. A contract to do an act "
         "which, after the contract is made, becomes impossible, or, by reason of some event "
         "which the promisor could not prevent, unlawful, becomes void when the act becomes "
         "impossible or unlawful. Where a person has promised to do something which he knew, "
         "or with reasonable diligence might have known, to be impossible or unlawful, but "
         "which the promisee did not know to be impossible or unlawful, such promisor must "
         "compensate the promisee for any loss which the promisee sustains through the "
         "non-performance of the promise."),
        ("General Contract Code", "73", "Compensation for loss or damage caused by breach of contract",
         "When a contract has been broken, the party who suffers by such breach is entitled to "
         "receive, from the party who has broken the contract, compensation for any loss or "
         "damage caused to him thereby which naturally arose in the usual course of things "
         "from such breach, or which the parties knew, when they made the contract, to be "
         "likely to result from the breach of it. Compensation is not to be given for any "
         "remote and indirect loss or damage sustained by reason of the breach."),
        ("Consumer Protection Code", "2", "Definition of unfair trade practice",
         "\"Unfair trade practice\" means a trade practice which, for the purpose of promoting "
         "the sale, use or supply of any goods or the provision of any service, adopts an "
         "unfair method or deceptive practice, including the making of a false or misleading "
         "representation concerning the standard, quality, quantity or grade of goods or "
         "services, or the failure to remedy a defect within a reasonable time and on "
         "reasonable terms after due notice."),
        ("Consumer Protection Code", "14", "Reliefs available to a consumer",
         "Where a complaint is proved, the appropriate authority may direct the opposite party "
         "to remove the defect, replace the goods with new goods of similar description free "
         "from defect, return to the complainant the price paid, pay compensation for any loss "
         "or injury suffered due to the negligence of the opposite party, and/or discontinue "
         "the unfair trade practice complained of and not repeat it."),
        ("Companies Code", "166", "Duties of directors",
         "A director shall act in good faith in order to promote the objects of the company "
         "for the benefit of its members as a whole, and in the best interests of the company, "
         "its employees, shareholders and the community. A director shall not achieve or "
         "attempt to achieve any undue gain or advantage either to himself or to his relatives, "
         "partners or associates, and if found guilty of making such undue gain, shall be "
         "liable to pay an amount equal to that gain to the company."),
        ("Companies Code", "241", "Application to tribunal for relief in cases of oppression",
         "Any member of a company who complains that the affairs of the company are being "
         "conducted in a manner prejudicial to public interest, or in a manner oppressive to "
         "any member or members, may apply to the Tribunal for an order, provided such member "
         "holds not less than the prescribed threshold of the issued share capital, or such "
         "other threshold as the Tribunal may waive on application."),
        ("Labour Welfare Code", "25", "Conditions precedent to retrenchment of workmen",
         "No workman employed for not less than one year under an employer shall be "
         "retrenched until the workman has been given one month's notice in writing indicating "
         "the reasons for retrenchment, or wages in lieu of such notice, and has been paid "
         "compensation equivalent to fifteen days' average pay for every completed year of "
         "continuous service, and notice in the prescribed manner has been served on the "
         "appropriate government authority."),
        ("Labour Welfare Code", "33", "Conditions of service during pendency of proceedings",
         "During the pendency of any dispute before an adjudicating authority, no employer "
         "shall alter, to the prejudice of a workman concerned in the dispute, the conditions "
         "of service applicable to that workman immediately before the commencement of the "
         "proceedings, nor discharge or punish that workman, save with the express permission "
         "in writing of the authority before which the proceeding is pending."),
        ("Intellectual Property Code", "29", "Infringement of registered trademarks",
         "A registered trademark is infringed by a person who, not being a registered "
         "proprietor or a permitted user, uses in the course of trade a mark which is "
         "identical with, or deceptively similar to, the registered trademark in relation to "
         "goods or services in respect of which the trademark is registered, in such manner "
         "as to render the use of the mark likely to be taken as being used as a trademark."),
        ("Intellectual Property Code", "52", "Certain acts not to be infringement of copyright",
         "A fair dealing with a literary, dramatic, musical or artistic work for the purposes "
         "of private or personal use, including research, criticism, review, or reporting of "
         "current events, shall not constitute an infringement of copyright, provided that the "
         "extent of use is reasonable and proper attribution is made where practicable."),
        ("Arbitration and Conciliation Code", "34", "Application for setting aside arbitral award",
         "An arbitral award may be set aside by the court only if the party making the "
         "application furnishes proof that a party was under some incapacity, the arbitration "
         "agreement is not valid, the party was not given proper notice of the appointment of "
         "an arbitrator or of the arbitral proceedings, the award deals with a dispute not "
         "contemplated by or not falling within the terms of submission to arbitration, or the "
         "award is in conflict with the public policy of the jurisdiction."),
        ("Arbitration and Conciliation Code", "48", "Conditions for enforcement of foreign awards",
         "Enforcement of a foreign arbitral award may be refused only if the party against whom "
         "it is invoked furnishes proof of one of the limited grounds specified by this Code, "
         "including incapacity of a party, invalidity of the arbitration agreement, lack of "
         "proper notice, the award exceeding the scope of submission, or the award or its "
         "enforcement being contrary to the public policy of the jurisdiction where enforcement "
         "is sought."),
        ("Property Transfer Code", "54", "Sale defined; how made",
         "\"Sale\" is a transfer of ownership in exchange for a price paid or promised or part "
         "paid and part promised. In the case of tangible immovable property of value one "
         "hundred rupees and upwards, a sale can be made only by a registered instrument; in "
         "the case of tangible immovable property of value less than one hundred rupees, a "
         "sale may be made either by a registered instrument or by delivery of the property."),
        ("Property Transfer Code", "106", "Duration of certain leases in absence of written contract",
         "In the absence of a contract or local law to the contrary, a lease of immovable "
         "property for agricultural or manufacturing purposes shall be deemed to be a lease "
         "from year to year, terminable on either side by six months' notice; a lease for any "
         "other purpose shall be deemed to be a lease from month to month, terminable on "
         "either side by fifteen days' notice, expiring with the end of a month of the "
         "tenancy."),
    ]
    return [StatuteSection(*s) for s in sections]
