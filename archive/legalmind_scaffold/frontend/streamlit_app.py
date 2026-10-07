"""Layer 6 presentation: chat UI + side-by-side PDF viewer with clause
highlighting, backed by the FastAPI service in src/api.

Streamlit chosen to ship fast within the capstone timeline (see
docs/ARCHITECTURE.md 07); swap for React + PDF.js later for more polished
highlight rendering.
"""
from __future__ import annotations

import os

import requests
import streamlit as st

API_BASE = os.environ.get("LEGALMIND_API_BASE", "http://localhost:8000")

st.set_page_config(page_title="LegalMind", layout="wide")
st.title("⚖️ LegalMind — grounded legal document Q&A")

with st.sidebar:
    st.header("Documents")
    uploaded = st.file_uploader("Upload a contract, notice, or judgment", type=["pdf", "png", "jpg", "jpeg"])
    doc_type = st.text_input("Document type (optional)", placeholder="NDA, lease, judgment…")
    jurisdiction = st.text_input("Jurisdiction (optional)", placeholder="Delaware, India…")

    if uploaded is not None and st.button("Ingest document"):
        files = {"file": (uploaded.name, uploaded.getvalue())}
        params = {"doc_type": doc_type or None, "jurisdiction": jurisdiction or None}
        response = requests.post(f"{API_BASE}/documents/upload", files=files, params=params)
        if response.ok:
            doc_id = response.json()["doc_id"]
            st.session_state.setdefault("doc_ids", []).append(doc_id)
            st.success(f"Queued for ingestion — doc_id={doc_id}")
        else:
            st.error(response.text)

    st.divider()
    st.caption("Documents in this session")
    for doc_id in st.session_state.get("doc_ids", []):
        status = requests.get(f"{API_BASE}/documents/{doc_id}/status").json().get("status", "unknown")
        st.write(f"`{doc_id[:8]}…` — {status}")

tab_qa, tab_risk, tab_conflict, tab_translate = st.tabs(
    ["💬 Ask", "🚦 Risk analysis", "⚔️ Conflict detection", "📝 Plain English"]
)

with tab_qa:
    question = st.text_input("Ask a question about your uploaded documents")
    if question and st.button("Ask", key="ask_button"):
        payload = {"question": question, "doc_ids": st.session_state.get("doc_ids") or None}
        response = requests.post(f"{API_BASE}/query", json=payload)
        if response.ok:
            result = response.json()
            if result["insufficient_context"]:
                st.warning("LegalMind doesn't have enough context in your documents to answer confidently.")
            st.markdown(result["answer"])
            st.subheader("Sources")
            for citation in result["citations"]:
                with st.expander(f"Page {citation['page']} — {citation['doc_id'][:8]}…"):
                    st.write(citation["quote"])
                    # TODO: render the source PDF page here and draw
                    # citation["bbox"] as a highlight rectangle (PDF.js /
                    # pdf-lib overlay) for the "clickable citation" experience.
        else:
            st.error(response.text)

with tab_risk:
    doc_id_for_risk = st.selectbox("Document", st.session_state.get("doc_ids", []))
    if doc_id_for_risk and st.button("Analyze risks"):
        response = requests.get(f"{API_BASE}/features/risks/{doc_id_for_risk}")
        if response.ok:
            colors = {"RED": "🔴", "AMBER": "🟠", "GREEN": "🟢"}
            for finding in response.json():
                st.write(f"{colors.get(finding['level'], '⚪')} **{finding['section_path'] or 'Unlabelled clause'}** (page {finding['page']}) — {finding['explanation']}")
        else:
            st.error(response.text)

with tab_conflict:
    clause_type = st.text_input("Clause type to compare", placeholder="termination notice period")
    selected_docs = st.multiselect("Documents to compare", st.session_state.get("doc_ids", []))
    if clause_type and selected_docs and st.button("Detect conflicts"):
        response = requests.post(f"{API_BASE}/features/conflicts", json={"clause_type": clause_type, "doc_ids": selected_docs})
        if response.ok:
            findings = response.json()
            if not findings:
                st.success("No conflicts detected across the selected documents.")
            for f in findings:
                st.warning(f"{f['doc_id_a'][:8]}… vs {f['doc_id_b'][:8]}…: {f['explanation']}")
        else:
            st.error(response.text)

with tab_translate:
    clause_text = st.text_area("Paste a clause to translate")
    if clause_text and st.button("Translate to plain English"):
        response = requests.post(f"{API_BASE}/features/translate", json={"clause_text": clause_text})
        if response.ok:
            st.markdown(response.json()["plain_english"])
        else:
            st.error(response.text)
