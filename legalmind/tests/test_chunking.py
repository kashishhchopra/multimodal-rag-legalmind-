from src.chunking.structure_splitter import split_into_clauses


def test_splits_on_numbered_sections():
    text = (
        "1. Definitions\n"
        "This section defines terms.\n"
        "2. Termination\n"
        "Either party may terminate upon 30 days' notice.\n"
        "(a) Notice must be in writing.\n"
        "(b) Notice must be sent to the address on file.\n"
    )
    spans = split_into_clauses(text)

    assert len(spans) >= 3
    assert any("Definitions" in s.text for s in spans)
    assert any(s.section_path and "2." in s.section_path for s in spans)


def test_subclauses_carry_parent_section_path():
    text = "12. Termination for Cause\nSome text.\n(b) A carve-out.\n"
    spans = split_into_clauses(text)

    subclause = next(s for s in spans if "(b)" in s.text)
    assert subclause.section_path is not None
    assert "12." in subclause.section_path


def test_falls_back_to_recursive_split_when_no_numbering():
    text = "This is a recital with no numbering at all. " * 50
    spans = split_into_clauses(text)

    assert len(spans) >= 1
    assert all(s.text for s in spans)
