"""Generate a small, realistic technical PDF for demos and tests.

Produces `electronics_manual.pdf` containing:
  * explanatory text about R1 and Q1,
  * a vector-drawn switching-circuit schematic labelled "Figure 12",
  * a component table.

This gives the project a genuine multimodal document to demonstrate on, and the
test-suite reuses it as a fixture. Run:  python scripts/make_sample_pdf.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas


def _draw_schematic(c: canvas.Canvas, x: float, y: float) -> None:
    """Draw a large, clearly-labelled R1 -> Q1 -> LED switching circuit.

    Geometry is scaled up (and labels use a large bold font) so that OCR can
    read the component references reliably for the visual-grounding demo.
    """
    s = 2.0  # scale factor
    c.setLineWidth(2.0)
    label_font = ("Helvetica-Bold", 20)

    # supply rail (VCC across the top)
    c.line(x, y + 60 * s, x + 220 * s, y + 60 * s)
    c.setFont(*label_font)
    c.drawString(x - 6, y + 66 * s, "VCC")

    # R1 (resistor as a box) on the base line
    c.rect(x + 20 * s, y + 30 * s, 40 * s, 16 * s)
    c.drawString(x + 22 * s, y + 50 * s, "R1")
    c.line(x + 10 * s, y + 38 * s, x + 20 * s, y + 38 * s)     # into R1
    c.line(x + 60 * s, y + 38 * s, x + 95 * s, y + 38 * s)     # R1 -> Q1 base
    c.line(x + 10 * s, y + 60 * s, x + 10 * s, y + 38 * s)     # rail down to R1

    # Q1 (transistor as a circle)
    c.circle(x + 115 * s, y + 38 * s, 18 * s, stroke=1, fill=0)
    c.drawString(x + 128 * s, y + 44 * s, "Q1")
    c.line(x + 95 * s, y + 38 * s, x + 97 * s, y + 38 * s)     # base wire
    c.line(x + 128 * s, y + 50 * s, x + 150 * s, y + 60 * s)   # collector -> rail
    c.line(x + 150 * s, y + 60 * s, x + 150 * s, y + 12 * s)   # collector rail down
    c.line(x + 128 * s, y + 26 * s, x + 150 * s, y + 12 * s)   # emitter -> LED node

    # LED then to ground
    c.line(x + 150 * s, y + 12 * s, x + 175 * s, y + 12 * s)
    c.drawString(x + 158 * s, y - 6 * s, "LED")
    c.line(x + 190 * s, y + 12 * s, x + 190 * s, y - 10 * s)   # to ground
    # ground symbol
    c.line(x + 176 * s, y - 10 * s, x + 204 * s, y - 10 * s)
    c.line(x + 181 * s, y - 15 * s, x + 199 * s, y - 15 * s)
    c.line(x + 186 * s, y - 20 * s, x + 194 * s, y - 20 * s)
    c.setFont("Helvetica", 14)
    c.drawString(x + 176 * s, y - 34 * s, "GND")


def build_sample_pdf(out_path: Path) -> Path:
    c = canvas.Canvas(str(out_path), pagesize=A4)
    width, height = A4

    # ---- Page 1: title + explanatory text ----
    c.setFont("Helvetica-Bold", 18)
    c.drawString(25 * mm, height - 30 * mm, "Basic Electronics Manual")
    c.setFont("Helvetica", 11)
    text = c.beginText(25 * mm, height - 45 * mm)
    for line in [
        "Chapter 4: Transistor Switching Circuits",
        "",
        "A transistor can be used as an electronic switch. In the switching",
        "circuit, R1 is a current-limiting resistor connected to the base of the",
        "transistor Q1. R1 limits the current entering the base of Q1 so that the",
        "transistor is not damaged.",
        "",
        "Q1 functions as a switching transistor. When a sufficient base current",
        "flows through R1 into Q1, the transistor turns on and allows current to",
        "flow from the collector to the emitter, lighting the LED.",
        "",
        "The value of R1 determines the base current and therefore whether Q1",
        "saturates. A larger R1 reduces the base current. The LED acts as the load",
        "in this circuit and is protected by the switching action of Q1.",
    ]:
        text.textLine(line)
    c.drawText(text)
    c.showPage()

    # ---- Page 2: the schematic (Figure 12) ----
    c.setFont("Helvetica-Bold", 13)
    c.drawString(25 * mm, height - 30 * mm, "4.3 The Switching Circuit")
    _draw_schematic(c, 25 * mm, height - 110 * mm)
    c.setFont("Helvetica-Oblique", 10)
    c.drawString(25 * mm, height - 130 * mm, "Figure 12: A transistor switching circuit (R1, Q1 and an LED).")
    c.setFont("Helvetica", 10)
    caption = c.beginText(25 * mm, height - 160 * mm)
    for line in [
        "In Figure 12, R1 is placed before Q1 and controls the current entering",
        "the transistor. Q1 is the switching element that drives the LED load.",
    ]:
        caption.textLine(line)
    c.drawText(caption)
    c.showPage()

    # ---- Page 3: component table (ruled, so pdfplumber detects it) ----
    c.setFont("Helvetica-Bold", 13)
    c.drawString(25 * mm, height - 30 * mm, "Table 3: Component List")
    rows = [
        ("Ref", "Component", "Value", "Function"),
        ("R1", "Resistor", "10k", "Current-limiting base resistor"),
        ("Q1", "NPN Transistor", "BC547", "Switching element"),
        ("LED", "Light Emitting Diode", "-", "Load / indicator"),
    ]
    col_x = [25 * mm, 45 * mm, 85 * mm, 110 * mm, 180 * mm]  # 4 columns
    row_h = 9 * mm
    top = height - 42 * mm
    n_rows = len(rows)
    c.setLineWidth(0.8)
    # horizontal rules
    for i in range(n_rows + 1):
        y = top - i * row_h
        c.line(col_x[0], y, col_x[-1], y)
    # vertical rules
    for x in col_x:
        c.line(x, top, x, top - n_rows * row_h)
    # cell text
    for ri, r in enumerate(rows):
        c.setFont("Helvetica-Bold" if ri == 0 else "Helvetica", 9)
        for ci, cell in enumerate(r):
            c.drawString(col_x[ci] + 2 * mm, top - (ri + 1) * row_h + 3 * mm, cell)
    c.showPage()

    c.save()
    return out_path


if __name__ == "__main__":
    dest = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "data" / "electronics_manual.pdf"
    dest.parent.mkdir(parents=True, exist_ok=True)
    p = build_sample_pdf(dest)
    print(f"Wrote sample PDF -> {p}")
