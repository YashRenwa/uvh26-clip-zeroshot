"""Render report/report.html from report_template.html + numbers.json, check numbers, print the PDF.

Every number in the report text is a placeholder {{key:fmt}} filled from report/numbers.json, so the
report cannot drift from results/. The check lists any other numeric literal left in the template text;
each must be a design constant on the allow-list below (or be turned into a placeholder).

    python report/extract_outputs.py && python report/build_report.py
"""

from __future__ import annotations

import html
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
PDF = REPORT / "Yash_Renwa_2025AG05357_EC1_A2_Report.pdf"
FIGS = {"val_examples.png": "results/figures/val_examples.png",
        "test_reliability.png": "results/figures/test_reliability.png",
        "test_coverage.png": "results/figures/test_coverage.png"}   # the rest are written to report/figures directly

PH = re.compile(r"\{\{([^}:]+?)(?::(\w+))?\}\}")

# Design constants and identifiers that may appear as literals (not results).
ALLOWED = {"1", "2", "3", "5", "8", "10", "14", "15", "30", "32", "95", "96", "100", "400", "0.3", "0.99",
           "1.1", "1920", "1080", "2021", "2025", "2004", "23", "7", "106", "1990", "2025AG05357", "16",
           "2511.02563", "26", "0", "20", "4", "6", "11", "12", "40"}  # + section, figure and pair numbers


def fmt(v, f: str | None) -> str:
    if f is None or f == "raw":
        return str(v)
    if f == "int":
        return f"{int(round(v)):,}"
    if f.startswith("pct"):
        return f"{100 * v:.{int(f[3:])}f}%"
    if f.startswith("sf"):
        return f"{v:+,.{int(f[2:])}f}".replace("-", "−")
    if f.startswith("f"):
        return f"{v:,.{int(f[1:])}f}".replace("-", "−")
    raise ValueError(f)


def literals(template: str) -> list[str]:
    """Numeric literals in the visible template text (placeholders, tags, attributes, code removed)."""
    t = PH.sub(" ", template)
    t = re.sub(r"<style.*?</style>", " ", t, flags=re.S)
    t = re.sub(r"<code>.*?</code>", " ", t, flags=re.S)
    t = re.sub(r"<[^>]+>", " ", t)
    t = html.unescape(t)
    t = re.sub(r"\b[0-9a-f]{7,}\b", " ", t)           # hashes
    t = re.sub(r"2025AG05357|sha256|v1\.0|ViT-B/16|YOLOv8|UVH-26|TMI 23\(7\)|IRC:106-1990|EC-1|ZG525|4\.0", " ", t)
    return [m for m in re.findall(r"\d[\d,]*(?:\.\d+)?", t) if m.replace(",", "") not in ALLOWED]


def main() -> None:
    n = json.loads((REPORT / "numbers.json").read_text())
    tpl = (REPORT / "report_template.html").read_text()
    missing = sorted({k for k, _ in PH.findall(tpl) if k not in n})
    assert not missing, f"placeholders without a number: {missing}"
    stray = literals(tpl)
    out = PH.sub(lambda m: fmt(n[m.group(1)], m.group(2)), tpl)
    for name, src in FIGS.items():
        shutil.copy2(ROOT / src, REPORT / "figures" / name)
    (REPORT / "report.html").write_text(out)
    used = len(PH.findall(tpl))
    print(f"report.html: {used} numbers filled from numbers.json; literals not on the allow-list: {stray or 'none'}")
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={PDF}", (REPORT / "report.html").as_uri()],
                   check=True, capture_output=True)
    pages = subprocess.run(["pdfinfo", str(PDF)], capture_output=True, text=True).stdout
    print(PDF.name, [l for l in pages.splitlines() if l.startswith("Pages")])
    if stray:
        sys.exit(1)


if __name__ == "__main__":
    main()
