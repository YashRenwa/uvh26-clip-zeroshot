"""Replace markdown cells of the EXECUTED notebook by cell id, without re-running it.

Code cells, outputs and execution counts are never touched; the script asserts this. Only markdown cells
can be replaced.

    python notebooks/replace_markdown.py yash-step1=notes/step1.md yash-contamination=notes/contamination.md
    python notebooks/replace_markdown.py --list        # show markdown cell ids and their first line

Mirror the same text in notebooks/build_notebook.py (cell_id=...) so a later rebuild keeps it.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import nbformat as nbf

NB = Path(__file__).with_name("uvh26_clip_zeroshot.ipynb")


def frozen_part(nb) -> str:
    """Everything except markdown sources: must not change."""
    cells = []
    for c in nb.cells:
        c = copy.deepcopy(dict(c))
        if c["cell_type"] == "markdown":
            c["source"] = ""
        cells.append(c)
    return json.dumps(cells, sort_keys=True, default=str)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pairs", nargs="*", help="cell_id=path/to/text.md")
    ap.add_argument("--notebook", default=str(NB))
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    path = Path(args.notebook)
    nb = nbf.read(path, as_version=4)
    by_id = {c["id"]: c for c in nb.cells}
    if args.list:
        for c in nb.cells:
            if c["cell_type"] == "markdown":
                print(f"{c['id']:20s} {c['source'].splitlines()[0][:80] if c['source'] else ''}")
        return
    before = frozen_part(nb)
    for pair in args.pairs:
        cid, _, src = pair.partition("=")
        assert cid in by_id, f"no cell with id {cid!r} (use --list)"
        assert by_id[cid]["cell_type"] == "markdown", f"{cid} is a code cell; only markdown cells can be replaced"
        by_id[cid]["source"] = Path(src).read_text().strip("\n")
        print(f"replaced {cid} from {src}")
    assert frozen_part(nb) == before, "code cells or outputs changed"
    nbf.validate(nb)
    nbf.write(nb, path)
    print(f"wrote {path} (code and outputs unchanged)")


if __name__ == "__main__":
    main()
