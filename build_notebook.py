"""Convert notebook_src.py (percent-format cells) into IMF_Canada_LLM_Pipeline.ipynb."""
import re
from pathlib import Path
import nbformat

src = Path(__file__).with_name("notebook_src.py").read_text(encoding="utf-8")
cells = []
for block in re.split(r"^# %%", src, flags=re.M)[1:]:
    header, _, body = block.partition("\n")
    body = body.strip("\n")
    if header.strip() == "[markdown]":
        md = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.splitlines())
        cells.append(nbformat.v4.new_markdown_cell(md))
    else:
        cells.append(nbformat.v4.new_code_cell(body))

nb = nbformat.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"name": "python3", "display_name": "Python 3"},
    "language_info": {"name": "python"},
    "colab": {"provenance": []}})
out = Path(__file__).with_name("IMF_Canada_LLM_Pipeline.ipynb")
nbformat.write(nb, out)
print(f"Wrote {out.name}: {len(cells)} cells")
