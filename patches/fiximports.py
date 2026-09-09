import re
from pathlib import Path
import so101_nexus.mujoco as m
d = Path(m.__file__).parent
for p in sorted(d.glob("*.py")):
    lines = p.read_text(encoding="utf-8").splitlines(keepends=True)
    head = []
    while lines and lines[0].strip() in ("import os", "import uuid"):
        head.append(lines.pop(0))
    if not head:
        continue
    fut = next((i for i, l in enumerate(lines) if l.startswith("from __future__")), None)
    if fut is None:
        lines = head + lines
    else:
        lines = lines[:fut+1] + ["\n"] + head + lines[fut+1:]
    p.write_text("".join(lines), encoding="utf-8")
    print(f"  {p.name}: moved {len(head)} import(s) below __future__")
print("done")
