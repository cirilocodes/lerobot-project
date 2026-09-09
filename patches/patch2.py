import re, shutil, sys
from pathlib import Path
try:
    import so101_nexus.mujoco as m
except ImportError:
    sys.exit("Activate the venv first.")
d = Path(m.__file__).parent
print("Dir:", d)
pat = re.compile(
    r"^([ \t]*)with tempfile\.NamedTemporaryFile\([^\n]*\) as f:\n"
    r"[ \t]*f\.write\(([^\n]*?)\)\n"
    r"(?:[ \t]*f\.flush\(\)\n)?"
    r"(?:[ \t]*os\.fsync\([^\n]*\)\n)?"
    r"[ \t]*(.+?)\s*=\s*mujoco\.MjModel\.from_xml_path\(f\.name\)\n",
    re.M)
def sub(mo):
    p, var, lhs = mo.group(1), mo.group(2), mo.group(3)
    return (f'{p}tmp_path = os.path.join(_SO101_DIR, f"tmp_{{uuid.uuid4().hex}}.xml")\n'
            f"{p}try:\n"
            f'{p}    with open(tmp_path, "w") as f:\n'
            f"{p}        f.write({var})\n"
            f"{p}    {lhs} = mujoco.MjModel.from_xml_path(tmp_path)\n"
            f"{p}finally:\n"
            f"{p}        if os.path.exists(tmp_path):\n"
            f"{p}            os.remove(tmp_path)\n")
for p in sorted(d.glob("*.py")):
    t = p.read_text(encoding="utf-8")
    if "NamedTemporaryFile" not in t:
        continue
    new, n = pat.subn(sub, t)
    if n == 0:
        print(f"  {p.name}: MATCH FAILED")
        continue
    for imp in ("import uuid\n", "import os\n"):
        if not re.search(rf"^{imp.strip()}\s*$", new, re.M):
            new = imp + new
    b = p.with_suffix(".py.bak")
    if not b.exists():
        shutil.copy2(p, b)
    p.write_text(new, encoding="utf-8")
    print(f"  {p.name}: patched ({n})")
print("done")
