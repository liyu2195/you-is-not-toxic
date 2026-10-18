"""A flattened, readable copy of a paper's LaTeX source: every \\input / \\tabinput inlined (sections, tables) and every generated
number macro (from the numbers_*.tex files that main.tex inputs) replaced by its value. For reading and review only.
usage: python3 x99c_flatten.py PAPER_DIR OUT_TEX"""
import re, sys, os

D, OUT = sys.argv[1], sys.argv[2]
main = open(f"{D}/main.tex").read()
M = {}
for f in re.findall(r"\\input\{(numbers_[A-Za-z0-9_]+)\}", main):
    for line in open(f"{D}/{f}.tex"):
        m = re.match(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}\s*$", line)
        if m:
            M[m.group(1)] = m.group(2)


def inline(text, depth=0):
    def rep(m):
        name = m.group(2)
        if name.startswith("numbers_"):
            return ""
        p = f"{D}/{name}" + ("" if name.endswith(".tex") else ".tex")
        return inline(open(p).read(), depth + 1) if os.path.exists(p) else m.group(0)
    return re.sub(r"\\(input|tabinput)\{([^}]+)\}", rep, text)


txt = inline(main)
for _ in range(3):                                   # macros that contain macros
    txt = re.sub(r"\\([A-Za-z]+)(\{\})?", lambda m: M[m.group(1)] if m.group(1) in M else m.group(0), txt)
open(OUT, "w").write(txt)
left = sorted(set(re.findall(r"\\([a-z]{1,2}[A-Z][A-Za-z]+)", txt)) - set(M))
print("flattened ->", OUT, "| macros replaced:", len(M), "| camelCase commands left (check):", left[:20])
