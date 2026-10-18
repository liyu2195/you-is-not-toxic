#!/usr/bin/env python3
"""Package the arXiv source of the paper: one flattened main.tex plus exactly the files its build reads.

usage: python3 -I scripts/x99d_package_arxiv.py [--paper DIR] [--dist DIR] [--test-dir DIR] [--strict-words]

 1. Refuses (exit 1) when main.tex or any file it reads contains placeholder text (TBD, TODO, FIXME, [EMAIL, ...)
    outside comments.
 2. Writes DIST/arxiv/main.tex: every \\input and \\tabinput inlined, the numbers_*.tex inputs removed and every number
    macro replaced by its value (the logic of x99c_flatten.py, made aware of comments and of verbatim text such as the
    promptbox listings, which it never rewrites); every comment stripped (comment-only lines deleted, a comment after
    content cut to a bare %; \\%, verbatim environments, \\verb, \\lstinline and \\url-like arguments left alone); runs
    of more than two blank lines collapsed to two; \\pdfoutput=1 as the first line; the \\tabinput definition dropped
    once nothing uses it.
 3. Copies the figures named by \\includegraphics, main.bbl and the local .cls/.sty/.bst/.clo files the build uses, and
    pins the document class: the acmart.cls that TeX finds for the paper (kpsewhich acmart.cls, run in the paper
    directory) goes next to main.tex, after checking that the reference build loaded that same file and version; its
    \\ProvidesClass date and version go into the report, and the test build must load this copy.
 4. Refuses (exit 3) when a packaged file contains a private path, a script or file name or a submission note (LEAKS).
 5. Compiles a clean copy three times with pdflatex (no bibtex: main.bbl is used) and reports errors, undefined
    references and citations, warnings by kind, boxes and pages (exit 4 when not clean); then compares
    pdftotext -layout of the result with that of the reference build PAPER/main.pdf and lists every differing line
    with its likely source location.
 6. Writes DIST/arxiv_source.tar.gz holding the files of DIST/arxiv/ at the archive's top level.

Deterministic and rerunnable: DIST/arxiv/ and the test directory are cleared on every run, and DIST/arxiv/ and the
archive exist only after a run that passed every check. The archive has fixed metadata (sorted members, mode 0644,
uid/gid 0, mtime SOURCE_DATE_EPOCH or 1980-01-01, no name or time in the gzip header).
Exit status 2: a structural problem (unparseable numbers file, missing input or figure, unsupported command).
"""
import argparse
import bisect
import collections
import difflib
import gzip
import hashlib
import io
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unicodedata
import zlib
from pathlib import Path

ROOT = Path(os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or Path(__file__).resolve().parents[2])   # repository root

PLACEHOLDERS = ["TBD", "[REPOSITORY URL", "[EMAIL", "[FULL NAME", "TODO", "FIXME"]
LEAKS = ["/home/", "/tmp/", "gs://", "wrkspc_", "sk-ant", "hf_", "scratchpad",
         "x98", "x99", "provenance", "numbers_", "review round"] + \
    [s for s in os.environ.get("YOU_IS_NOT_TOXIC_PRIVATE_STRINGS", "").split(",") if s]   # e.g. your cloud project and bucket names
# a submission note naming the venue is refused when YOU_IS_NOT_TOXIC_VENUE_NOTE holds a regular expression for it
VENUE_NOTE = re.compile(os.environ["YOU_IS_NOT_TOXIC_VENUE_NOTE"], re.I) if os.environ.get("YOU_IS_NOT_TOXIC_VENUE_NOTE") else None
# "certify" is also a word of the paper ("we therefore certify the law"). Used as an English word in prose it is not a
# file, script or path name, so such uses are listed but do not block; --strict-words makes every occurrence block.
PROSE_WORDS = {"certify": re.compile(r"(?<![\w/\\.@:-])certify(?:ing)?(?![\w/@:-]|\.\w|\\_)", re.I)}

VERBATIM_ENVS = {"verbatim", "verbatim*", "Verbatim", "Verbatim*", "BVerbatim", "LVerbatim", "lstlisting",
                 "minted", "alltt", "comment", "filecontents", "filecontents*"}
VERB_ENV_DEF = re.compile(r"\\(?:lstnewenvironment|DefineVerbatimEnvironment|CustomVerbatimEnvironment|"
                          r"RecustomVerbatimEnvironment|newtcblisting|NewTCBListing|DeclareTCBListing)"
                          r"\s*\{\s*([A-Za-z@*]+)\s*\}")
URL_LIKE = {"url", "nolinkurl", "href"}           # their (first) argument is read verbatim: % and \ are literal there
CWORD = re.compile(r"[A-Za-z]+")
BEGIN_ARG = re.compile(r"\s*\{([^{}]*)\}")
INPUT_CMD = re.compile(r"\\(input|tabinput)\s*\{([^{}]*)\}")
UNSUPPORTED = re.compile(r"\\(include|includeonly|subfile|import|subimport|inputfrom|subinputfrom)(?![A-Za-z])")
NEWCMD = re.compile(r"\\newcommand\s*\{\\([A-Za-z]+)\}\s*\{")
GRAPHICS = re.compile(r"\\includegraphics\*?\s*(?:\[[^\]]*\])?\s*\{([^{}]+)\}")
GRAPHICSPATH = re.compile(r"\\graphicspath\s*\{((?:\s*\{[^{}]*\})*)\s*\}")
GRAPHIC_EXTS = [".pdf", ".PDF", ".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG", ".jbig2", ".jb2", ".eps", ".EPS"]
STYLE_EXTS = {".cls", ".sty", ".bst", ".clo", ".cfg", ".def", ".fd"}
USES_STYLE = re.compile(r"\\(documentclass|LoadClass|usepackage|RequirePackage|bibliographystyle)\s*(?:\[[^\]]*\])?"
                        r"\s*\{([^{}]*)\}")
TABINPUT_DEF = re.compile(r"\\makeatletter\s*\\newcommand\s*\{?\\tabinput\}?\s*\[1\]\s*"
                          r"\{\\@@input\s*#1\.tex\s*\}\s*\\makeatother")
BOX = re.compile(r"^(Overfull|Underfull) \\([hv]box) \(([^)]*)\) (.*?)(?: at lines? (\d+)(?:--(\d+))?)?$")
WARNING = re.compile(r"^(LaTeX(?: Font)?|Package \S+|Class \S+|pdfTeX) [Ww]arning(?: \([^)]*\))?: ?(.*)")
LOGFILE = re.compile(r"\(((?:\./|/)[^\s()]*)")   # TeX logs a file it opens as "(./name" or "(/abs/name"
WORD_BOX = re.compile(r'<word xMin="([^"]*)" yMin="([^"]*)" xMax="([^"]*)" yMax="([^"]*)">([^<]*)</word>')
TAR_MTIME = int(os.environ.get("SOURCE_DATE_EPOCH", "315532800"))   # 1980-01-01 00:00 UTC
TEST_MARKER = ".x99d_pkg_test"
MAX_BLANK = 2
PINNED_CLASS = "acmart"         # shipped with the package, so that arXiv's TeX Live does not change the layout
DOCCLASS = re.compile(r"\\documentclass\s*(?:\[[^\]]*\])?\s*\{([^{}]*)\}")

OUT = []


def say(*parts):
    line = " ".join(str(p) for p in parts)
    OUT.append(line)
    print(line, flush=True)


class Refuse(Exception):
    def __init__(self, code, msg):
        super().__init__(msg)
        self.code = code


# ---------------------------------------------------------------------------------------------------------- scanning

def raw_group_end(s, k, open_="{", close="}"):
    """Index just past the group opening at s[k], counting braces only (verbatim-style arguments); len(s) if open."""
    depth = 0
    for j in range(k, len(s)):
        if s[j] == open_:
            depth += 1
        elif s[j] == close:
            depth -= 1
            if depth == 0:
                return j + 1
    return len(s)


def brace_end(s, k):
    """Index just past the balanced {...} opening at s[k], skipping escaped characters; -1 when it does not close."""
    depth, i, n = 0, k, len(s)
    while i < n:
        c = s[i]
        if c == "\\":
            i += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return -1


def scan_line(s, env, verb_envs):
    """Split one source line (without its newline) into (kind, start, end) pieces of kind 'code', 'prot' (text TeX
    reads verbatim: verbatim-like environments, \\verb, \\lstinline, \\url-like arguments) or 'comment'.
    env is the verbatim-like environment the line starts in (None outside); returns (pieces, env after the line)."""
    out = []

    def add(kind, a, b):
        if a < b:
            if out and out[-1][0] == kind and out[-1][2] == a:
                out[-1] = (kind, out[-1][1], b)
            else:
                out.append((kind, a, b))

    i, n = 0, len(s)
    while i < n:
        if env is not None:
            end = "\\end{%s}" % env
            j = s.find(end, i)
            if j < 0:
                add("prot", i, n)
                break
            add("prot", i, j)
            add("code", j, j + len(end))
            i, env = j + len(end), None
            continue
        c = s[i]
        if c == "%":
            add("comment", i, n)
            break
        if c != "\\":
            j = i + 1
            while j < n and s[j] not in "\\%":
                j += 1
            add("code", i, j)
            i = j
            continue
        m = CWORD.match(s, i + 1)
        if not m:                                   # control symbol (\%, \\, \{, \ ...) or a backslash at line end
            add("code", i, min(n, i + 2))
            i += 2
            continue
        name, j = m.group(0), m.end()
        if name == "verb":
            k = j + (1 if s[j:j + 1] == "*" else 0)
            if k < n:
                e = s.find(s[k], k + 1)
                e = n if e < 0 else e + 1
                add("prot", i, e)
                i = e
                continue
        elif name == "lstinline":
            k = j
            if s[k:k + 1] == "[":
                k = raw_group_end(s, k, "[", "]")
            if k < n:
                if s[k] == "{":
                    e = raw_group_end(s, k)
                else:
                    e = s.find(s[k], k + 1)
                    e = n if e < 0 else e + 1
                add("prot", i, e)
                i = e
                continue
        elif name in URL_LIKE:
            k = j
            while k < n and s[k] in " \t":
                k += 1
            if name == "href" and s[k:k + 1] == "[":
                k = raw_group_end(s, k, "[", "]")
            if k < n and s[k] == "{":
                e = raw_group_end(s, k)
                add("prot", i, e)
                i = e
                continue
            if k < n and name != "href" and not (s[k].isalnum() or s[k] in " \t"):     # \url|...|
                e = s.find(s[k], k + 1)
                e = n if e < 0 else e + 1
                add("prot", i, e)
                i = e
                continue
        elif name == "begin":
            mm = BEGIN_ARG.match(s, j)
            if mm and mm.group(1).strip() in verb_envs:
                add("code", i, mm.end())
                i, env = mm.end(), mm.group(1).strip()
                continue
        add("code", i, j)
        i = j
    return out, env


def code_spans(lines, verb_envs, with_env=False):
    """For every line, its pieces (scan_line), tracking verbatim environments across lines; with_env=True gives
    (pieces, the verbatim-like environment the line starts in) instead."""
    env, res = None, []
    for text, _ in lines:
        start = env
        pieces, env = scan_line(text, env, verb_envs)
        res.append((pieces, start) if with_env else pieces)
    return res


def snip(s, a, b, width=50):
    lo, hi = max(0, a - width), min(len(s), b + width)
    return ("..." if lo else "") + s[lo:hi].replace("\n", " ").replace("\f", " ") + ("..." if hi < len(s) else "")


# --------------------------------------------------------------------------------------------------------- flattening

class Ctx:
    def __init__(self, paper):
        self.paper = paper
        self.verb_envs = set(VERBATIM_ENVS)
        self.files = []                 # every source file read, in order
        self.placeholders = []          # (file, line, needle, context)
        self.M, self.Morigin = {}, {}   # number macros: name -> value, name -> (file, line)
        self.numbers = []               # numbers files whose inputs were removed
        self.inlined = collections.Counter()
        self.external = []              # \input of files outside the paper directory: left as they are
        self.junctions = []             # inputs that shared their source line with other text
        self.comment_lines = 0
        self.trailing_comments = 0
        self.expanded = {}
        self.uses = collections.Counter()
        self.spacing = []               # number macros followed by a space (or the line end) in the source
        self.risks = []                 # other replacement contexts that need a look
        self.protected_names = collections.Counter()
        self.pdftotext_warnings = (collections.Counter(), collections.Counter())


def read_text(path):
    return path.read_bytes().decode("utf-8", "surrogateescape").replace("\r\n", "\n")


def split_lines(raw, rel):
    lines = raw.split("\n")
    if lines and lines[-1] == "":
        lines.pop()                     # the newline that ends the last line opens no further line
    return [(t, (rel, i)) for i, t in enumerate(lines, 1)]


def safe_rel(paper, rel, what):
    p = Path(rel)
    if p.is_absolute() or ".." in p.parts:
        raise Refuse(2, f"{what}: {rel} is outside the paper directory; arXiv needs paths inside the package")
    return paper / p


def find_placeholders(lines, ctx):
    for text, (f, i) in lines:
        for needle in PLACEHOLDERS:
            for m in re.finditer(re.escape(needle), text):
                ctx.placeholders.append((f, i, needle, snip(text, m.start(), m.end())))


def strip_comments(lines, ctx):
    """Delete comment-only lines; cut a comment after content to a bare % (the end of line stays swallowed)."""
    out, env = [], None
    for text, org in lines:
        pieces, env_next = scan_line(text, env, ctx.verb_envs)
        com = [a for kind, a, _ in pieces if kind == "comment"]
        env = env_next
        if com:
            before = text[:com[0]]
            if before.strip() == "":
                ctx.comment_lines += 1
                continue
            if text[com[0]:] != "%":
                ctx.trailing_comments += 1
            text = before + "%"
        out.append((text, org))
    return out


def load_source(rel, ctx):
    raw = read_text(safe_rel(ctx.paper, rel, "input"))
    ctx.files.append(rel)
    ctx.verb_envs.update(m.group(1) for m in VERB_ENV_DEF.finditer(raw))
    lines = strip_comments(split_lines(raw, rel), ctx)
    find_placeholders(lines, ctx)
    return lines


def read_numbers(rel, ctx):
    """Read the \\newcommand{\\name}{value} lines of a numbers file; refuse on anything else that is not a comment."""
    raw = read_text(safe_rel(ctx.paper, rel, "input"))
    ctx.files.append(rel)
    ctx.numbers.append(rel)
    kept = []
    for text, (f, i) in split_lines(raw, rel):
        pieces, _ = scan_line(text, None, ctx.verb_envs)
        code = "".join(text[a:b] for kind, a, b in pieces if kind != "comment")
        kept.append((code, (f, i)))
        s = code.strip()
        if not s:
            continue
        m = NEWCMD.match(s)
        end = brace_end(s, m.end() - 1) if m else -1
        if end < 0 or s[end:].strip():
            raise Refuse(2, f"{f}:{i}: not a one-line \\newcommand{{\\name}}{{value}}; the package drops the numbers "
                            f"files, so this line would be lost: {s[:120]}")
        name, value = m.group(1), s[m.end():end - 1]
        if name in ctx.M:
            raise Refuse(2, f"{f}:{i}: \\{name} is defined twice (first at {ctx.Morigin[name][0]}:{ctx.Morigin[name][1]})")
        ctx.M[name], ctx.Morigin[name] = value, (f, i)
    find_placeholders(kept, ctx)


def resolve_input(cmd, name, ctx, org):
    if not name:
        raise Refuse(2, f"{org[0]}:{org[1]}: empty \\{cmd}{{}}")
    safe_rel(ctx.paper, name, f"{org[0]}:{org[1]}: \\{cmd}")
    if cmd == "tabinput":                       # \tabinput{x} is \@@input x.tex
        cands = [name + ".tex"]
    else:                                       # LaTeX's \input{x} tries x.tex, then x
        cands = [name] if name.endswith(".tex") else [name + ".tex", name]
    for c in cands:
        if (ctx.paper / c).is_file():
            return c
    ctx.external.append((org, cmd, name))
    return None


def flatten(rel, ctx, depth=0):
    """The comment-free lines of a source file with every \\input / \\tabinput of a paper file inlined, as
    (text, (file, line)) pairs; the inputs of numbers_*.tex files are removed and their macros collected."""
    if depth > 30:
        raise Refuse(2, f"inputs nested more than 30 deep at {rel} (a cycle?)")
    lines = load_source(rel, ctx)
    out = []
    for (text, org), pieces in zip(lines, code_spans(lines, ctx.verb_envs)):
        code = [(a, b) for kind, a, b in pieces if kind == "code"]
        for a, b in code:
            m = UNSUPPORTED.search(text, a, b)
            if m:
                raise Refuse(2, f"{org[0]}:{org[1]}: \\{m.group(1)} is not supported by this flattener")
        hits = [m for a, b in code for m in INPUT_CMD.finditer(text, a, b)]
        if not hits:
            out.append((text, org))
            continue
        buf, pos, inlined = "", 0, False
        for m in hits:
            cmd, name = m.group(1), m.group(2).strip()
            target = resolve_input(cmd, name, ctx, org)
            if target is None:
                continue
            buf += text[pos:m.start()]
            pos = m.end()
            if Path(name).name.startswith("numbers_"):
                read_numbers(target, ctx)
                continue
            sub = flatten(target, ctx, depth + 1)
            ctx.inlined[cmd] += 1
            if buf.strip():
                out.append((buf + "%", org))    # the file began mid-line: end this line without an end-of-line space
                ctx.junctions.append((org, f"text before \\{cmd}{{{name}}}"))
            buf = ""
            out.extend(sub)
            inlined = True
        buf += text[pos:]
        if buf.strip() not in ("", "%"):
            out.append((buf, org))
            if inlined:
                ctx.junctions.append((org, "text after an inlined file"))
    return out


# ------------------------------------------------------------------------------------------------- number macros

def plain_value(v):
    """Roughly how a value reads in the PDF text (to find it there)."""
    v = v.replace("{,}", ",").replace("\\%", "%").replace("$-$", "\u2212").replace("$+$", "+")
    v = re.sub(r"\\[A-Za-z]+\*?", "", v)
    return re.sub(r"[{}$~]", "", v).strip()


def ends_with_cw(s):
    m = re.search(r"(\\+)[A-Za-z]+$", s)
    return bool(m) and len(m.group(1)) % 2 == 1


def expand(name, ctx, stack):
    if name in stack:
        raise Refuse(2, f"\\{name} is defined in terms of itself ({' -> '.join(stack + (name,))})")
    if name not in ctx.expanded:
        ctx.expanded[name] = sub_numbers(ctx.M[name], ctx, ctx.Morigin[name], stack + (name,), False)
    return ctx.expanded[name]


def sub_numbers(seg, ctx, org, stack, at_eol):
    """Replace the number macros of one code piece by their values, the way TeX would read them."""
    res, i, n, cw_end = "", 0, len(seg), -1
    while i < n:
        j = seg.find("\\", i)
        if j < 0:
            res += seg[i:]
            break
        res += seg[i:j]
        i = j
        m = CWORD.match(seg, i + 1)
        if not m:
            res += seg[i:i + 2]
            i += 2
            continue
        name, j = m.group(0), m.end()
        if name not in ctx.M:
            res += seg[i:j]
            cw_end, i = len(res), j
            continue
        val = expand(name, ctx, stack)
        if not stack:
            ctx.uses[name] += 1
        if cw_end >= 0 and res[cw_end:].strip(" \t") == "":
            prev = re.search(r"\\[A-Za-z]+[ \t]*$", res).group(0).strip()
            if val[:1].isspace():
                ctx.risks.append((org, name, f"follows {prev}: TeX keeps the value's leading space, the package loses it"))
            elif res[cw_end:] == "" and val[:1].isalpha():
                res += " "                      # keep the control word and the value apart (TeX skips this space)
                ctx.risks.append((org, name, f"follows {prev} directly; a space now separates them"))
            else:
                ctx.risks.append((org, name, f"follows {prev}: if that command takes an argument, it now gets only "
                                             f"the first token of the value"))
        if seg[j:j + 1] == "@":
            ctx.risks.append((org, name, "followed by @: under \\makeatletter TeX reads a longer control word"))
        if seg.startswith("{}", j):
            k = j + 2
            nxt = seg[k:k + 1]
            if (nxt in (" ", "\t") or (nxt == "" and at_eol)) and val and not val[-1].isspace() and not ends_with_cw(val):
                res += val                      # the braces only ended the name before a space: same tokens without
            else:
                res += val + "{}"               # kept: they separate the value from what follows, as in the source
            i = k
            cw_end = -1
            continue
        nxt = seg[j:j + 1]
        if nxt in (" ", "\t") or (nxt == "" and at_eol):
            rest = seg[j:].lstrip(" \t")
            where = ("end of line" if not rest else
                     "table cell end" if rest.startswith("&") or rest.startswith("\\\\") else "running text")
            word = re.match(r"[^\s\\{}$&%~]*", rest).group(0)
            ctx.spacing.append((org, name, where, snip(seg, i, j, 40), plain_value(val), word))
        res += val
        i = j
        cw_end = len(res) if ends_with_cw(val) else -1
    return res


def replace_numbers(lines, ctx):
    out = []
    for k, ((text, org), pieces) in enumerate(zip(lines, code_spans(lines, ctx.verb_envs))):
        n0 = len(ctx.spacing)
        parts = []
        for kind, a, b in pieces:
            seg = text[a:b]
            if kind == "code":
                seg = sub_numbers(seg, ctx, org, (), b == len(text))
            else:
                for m in re.finditer(r"\\([A-Za-z]+)", seg):
                    if m.group(1) in ctx.M:
                        ctx.protected_names[m.group(1)] += 1
            parts.append(seg)
        out.append(("".join(parts), org))
        for s in range(n0, len(ctx.spacing)):          # at the line end, the next word starts the next line
            if ctx.spacing[s][2] == "end of line" and k + 1 < len(lines):
                word = re.match(r"\s*([^\s\\{}$&%~]*)", lines[k + 1][0]).group(1)
                ctx.spacing[s] = ctx.spacing[s][:5] + (word,)
    return out


def control_words(lines, ctx):
    """Every control word in code, with the line it is on."""
    for (text, org), pieces in zip(lines, code_spans(lines, ctx.verb_envs)):
        for kind, a, b in pieces:
            if kind != "code":
                continue
            seg = text[a:b]
            for m in re.finditer(r"(\\+)([A-Za-z]+)", seg):
                if len(m.group(1)) % 2 == 1:
                    yield m.group(2), org, text


def drop_tabinput_def(lines, ctx):
    uses = [org for name, org, text in control_words(lines, ctx)
            if name == "tabinput" and not TABINPUT_DEF.search(text)]
    if uses:
        return lines, f"kept: \\tabinput is still used at {uses[0][0]}:{uses[0][1]}"
    out, dropped = [], False
    for text, org in lines:
        if not dropped and TABINPUT_DEF.search(text):
            text, dropped = TABINPUT_DEF.sub("", text, count=1), True
            if not text.strip():
                continue
        out.append((text, org))
    if not dropped:
        if any(name == "tabinput" for name, _, _ in control_words(lines, ctx)):
            return lines, "kept: unused, but not in the expected form"
        return lines, "none in the source"
    return out, "removed (no longer used)"


def collapse_blank_lines(lines, ctx):
    out, run, n = [], 0, 0
    for (text, org), (pieces, env) in zip(lines, code_spans(lines, ctx.verb_envs, with_env=True)):
        blank = text.strip() == "" and env is None      # blank lines inside verbatim text are printed: keep them
        if blank:
            run += 1
            if run > MAX_BLANK:
                n += 1
                continue
            text = ""
        else:
            run = 0
        out.append((text, org))
    return out, n


def build_main(ctx):
    lines = flatten("main.tex", ctx)
    defined = [(org, name) for name, org, text in control_words(lines, ctx)
               if name in ctx.M and re.search(r"\\(?:newcommand|renewcommand|providecommand|def|let)\s*\{?\\" + name
                                              + r"(?![A-Za-z])", text)]
    if defined:
        raise Refuse(2, "number macros redefined in the source: " +
                     ", ".join(f"\\{n} at {o[0]}:{o[1]}" for o, n in defined[:10]))
    lines = replace_numbers(lines, ctx)
    left = sorted({n for n, _, _ in control_words(lines, ctx) if n in ctx.M})
    if left:
        raise Refuse(2, "number macros left after replacement: " + ", ".join(left[:20]))
    for _, org, text in [(n, o, t) for n, o, t in control_words(lines, ctx) if n == "input"]:
        if not any(o == org for o, _, _ in ctx.external):
            raise Refuse(2, f"{org[0]}:{org[1]}: an \\input is left in the flattened text: {text[:100]}")
    lines, tabdef = drop_tabinput_def(lines, ctx)
    lines, collapsed = collapse_blank_lines(lines, ctx)
    if not re.match(r"\s*\\pdfoutput\s*=\s*1(?!\d)", lines[0][0] if lines else ""):
        lines.insert(0, ("\\pdfoutput=1", ("(added)", 0)))
    return lines, tabdef, collapsed


# --------------------------------------------------------------------------------------------------- package files

def unwrap_log(text, width=79):
    """Undo TeX's wrapping of log lines at max_print_line (79 by default)."""
    out, cur = [], ""
    for line in text.split("\n"):
        cur += line
        if len(line) != width:
            out.append(cur)
            cur = ""
    if cur:
        out.append(cur)
    return out


def log_files(lines):
    """For each log line, the innermost input file open where it starts; and every file opened."""
    stack, cur, opened, skip = [], [], [], False
    for line in lines:
        cur.append(next((f for f in reversed(stack) if f), None))
        if skip:
            skip = line.strip() != ""
            continue
        if BOX.match(line):
            skip = True                         # the box contents that follow can hold unbalanced parentheses
            continue
        i = 0
        while i < len(line):
            c = line[i]
            if c == "(":
                m = LOGFILE.match(line, i)
                if m:
                    stack.append(m.group(1))
                    opened.append(m.group(1))
                    i = m.end()
                    continue
                stack.append(None)
            elif c == ")" and stack:
                stack.pop()
            i += 1
    return cur, opened


def find_figures(lines, ctx):
    paths, gpath = [], [""]
    for (text, org), pieces in zip(lines, code_spans(lines, ctx.verb_envs)):
        for kind, a, b in pieces:
            if kind != "code":
                continue
            for m in GRAPHICSPATH.finditer(text, a, b):
                gpath += re.findall(r"\{([^{}]*)\}", m.group(1))
            for m in GRAPHICS.finditer(text, a, b):
                paths.append((m.group(1).strip(), org))
    figs = []
    for name, org in paths:
        cands = [name] if Path(name).suffix in GRAPHIC_EXTS else [name + e for e in GRAPHIC_EXTS]
        found = None
        for d in gpath:
            for c in cands:
                rel = (Path(d) / c).as_posix() if d else c
                if safe_rel(ctx.paper, rel, f"{org[0]}:{org[1]}: \\includegraphics").is_file():
                    found = rel
                    break
            if found:
                break
        if not found:
            raise Refuse(2, f"{org[0]}:{org[1]}: \\includegraphics{{{name}}}: no such file in the paper directory")
        if found not in figs:
            figs.append(found)
    return figs


def class_version(path, name):
    m = re.search(r"\\ProvidesClass\s*\{" + re.escape(name) + r"\}\s*\[([^\]]*)\]", read_text(path))
    if not m:
        raise Refuse(2, f"{path}: no \\ProvidesClass{{{name}}}[date version] line")
    return " ".join(m.group(1).split())


def logged_class(lines, name):
    """The path of NAME.cls a build loaded and the version its log reports ("Document Class: NAME date version")."""
    _, opened = log_files(lines)
    paths = [o for o in opened if Path(o).name == f"{name}.cls"]
    doc = next((x for x in lines if x.startswith(f"Document Class: {name} ")), None)
    return (paths[0] if paths else None), (" ".join(doc.split()[3:]) if doc else None)


def pin_class(paper, ref_lines, name):
    """The class file TeX loads for the paper (kpsewhich, run in the paper directory), checked against the file and
    the version the reference build logged; returns (its path, the date and version of its \\ProvidesClass)."""
    r = subprocess.run(["kpsewhich", f"{name}.cls"], cwd=paper, capture_output=True, text=True)
    found = r.stdout.strip().split("\n")[0] if r.returncode == 0 else ""
    if not found:
        raise Refuse(2, f"kpsewhich {name}.cls finds nothing")
    src = (paper / found).resolve()
    version = class_version(src, name)
    if ref_lines is not None:
        path, logged = logged_class(ref_lines, name)
        if path is None or (paper / path).resolve() != src:
            raise Refuse(2, f"the reference build loaded {path}, but kpsewhich finds {found}: not pinning a different "
                            f"{name}.cls than the one the reference used")
        if logged != version:
            raise Refuse(2, f"the reference build logged {name} [{logged}], but {src} provides [{version}]")
    return src, version


def find_styles(lines, ctx, ref_opened):
    """Local class/style files in the paper directory, and the ones the build reads."""
    local = sorted(p.relative_to(ctx.paper).as_posix() for p in ctx.paper.rglob("*")
                   if p.is_file() and p.suffix in STYLE_EXTS)
    named = set()
    for text, _ in lines:
        for m in USES_STYLE.finditer(text):
            ext = {"documentclass": ".cls", "LoadClass": ".cls", "bibliographystyle": ".bst"}.get(m.group(1), ".sty")
            named.update(x.strip() + ext for x in m.group(2).split(",") if x.strip())
    opened = {re.sub(r"^\./", "", f) for f in ref_opened if not f.startswith("/")}
    used = [f for f in local if f in named or f in opened]
    return local, used, sorted(named)


# ----------------------------------------------------------------------------------------------------------- leaks

def pdf_texts(data):
    yield "raw bytes", data.decode("latin-1")
    for k, m in enumerate(re.finditer(rb"stream\r?\n", data)):
        end = data.find(b"endstream", m.end())
        if end < 0:
            continue
        try:
            yield f"stream {k}", zlib.decompressobj().decompress(data[m.end():end]).decode("latin-1")
        except zlib.error:
            pass


def scan_leaks(name, text, strict, hits, prose, line_based=True):
    starts = [0] + [m.end() for m in re.finditer("\n", text)] if line_based else None
    found = []
    for p in LEAKS:
        for m in re.finditer(re.escape(p), text, re.I):
            found.append((p, m.start(), m.end()))
    for m in (VENUE_NOTE.finditer(text) if VENUE_NOTE else ()):
        found.append(("venue note", m.start(), m.end()))
    for p, a, b in sorted(found, key=lambda x: x[1]):
        where = f"{name}:{bisect.bisect_right(starts, a)}" if line_based else name
        if line_based:
            ls = starts[bisect.bisect_right(starts, a) - 1]
            le = text.find("\n", a)
            le = len(text) if le < 0 else le
            context = snip(text[ls:le], a - ls, b - ls, 60)
        else:
            context = re.sub(r"[^\x20-\x7e]", ".", snip(text, a, b, 40))
        ok = PROSE_WORDS.get(p)
        if ok and not strict and ok.match(text, a):
            prose.append((where, p, context))
        else:
            hits.append((where, p, context))


def leak_check(pkg, strict):
    hits, prose = [], []
    for f in sorted(pkg.rglob("*")):
        if not f.is_file():
            continue
        rel = f.relative_to(pkg).as_posix()
        scan_leaks(f"(file name) {rel}", rel, strict, hits, prose, line_based=False)
        data = f.read_bytes()
        if f.suffix.lower() == ".pdf":
            for where, t in pdf_texts(data):
                scan_leaks(f"{rel} [{where}]", t, strict, hits, prose, line_based=False)
        else:
            scan_leaks(rel, data.decode("utf-8", "surrogateescape"), strict, hits, prose)
    return hits, prose


# ---------------------------------------------------------------------------------------------------- test compile

def parse_log(lines, origins=None):
    """Errors, warnings by kind, undefined references and citations, boxes (with their source) and pages."""
    files, opened = log_files(lines)
    res = {"errors": [], "kinds": collections.Counter(), "undef_refs": [], "undef_cites": [], "rerun": [],
           "multiply": [], "boxes": [], "pages": None, "opened": opened}
    for k, line in enumerate(lines):
        if line.startswith("! "):
            ctxl = next((x for x in lines[k + 1:k + 12] if re.match(r"l\.\d+", x)), "")
            m = re.match(r"l\.(\d+)", ctxl)
            loc = ""
            if m and origins and files[k] in ("./main.tex", "main.tex"):
                o = origins[int(m.group(1)) - 1] if int(m.group(1)) <= len(origins) else None
                loc = f" [{o[0]}:{o[1]}]" if o else ""
            res["errors"].append(f"{line[2:]} {ctxl.strip()}{loc}".strip())
            res["kinds"]["error"] += 1
            continue
        m = BOX.match(line)
        if m:
            kind, box, detail, what, l1, l2 = m.groups()
            res["kinds"][f"{kind} \\{box}"] += 1
            src, l1, l2 = files[k] or "?", int(l1) if l1 else None, int(l2 or l1) if l1 else None
            src = re.sub(r"^\./", "", src)
            if l1 and origins and src == "main.tex" and l1 <= len(origins):
                a, b = origins[l1 - 1], origins[min(l2, len(origins)) - 1]
                loc = f"{a[0]}:{a[1]}" + (f"-{b[1]}" if b != a and b[0] == a[0] else "")
            elif l1:
                loc = f"{src}:{l1}" + (f"-{l2}" if l2 != l1 else "")
            else:
                loc = src
            res["boxes"].append((f"{kind} \\{box}", detail, what, loc))
            continue
        m = WARNING.match(line)
        if m:
            who, msg = m.groups()
            kind = f"{who} Warning"
            if re.search(r"Reference `[^']*' on page \S+ undefined", msg):
                res["undef_refs"].append(re.search(r"`([^']*)'", msg).group(1))
                kind += ": reference undefined"
            elif re.search(r"Citation `[^']*' on page \S+ undefined", msg):
                res["undef_cites"].append(re.search(r"`([^']*)'", msg).group(1))
                kind += ": citation undefined"
            elif re.search(r"There were undefined (references|citations)", msg):
                kind += ": " + msg.strip().rstrip(".")
            elif "Rerun" in msg or "may have changed" in msg or "has changed" in msg:
                res["rerun"].append(f"{who}: {msg.strip()}")
                kind += ": rerun requested"
            elif "multiply" in msg:
                res["multiply"].append(msg.strip())
                kind += ": " + msg.strip()[:48]
            else:
                kind += ": " + re.sub(r"`[^']*'", "`...'", re.sub(r"\d+(\.\d+)?", "N", msg.strip()))[:72]
            res["kinds"][kind] += 1
        elif line.startswith("Missing character:"):
            res["kinds"]["Missing character"] += 1
        m = re.match(r"Output written on \S+ \((\d+) pages?", line)
        if m:
            res["pages"] = int(m.group(1))
    return res


def pdf_pages(pdf):
    r = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True)
    m = re.search(r"^Pages:\s+(\d+)", r.stdout, re.M)
    return int(m.group(1)) if m else None


def prepare_test_dir(test_dir):
    if test_dir.exists():
        if any(test_dir.iterdir()) and not (test_dir / TEST_MARKER).exists():
            raise Refuse(2, f"test directory {test_dir} is not empty and was not made by this script; "
                            f"give a new or empty one")
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True)
    (test_dir / TEST_MARKER).write_text("made by x99d_package_arxiv.py; cleared on every run\n")


def test_compile(pkg, test_dir, origins):
    prepare_test_dir(test_dir)
    build = test_dir / "arxiv"
    shutil.copytree(pkg, build)
    home = test_dir / "texmf-home-empty"
    home.mkdir()
    env = {k: v for k, v in os.environ.items() if k not in ("TEXINPUTS", "BIBINPUTS", "BSTINPUTS", "TEXMFHOME")}
    env.update(TEXMFHOME=str(home), max_print_line="100000", error_line="254", half_error_line="238")
    codes = []
    for k in (1, 2, 3):
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "main.tex"], cwd=build, env=env,
                           stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=1200)
        (test_dir / f"pdflatex_run{k}.txt").write_bytes(r.stdout)
        codes.append(r.returncode)
    log = build / "main.log"
    lines = read_text(log).split("\n") if log.exists() else []
    res = parse_log(lines, origins)
    res["codes"] = codes
    res["pdf"] = build / "main.pdf"
    res["lines"] = lines
    return res


def plain_tokens(s):
    s = unicodedata.normalize("NFKC", s).lower()
    return re.findall(r"[a-z0-9]+(?:[.,][0-9]+)*", s)


def source_index(lines):
    """Word trigrams of the flattened source lines (control words dropped), for locating differing PDF lines."""
    idx = collections.defaultdict(set)
    for k, (text, _) in enumerate(lines):
        t = re.sub(r"\\[A-Za-z]+\*?", " ", text.replace("{,}", ","))
        toks = plain_tokens(re.sub(r"[{}$~\\]", " ", t))
        for g in zip(toks, toks[1:], toks[2:]):
            idx[g].add(k)
        for tok in toks:
            idx[(tok,)].add(k)
    return idx


def dehyphenate(s):
    return re.sub(r"(\w)-\n\s*(?=\w)", r"\1", unicodedata.normalize("NFKC", s))


def locate(old_text, new_text, idx, lines, ctx):
    """Likely sources of one differing hunk (texts include a line of context on each side): first the number macros
    followed by a space whose value runs into the next word in the reference text, then the source lines that share
    the most word trigrams with the words around the change."""
    old, new = dehyphenate(old_text), dehyphenate(new_text)
    res, windows = [], []
    ow, nw = old.split(), new.split()
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, ow, nw, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        o_win, n_win = " ".join(ow[max(0, i1 - 4):i2 + 4]), " ".join(nw[max(0, j1 - 4):j2 + 4])
        explained = False
        for org, name, where, _, pv, word in ctx.spacing:
            if pv and word and re.search(re.escape(pv) + re.escape(word[:4]), o_win) \
                    and re.search(re.escape(pv) + r"\s+" + re.escape(word[:4]), n_win):
                explained = True
                r = (f"{org[0]}:{org[1]} (\\{name} followed by a space in the source: the reference build ran "
                     f"{pv!r} into {word!r})")
                if r not in res:
                    res.append(r)
        if not explained:
            windows += [o_win, n_win]
    score = collections.Counter()
    for n in (3, 2):
        for s in windows:
            toks = plain_tokens(s)
            for g in zip(*(toks[k:] for k in range(n))):
                if n == 3:
                    for k in idx.get(g, ()):
                        score[k] += 1
                else:                                   # bigrams: lines holding both words
                    for k in idx.get((g[0],), set()) & idx.get((g[1],), set()):
                        score[k] += 1
        if score:
            break
    if score:
        best = max(score.values())
        picks = sorted((k for k, v in score.items() if v >= max(1, best * 0.6)), key=lambda k: (-score[k], k))[:4]
        near = list(dict.fromkeys(f"{lines[k][1][0]}:{lines[k][1][1]}" for k in sorted(picks)))
        res.append(f"near {', '.join(near)} (a change that no number macro explains)")
    elif windows:
        res.append("a change that no number macro explains; its source line was not found")
    return res


def pdftotext(pdf, out, *opts):
    r = subprocess.run(["pdftotext", *opts, str(pdf), str(out)], capture_output=True, text=True)
    if r.returncode:
        raise Refuse(4, f"pdftotext {' '.join(opts)} {pdf} failed: {r.stderr.strip()[:300]}")
    return read_text(out), collections.Counter(x for x in r.stderr.split("\n") if x.strip())


def word_boxes(html):
    pages = []
    for chunk in html.split("<page ")[1:]:
        pages.append(WORD_BOX.findall(chunk))
    return pages


def compare_layout(ref_pdf, test_pdf, test_dir):
    """Every word's bounding box (pdftotext -bbox) in both PDFs: a check of the layout, not only of the text."""
    boxes = []
    for name, pdf in (("reference", ref_pdf), ("test", test_pdf)):
        html, _ = pdftotext(pdf, test_dir / f"{name}_bbox.html", "-bbox")
        boxes.append(word_boxes(html))
    nwords = sum(len(p) for p in boxes[0])
    diffs = []
    for k in range(max(len(boxes[0]), len(boxes[1]))):
        a = boxes[0][k] if k < len(boxes[0]) else []
        b = boxes[1][k] if k < len(boxes[1]) else []
        if a != b:
            first = next((j for j in range(min(len(a), len(b))) if a[j] != b[j]), min(len(a), len(b)))
            diffs.append((k + 1, len(a), len(b), a[first] if first < len(a) else None, b[first] if first < len(b) else None))
    return nwords, diffs


def compare_text(ref_pdf, test_pdf, test_dir, lines, ctx):
    texts, warns = [], []
    for name, pdf in (("reference", ref_pdf), ("test", test_pdf)):
        t, w = pdftotext(pdf, test_dir / f"{name}.txt", "-layout")
        texts.append(t)
        warns.append(w)
    ctx.pdftotext_warnings = warns
    numbered = []
    for t in texts:
        page, row, res = 1, 0, []
        for ln in t.split("\n"):
            while ln.startswith("\f"):
                page, row, ln = page + 1, 0, ln[1:]
            row += 1
            res.append((page, row, ln))
        numbered.append(res)
    a, b = [x[2] for x in numbered[0]], [x[2] for x in numbered[1]]
    if a == b:
        return True, [], len(a)
    idx = source_index(lines)
    hunks = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        old = [numbered[0][i] for i in range(i1, i2)]
        new = [numbered[1][j] for j in range(j1, j2)]
        old_ctx = "\n".join(a[max(0, i1 - 1):i2 + 1])
        new_ctx = "\n".join(b[max(0, j1 - 1):j2 + 1])
        hunks.append((old, new, locate(old_ctx, new_ctx, idx, lines, ctx)))
    return False, hunks, len(a)


# ---------------------------------------------------------------------------------------------------------- archive

def write_tar(pkg, out):
    names = sorted(p.relative_to(pkg).as_posix() for p in pkg.rglob("*") if p.is_file())
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.USTAR_FORMAT) as tf:
        for rel in names:
            data = (pkg / rel).read_bytes()
            ti = tarfile.TarInfo(rel)
            ti.size, ti.mtime, ti.mode = len(data), TAR_MTIME, 0o644
            ti.uid = ti.gid = 0
            ti.uname = ti.gname = ""
            tf.addfile(ti, io.BytesIO(data))
    tmp = out.with_name(out.name + ".part")
    with open(tmp, "wb") as f:
        with gzip.GzipFile(filename="", mode="wb", fileobj=f, mtime=0, compresslevel=9) as gz:
            gz.write(buf.getvalue())
    os.replace(tmp, out)
    with tarfile.open(out, "r:gz") as tf:                  # read it back: members at the top level, sizes as on disk
        members = [(m.name, m.size) for m in tf.getmembers()]
    if members != [(n, (pkg / n).stat().st_size) for n in names] or any(n.startswith(("/", "arxiv/")) for n in names):
        raise Refuse(2, f"archive check failed: {members}")
    return members


# ------------------------------------------------------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--paper", default=str(ROOT / "paper_arxiv"), help="paper directory (read only)")
    ap.add_argument("--dist", default=str(ROOT / "dist"), help="output directory: DIST/arxiv/ and the archive")
    ap.add_argument("--test-dir", default=os.environ.get("ARXIV_PKG_TEST_DIR"),
                    help="scratch directory for the test compile (cleared each run; default: a new temporary one)")
    ap.add_argument("--strict-words", action="store_true",
                    help="treat every occurrence of a leak string as a leak, even an ordinary word in prose")
    args = ap.parse_args(argv)
    paper, dist = Path(args.paper).resolve(), Path(args.dist).resolve()
    pkg, tarball = dist / "arxiv", dist / "arxiv_source.tar.gz"
    test_dir = Path(args.test_dir).resolve() if args.test_dir else Path(tempfile.mkdtemp(prefix="arxiv_pkg_test_"))
    for p, what in ((dist, "--dist"), (test_dir, "--test-dir")):
        if p == paper or paper in p.parents:
            say(f"REFUSED: {what} {p} is inside the paper directory, which this script never writes")
            return 2
    if test_dir == dist or dist in test_dir.parents or test_dir in dist.parents:
        say("REFUSED: --test-dir and --dist must not contain each other")
        return 2
    if pkg.is_symlink() or tarball.is_symlink():
        say(f"REFUSED: {pkg} or {tarball} is a symbolic link")
        return 2

    def clear_outputs():
        if pkg.exists():
            shutil.rmtree(pkg)
        if tarball.exists():
            tarball.unlink()

    clear_outputs()                     # a failed run leaves no package that could pass for a current one
    try:
        code = run(paper, dist, pkg, tarball, test_dir, args.strict_words)
    except Refuse as e:
        say(f"REFUSED: {e}")
        code = e.code
    if code:
        clear_outputs()
    try:
        test_dir.mkdir(parents=True, exist_ok=True)
        (test_dir / "report.txt").write_text("\n".join(OUT) + "\n")
    except OSError:
        pass
    return code


def run(paper, dist, pkg, tarball, test_dir, strict):
    for tool in ("pdflatex", "pdftotext", "pdfinfo"):
        if not shutil.which(tool):
            raise Refuse(2, f"{tool} not found")
    if not (paper / "main.tex").is_file():
        raise Refuse(2, f"{paper}/main.tex not found")
    say(f"== arXiv package of {paper}/main.tex")

    # 1-2. read and flatten the source (placeholders are collected from every file read, outside comments)
    ctx = Ctx(paper)
    lines, tabdef, collapsed = build_main(ctx)
    bbl = paper / "main.bbl"
    if not bbl.is_file():
        raise Refuse(2, "main.bbl not found: run bibtex in the paper directory first")
    bbl_lines = strip_comments(split_lines(read_text(bbl), "main.bbl"), Ctx(paper))
    find_placeholders(bbl_lines, ctx)
    nsec = sum(1 for f in ctx.files if f.startswith("sec/"))
    ntab = sum(1 for f in ctx.files if f.startswith("tables/"))
    say(f"[1] placeholders {PLACEHOLDERS} outside comments: {len(ctx.placeholders)} in {len(ctx.files) + 1} files "
        f"(main.tex, {nsec} section files, {ntab} tables, {len(ctx.numbers)} numbers files, main.bbl)")
    if ctx.placeholders:
        for f, i, needle, c in ctx.placeholders:
            say(f"    {f}:{i}: {needle!r}: {c}")
        raise Refuse(1, f"{len(ctx.placeholders)} placeholder(s) left in the source; fill them in first")

    nuses = sum(ctx.uses.values())
    verb_blocks = collections.Counter(m.group(1) for text, _ in lines
                                      for m in re.finditer(r"\\begin\{([^{}]*)\}", text) if m.group(1) in ctx.verb_envs)
    say(f"[2] {pkg}/main.tex, {len(lines)} lines:")
    say(f"    inlined: {ctx.inlined['input']} \\input, {ctx.inlined['tabinput']} \\tabinput; removed: "
        f"{len(ctx.numbers)} numbers_*.tex inputs")
    say(f"    number macros: {len(ctx.M)} defined, {len(ctx.uses)} used, {nuses} uses replaced by their values; none left")
    say(f"    comments: {ctx.comment_lines} comment-only line(s) deleted, {ctx.trailing_comments} trailing comment(s) "
        f"cut to a bare %; verbatim text left untouched: {dict(verb_blocks) if verb_blocks else 'none'}")
    say(f"    blank lines: {collapsed} dropped from runs of more than {MAX_BLANK}; \\pdfoutput=1 on line 1; "
        f"\\tabinput definition {tabdef}")
    if ctx.protected_names:
        say(f"    number-macro names inside verbatim text, left as written: {dict(ctx.protected_names)}")
    for org, cmd, name in ctx.external:
        say(f"    left as it is (not a file of the paper directory): {org[0]}:{org[1]}: \\{cmd}{{{name}}}")
    for org, what in ctx.junctions:
        say(f"    inlined mid-line ({what}) at {org[0]}:{org[1]}: check the spacing there")
    for org, name, what in ctx.risks:
        say(f"    check {org[0]}:{org[1]}: \\{name} {what}")
    by_where = collections.Counter(s[2] for s in ctx.spacing)
    say(f"    number macros followed by a space or line end without {{}} (TeX dropped that space in the reference "
        f"build, the package keeps it): {len(ctx.spacing)}" + (f" ({dict(by_where)})" if ctx.spacing else ""))
    for org, name, where, c, _, _ in ctx.spacing:
        say(f"      {org[0]}:{org[1]}  [{where}]  {c}")

    # 3. the files the build reads
    pkg.mkdir(parents=True)
    (pkg / "main.tex").write_bytes(("\n".join(t for t, _ in lines) + "\n").encode("utf-8", "surrogateescape"))
    origins = [o for _, o in lines]
    ref_log = paper / "main.log"
    ref_lines = unwrap_log(read_text(ref_log)) if ref_log.is_file() else None
    ref = parse_log(ref_lines) if ref_lines else None
    figs = find_figures(lines, ctx)
    local, used_styles, named = find_styles(lines, ctx, ref["opened"] if ref else [])
    docclass = next((m.group(1).strip() for text, _ in lines for m in DOCCLASS.finditer(text)), None)
    pinned = None
    if docclass == PINNED_CLASS:
        src, version = pin_class(paper, ref_lines, PINNED_CLASS)
        pinned = (f"{PINNED_CLASS}.cls", src, version)
        used_styles = [f for f in used_styles if f != pinned[0]]
    for rel in figs + used_styles:
        (pkg / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(paper / rel, pkg / rel)
    shutil.copyfile(bbl, pkg / "main.bbl")
    if pinned:
        shutil.copyfile(pinned[1], pkg / pinned[0])
    say(f"[3] figures named by \\includegraphics: {figs}; main.bbl copied (bibtex output, used as it is)")
    say(f"    classes, packages and styles the source names: {named}")
    say(f"    local .cls/.sty/.bst/.clo/.cfg/.def/.fd files in the paper directory: {local if local else 'none'}; "
        f"packaged: {used_styles if used_styles else 'none'}")
    if pinned:
        say(f"    class pinned: {pinned[0]} copied next to main.tex from {pinned[1]} (kpsewhich {pinned[0]} in the paper "
            f"directory{'; the file and version the reference build loaded' if ref_lines else '; no reference log to check'})")
        say(f"    version line: \\ProvidesClass{{{PINNED_CLASS}}}[{pinned[2]}]")
    else:
        say(f"    NOTE: the document class is {docclass}, not {PINNED_CLASS}: no class pinned")
    say("    every other class and package comes from the TeX Live that compiles the package")
    if ref:
        reffigs = sorted({re.sub(r"^\./", "", m) for m in re.findall(r"pdftex\.def Info: (\S+)\s+used on input line",
                                                                     "\n".join(unwrap_log(read_text(ref_log))))})
        if reffigs != sorted(figs):
            say(f"    NOTE: the reference build read the figures {reffigs}")
        known = set(ctx.files) | set(figs) | {"main.tex", "main.aux", "main.out", "main.bbl", "main.toc", "main.lof",
                                              "main.lot", "main.nav", "main.snm", "main.vrb", f"{PINNED_CLASS}.cls"}
        extra = sorted({f for f in (re.sub(r"^\./", "", o) for o in ref["opened"] if not o.startswith("/"))
                        if f not in known and f not in used_styles})
        if extra:
            say(f"    NOTE: the reference build also read these local files, which are not packaged: {extra}")
    newest = max((paper / f).stat().st_mtime for f in ctx.files + figs + ["main.bbl"])
    ref_pdf = paper / "main.pdf"
    if not ref_pdf.is_file():
        raise Refuse(2, "the reference build main.pdf is missing")
    if ref_pdf.stat().st_mtime < newest:
        say("    NOTE: the reference main.pdf is older than some of its sources; the text comparison may show their edits")

    # 4. leaks
    hits, prose = leak_check(pkg, strict)
    say(f"[4] leak scan of {sum(1 for p in pkg.rglob('*') if p.is_file())} packaged files (names, text, PDF streams): "
        f"{len(hits)} hit(s)")
    for where, p, c in hits:
        say(f"    LEAK {p!r} at {where}: {c}")
    if prose:
        say(f"    not blocking: {len(prose)} uses of {sorted({p for _, p, _ in prose})} as an ordinary word in prose "
            f"(--strict-words blocks them):")
        for where, p, c in prose:
            say(f"      {where}: {c}")
    if hits:
        raise Refuse(3, f"{len(hits)} leak(s) in the package")

    # 5. test compile and text comparison
    say(f"[5] test compile of a clean copy in {test_dir}/arxiv: pdflatex -interaction=nonstopmode main.tex, 3 runs")
    res = test_compile(pkg, test_dir, origins)
    pages = pdf_pages(res["pdf"]) if res["pdf"].is_file() else None
    ref_pages = pdf_pages(ref_pdf)
    say(f"    exit codes {res['codes']}; pages {pages} (reference {ref_pages}); errors {len(res['errors'])}; "
        f"undefined references {len(res['undef_refs'])}; undefined citations {len(res['undef_cites'])}; "
        f"rerun requests in the last run {len(res['rerun'])}; multiply defined labels {len(res['multiply'])}")
    for e in res["errors"][:30]:
        say(f"    ERROR {e}")
    for r in res["undef_refs"] + res["undef_cites"]:
        say(f"    UNDEFINED {r}")
    for r in res["rerun"]:
        say(f"    RERUN {r}")
    pin_ok = True
    if pinned:
        tpath, tver = logged_class(res["lines"], PINNED_CLASS)
        pin_ok = tpath == f"./{pinned[0]}" and tver == pinned[2]
        say(f"    class the test build loaded: {tpath} [{tver}]" + ("" if pin_ok else "  NOT THE PINNED COPY"))
    nbox = lambda r, k: sum(1 for b in r["boxes"] if b[0].startswith(k)) if r else "?"
    say(f"    overfull boxes {nbox(res, 'Overfull')} (reference {nbox(ref, 'Overfull')}); underfull boxes "
        f"{nbox(res, 'Underfull')} (reference {nbox(ref, 'Underfull')})")
    kinds = sorted(set(res["kinds"]) | set(ref["kinds"] if ref else ()))
    say("    warnings by kind (test / reference build):" + ("" if kinds else " none"))
    for k in kinds:
        say(f"      {res['kinds'][k]:4d} / {ref['kinds'][k] if ref else '?':>4}  {k}")
    ref_boxes = collections.Counter((k, d, l) for k, d, _, l in ref["boxes"]) if ref else collections.Counter()
    new_boxes = []
    for k, d, w, l in res["boxes"]:
        same = ref_boxes[(k, d, l)] > 0
        if same:
            ref_boxes[(k, d, l)] -= 1
        elif k.startswith("Overfull"):
            new_boxes.append((k, d, l))
        say(f"    {k} ({d}) {w} at {l}" + ("  [same in the reference build]" if same else "  [NOT in the reference build]"))
    for (k, d, l), c in ref_boxes.items():
        for _ in range(c):
            say(f"    reference build only: {k} ({d}) at {l}")

    same, hunks, nlines = compare_text(ref_pdf, res["pdf"], test_dir, lines, ctx) if res["pdf"].is_file() \
        else (False, None, 0)
    if hunks is None:
        say("    text comparison: no test PDF")
    elif same:
        say(f"    pdftotext -layout: identical to the reference build ({nlines} lines)")
    else:
        nd = sum(len(o) + len(n) for o, n, _ in hunks)
        say(f"    pdftotext -layout: DIFFERS from the reference build: {len(hunks)} hunk(s), {nd} line(s) "
            f"(page:line; - reference, + package)")
        for old, new, where in hunks:
            for p, r, t in old:
                say(f"      - p{p}:{r:<3} {t}")
            for p, r, t in new:
                say(f"      + p{p}:{r:<3} {t}")
            say(f"        source: {', '.join(where) if where else 'not found'}")
    if hunks is not None:
        w_ref, w_test = ctx.pdftotext_warnings
        if w_ref or w_test:
            say(f"    pdftotext's own warnings (reference / test): " +
                "; ".join(f"{w!r} {w_ref[w]} / {w_test[w]}" for w in sorted(set(w_ref) | set(w_test))))
        nwords, ldiffs = compare_layout(ref_pdf, res["pdf"], test_dir)
        if not ldiffs:
            say(f"    pdftotext -bbox: every word's position identical to the reference build ({nwords:,} words)")
        else:
            say(f"    pdftotext -bbox: word positions differ on {len(ldiffs)} page(s) (first difference per page):")
            for page, na, nb, wa, wb in ldiffs[:40]:
                fmt = lambda w: f"{w[4]!r} at ({float(w[0]):.2f}, {float(w[1]):.2f})" if w else "none"
                say(f"      p{page}: {na} / {nb} words; first moved word: reference {fmt(wa)}, package {fmt(wb)}")
    if res["pdf"].is_file():
        out = test_dir / "test.txt"
        thits, tprose = [], []
        scan_leaks("test PDF text", read_text(out), strict, thits, tprose)
        say(f"    leak scan of the test PDF's text: {len(thits)} hit(s)" +
            (f", {len(tprose)} prose uses of {sorted({p for _, p, _ in tprose})}" if tprose else ""))
        for where, p, c in thits:
            say(f"    LEAK {p!r} at {where}: {c}")
        if thits:
            raise Refuse(3, f"{len(thits)} leak(s) in the compiled text")
    clean = (res["pdf"].is_file() and not res["errors"] and not res["undef_refs"] and not res["undef_cites"]
             and not res["rerun"] and not res["multiply"] and not new_boxes and pages and pin_ok)
    if not clean:
        raise Refuse(4, "the test compile is not clean (see above; logs in " + str(test_dir) + ")")

    # 6. archive
    members = write_tar(pkg, tarball)
    total = sum(s for _, s in members)
    digest = hashlib.sha256(tarball.read_bytes()).hexdigest()
    say(f"[6] {tarball}: {tarball.stat().st_size:,} bytes (sha256 {digest[:16]}), {len(members)} files at the top "
        f"level, {total:,} bytes unpacked:")
    for name, size in members:
        say(f"      {size:>9,}  {name}")
    say("OK" + ("" if same else f" (the text differs from the reference build in {len(hunks)} place(s); see [5])"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
