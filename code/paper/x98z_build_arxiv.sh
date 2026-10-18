#!/bin/sh
# arXiv version: regenerate every number, table and figure from the stored results, then compile: x98a-x98f write
# numbers_arxiv_{a,b,c,d,f}.tex, 42 tables and two figures; build_earlier_draft.sh writes the files of the earlier draft
# that the arXiv source also inputs (ten macro files, two of their tables, two figures).
# usage: sh code/paper/x98z_build_arxiv.sh   (root: $YOU_IS_NOT_TOXIC_ROOT, default the repository root; logs in paper_arxiv/build_*.log)
# The LaTeX steps run only when the paper's source (main.tex, sec/, refs.bib, ...) has been put into paper_arxiv/.
set -e
cd "${YOU_IS_NOT_TOXIC_ROOT:-$(dirname "$0")/../..}"
mkdir -p paper_arxiv
python3 -I code/paper/x98a_escalation.py   > paper_arxiv/build_x98a.log 2>&1
python3 -I code/paper/x98b_size_contagion.py > paper_arxiv/build_x98b.log 2>&1
python3 -I code/paper/x98c_agree_eval.py   > paper_arxiv/build_x98c.log 2>&1
python3 -I code/paper/x98d_arxiv_numbers.py > paper_arxiv/build_x98d.log 2>&1
python3 -I code/paper/x98e_split_tables.py > paper_arxiv/build_x98e.log 2>&1
python3 -I code/paper/x98f_arxiv_review.py > paper_arxiv/build_x98f.log 2>&1
sh code/paper/build_earlier_draft.sh
cd paper_arxiv
[ -f main.tex ] || { echo "numbers written to paper_arxiv/; no main.tex there, so no PDF built"; exit 0; }
pdflatex -interaction=nonstopmode main.tex > build_tex1.log 2>&1 || true
bibtex main > build_bib.log 2>&1 || true
# the ACM style adds a period after every title; drop it after a title that ends with a question mark
python3 -I -c "import re; p='main.bbl'; s=open(p).read(); n=s.count('?}.'); open(p,'w').write(s.replace('?}.', '?}')); print('bbl: removed', n, 'periods after question marks')" >> build_bib.log
pdflatex -interaction=nonstopmode main.tex > build_tex2.log 2>&1 || true
pdflatex -interaction=nonstopmode main.tex > build_tex3.log 2>&1 || true
grep -c "Warning" main.log || true
