#!/bin/sh
# The files of the earlier draft that the arXiv source also inputs, regenerated from the stored results into paper_arxiv/
# (logs in paper_arxiv/build_*.log): the ten macro files numbers_fig, numbers_v3, numbers_v4, numbers_v4b, numbers_v5,
# numbers_v5_examples, numbers_v6_tables, numbers_v6_eval, numbers_v6_dist and numbers_v7_fix, the earlier draft's tables
# (the arXiv source inputs tables/llms_v6.tex and tables/confirm_v6.tex), and the figures figs/dose.pdf and figs/decomp.pdf.
# x98z_build_arxiv.sh runs it; it needs code/data/assemble_populations.py and code/data/rebuild_cc_texts.py run first
# (x18_tables_v5.py reads the edit sample's texts and writes tables/examples_v5.tex, three Civil Comments pairs, and prints them).
# usage: sh code/paper/build_earlier_draft.sh   (root: $YOU_IS_NOT_TOXIC_ROOT, default the repository root)
set -e
cd "${YOU_IS_NOT_TOXIC_ROOT:-$(dirname "$0")/../..}"
mkdir -p paper_arxiv/tables paper_arxiv/figs
python3 -I code/paper/x15_paper_numbers.py   > paper_arxiv/build_x15.log 2>&1      # numbers_v3.tex (and results/x15_paper/gap_*.json)
python3 -I code/paper/x16_paper_numbers.py   > paper_arxiv/build_x16.log 2>&1      # numbers_v4.tex
python3 -I code/paper/x17_numbers_v4b.py     > paper_arxiv/build_x17.log 2>&1      # numbers_v4b.tex (reads x15's tables/n80_rows.json)
python3 -I code/paper/x18_numbers_v5.py      > paper_arxiv/build_x18n.log 2>&1     # numbers_v5.tex
python3 -I code/paper/x18_tables_v5.py       > paper_arxiv/build_x18t.log 2>&1     # numbers_v5_examples.tex, numbers_v6_tables.tex, tables/llms_v6.tex
python3 -I code/paper/x95_eval_v6.py         > paper_arxiv/build_x95.log 2>&1      # numbers_v6_eval.tex, tables/confirm_v6.tex
python3 -I code/paper/x96_dist_v6.py         > paper_arxiv/build_x96.log 2>&1      # numbers_v6_dist.tex
python3 -I code/paper/x99b_draft_numbers.py  > paper_arxiv/build_x99b.log 2>&1     # numbers_v7_fix.tex
python3 -I code/paper/f201_paper_fig.py . paper_arxiv > paper_arxiv/build_f201.log 2>&1   # numbers_fig.tex
python3 -I code/paper/p31_fig_dose.py paper_arxiv/figs/dose.pdf > paper_arxiv/build_p31.log 2>&1        # figs/dose.pdf (Figure 1)
python3 -I code/paper/p20b_fig_decomp.py paper_arxiv/figs/decomp.pdf > paper_arxiv/build_p20b.log 2>&1  # figs/decomp.pdf (Figure 2)
echo "files of the earlier draft written to paper_arxiv/"
