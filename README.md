# "You" Is Not Toxic: How LLM Labelers Rewrite the Web's Conversation Laws

The labels, prompts and code of the paper by Liang Yue and Eric A. Vance (University of Colorado Boulder), with scripts that
rebuild every gold-labeled population. Paper: [arXiv:2610.23446](https://arxiv.org/abs/2610.23446).

## What the paper finds

LLM labelers mistake second-person address for toxicity, and this one bias manufactures conversation laws that human labels
do not show. On 1.25 million Civil Comments news comments, Claude Haiku flags 12% of the non-toxic comments without "you" and
51% of those with four or more, so under Haiku toxicity rises with reply depth (+0.93 points per level) while under the crowd's
labels it slightly falls (-0.12). All seven LLMs tested over-flag "you", and Haiku does so on Reddit, Gab and Wikipedia talk
pages too. Agreement with human labels (Cohen's kappa) does not reveal the bias, so the paper certifies the law instead of the
labeler: Threadsafe spends a budget of human labels where they move the law and returns an unbiased estimate with a confidence
interval; a thousand human labels rule out Haiku's rising law.

## What this repository contains

| Folder | Contents |
|---|---|
| `labels/` | the seven populations: structure (original ids, parents, threads, depth), gold labels for Civil Comments and CAD, and every labeler's flags and scores; see `labels/README.md` |
| `prompts/` | every prompt the paper uses, as sent, with an index in `prompts/README.md` |
| `code/threadsafe2/` | the Threadsafe implementation: laws and placebo statistics (the paper's Algorithm 1: `pps.run_seq`, with the metrics of `laws.py`), contagion (`edges_multi.run_edge`; also `edges.py`, `gated_edges.py`), the comparators (`baselines.py`: active inference, DSL, calibrate-and-aggregate; PPI++ and uniform sampling in `laws.py`), and the unit-share, contrast and pair procedures of an earlier design (`units.py`, `contrast.py`, `pairs.py`). `code/analysis/x09_audit_sim.py` runs the simulated audits with it (designs `pps_seq4_ba` and `edge_m3_c15`) |
| `code/labeling/` | the labeling and scoring scripts (Claude by the Messages and Message Batches APIs, open LLMs by transformers and vLLM, classifiers) and `prompts.py` |
| `code/analysis/` | the analysis scripts that wrote the files in `results/`, including the simulated audits |
| `code/data/` | the scripts that built the populations and samples from the original downloads, and three scripts for this repository: `assemble_populations.py`, `rebuild_gold.py`, `rebuild_cc_texts.py` (see [Rebuilding the populations and samples](#rebuilding-the-populations-and-samples)) |
| `code/paper/` | the scripts that write every number, table and figure of the arXiv version: `x98a`-`x98f`, and the generators of the earlier draft's files that the arXiv source also inputs (`x15_paper_numbers.py`, `x16_paper_numbers.py`, `x17_numbers_v4b.py`, `x18_numbers_v5.py`, `x18_tables_v5.py`, `x95_eval_v6.py`, `x96_dist_v6.py`, `x99b_draft_numbers.py`, `f201_paper_fig.py`, `p31_fig_dose.py`, `p20b_fig_decomp.py`); the build scripts `x98z_build_arxiv.sh` and `build_earlier_draft.sh`; the flattening and packaging scripts `x99c_flatten.py`, `x99d_package_arxiv.py` |
| `results/` | the stored results those scripts read; see `results/README.md` |
| `data/` | four inputs of the scripts (Civil Comments metadata without text, the CGA-Wiki population's ids, the edit sample) |

No data file contains comment text, user names or user handles.

## Corpora

| Corpus | Source | License | Used for |
|---|---|---|---|
| Civil Comments (Borkan et al., *Nuanced Metrics for Measuring Unintended Bias with Real Data for Text Classification*, Companion Proceedings of the 2019 World Wide Web Conference) | the files of Jigsaw's Unintended Bias in Toxicity Classification competition (`train.csv.gz`, `test_public_expanded.csv.gz`, `test_private_expanded.csv.gz`), from the Hugging Face mirror `shuttie/jigsaw-unintended-bias`; subtype votes from `google/civil_comments` | CC0 | reply threads (`cc_threads`) and the 5% sample (`cc_s5`) |
| CCC (Xenos et al., *Toxicity Detection Sensitive to Conversational Context*, First Monday 2022) | GitHub `ipavlopoulos/context_toxicity` | CC0 | raters who saw the parent |
| CAD (Vidgen et al., *Introducing CAD: the Contextual Abuse Dataset*, NAACL 2021) | Zenodo record 4881008 (`cad_v1_1.tsv`) | CC BY 4.0 | `cad` |
| Qian et al. (*A Benchmark Dataset for Learning to Intervene in Online Hate Speech*, EMNLP 2019) | GitHub `jing-qian/A-Benchmark-Dataset-for-Learning-to-Intervene-in-Online-Hate-Speech` (`data/reddit.csv`, `data/gab.csv`) | none found | `qian_reddit`, `qian_gab` |
| CGA-Wiki (Zhang et al., *Conversations Gone Awry: Detecting Early Signs of Conversational Failure*, ACL 2018) | ConvoKit, corpus `conversations-gone-awry-corpus` | none found | `cga` |
| BAD (Xu et al., *Bot-Adversarial Dialogue for Safe Conversational Agents*, NAACL 2021) | ParlAI task `bot_adversarial_dialogue`, `dialogue_datasets_v0.2` with personas (`bot_adversarial_dialogue_datasets_with_persona/{train,valid,test}.txt`) | none found | `bad` |

Every population and sample is rebuilt by a script from the released files and the original datasets. For Civil Comments
and CAD, `labels/` includes the gold labels. For Qian et al.'s corpora, CGA-Wiki and BAD it includes ids, structure and our
labelers' outputs but no gold labels: their gold labels come from the original downloads through `code/data/rebuild_gold.py`,
which `code/data/assemble_populations.py` calls.

## Labelers

- **Claude Haiku 5.5 and Claude Sonnet 5.5** (`claude-haiku-5-5`, `claude-sonnet-5-5`), through Anthropic's Messages API
  (version `2023-06-01`), mostly by Message Batches: a system prompt, one user message, at most 5 output tokens, thinking
  disabled for Haiku and `between_tools` for Sonnet, no temperature or other sampling parameter. A flag is "toxic" or "very
  toxic" (own cut). Haiku-100 and Sonnet-100: the 0-100 rating, flagging as many comments as gold calls toxic. The tuned
  prompts T8 and T16 send few-shot examples; TR uses adaptive thinking at high effort with 2,048 output tokens.
- **Open LLMs**: `Qwen/Qwen2.5-7B-Instruct` (transformers, bfloat16), `RedHatAI/gemma-3-27b-it-quantized.w4a16`,
  `allenai/OLMo-2-1124-7B-Instruct`, `ibm-granite/granite-3.1-8b-instruct` and `meta-llama/Llama-3.1-8B-Instruct` (vLLM,
  temperature 0, top-20 log probabilities): the probability of the toxic answers at the first answer position, flagged at
  gold's flag rate (Qwen also at its own cut, probability above one half).
- **Classifiers**: Detoxify `original` and `unbiased` (threshold 0.6), `IMSyPP/hate_speech_en` (most probable class not
  acceptable), `s-nlp/roberta_toxicity_classifier`, `martin-ha/toxic-comment-model`,
  `facebook/roberta-hate-speech-dynabench-r4-target` and `cardiffnlp/twitter-roberta-base-offensive` (0.5), each also at gold's
  flag rate.
- **+ parent**: every LLM also labels each reply with the comment it replies to shown.

The paper's Appendix B gives every setting; `prompts/` has every prompt.

## Rebuilding the populations and samples

Every population and sample is rebuilt by a script from the released files and the original datasets. We reran each script
below from a copy of this repository and the original downloads: the populations match the digests in `labels/manifest.json`,
`rebuild_cc_texts.py` writes exactly the pipeline's records, and the other scripts write the files listed byte for byte as the
pipeline did.

| What | Script | Reads |
|---|---|---|
| the seven populations `results/x14_pops/*.parquet`, and `results/f10_population_thread_items.parquet` | `code/data/assemble_populations.py`; the gold labels of Qian et al.'s corpora, CGA-Wiki and BAD through `code/data/rebuild_gold.py` | `labels/`, and the downloads of Qian et al., CGA-Wiki and BAD |
| all 1,999,514 Civil Comments as text shards (`id`, `text`, `parent_id`, `parent_text`) | `code/data/f01_full_items.py CIVIL_COMMENTS_DIR OUT_DIR` | the Civil Comments download |
| the 10% and 5% samples of Civil Comments articles (`results/second/s{10,5}_{articles,ids}.txt`) and the 5% sample's text shards (`data/second/s5/`) | `code/data/f121_cc_samples.py SHARD_DIR data/second` | `results/f51_items_haiku.parquet`, the text shards |
| the edit sample (`data/you_edit_meta.parquet`, seed 80) | `code/data/f80_you_edit.py SHARD_DIR OUT_DIR` | `data/cc_meta.parquet`, the text shards |
| the reverse sample (`results/x74_edit/meta_rev.parquet`, seed 7410) and both samples' texts (`results/x74_edit/texts.jsonl`) | `code/analysis/x74_edit.py build SHARD_DIR` | `data/cc_meta.parquet`, `data/you_edit_meta.parquet`, the text shards |
| the texts the paper scripts read: the 5% sample, and the edit sample as written and with the rule rewrite | `code/data/rebuild_cc_texts.py CIVIL_COMMENTS_DIR` | `labels/cc_s5`, `data/you_edit_meta.parquet`, the Civil Comments download |
| CGA-Wiki's items and metadata (`data/cga_meta.parquet` with the gold labels, `data/cga_items.jsonl`) | `code/data/f70_cga_items.py CORPUS_DIR SHARD_DIR` | ConvoKit's `conversations-gone-awry-corpus` |
| Qian et al.'s items and metadata (`data/qian/`) | `code/data/f120_qian_items.py QIAN_DATA_DIR` | `reddit.csv`, `gab.csv` |

`SHARD_DIR` is the folder `f01_full_items.py` writes. What these scripts write from the downloads holds comment texts, gold
labels and, for CGA-Wiki, speaker names; `.gitignore` lists those files, and `f70_cga_items.py` overwrites the released
`data/cga_meta.parquet` with the full metadata (`git checkout data/cga_meta.parquet` restores it). Keep these files out of any
copy you share.

## Recomputing the paper's numbers

Requirements: Python 3.13 (tested with 3.13.16) and

    pip install numpy==2.5.3 pandas==3.0.5 pyarrow==26.0.0 scipy==1.18.1 matplotlib==3.11.2 scikit-learn==1.9.1

The scripts read the repository root from the environment variable `YOU_IS_NOT_TOXIC_ROOT` (default: the repository root,
found from the script's location) and run with `python3 -I`. Then:

    export YOU_IS_NOT_TOXIC_ROOT=$PWD
    # 1. the populations results/x14_pops/*.parquet (Qian et al., CGA-Wiki and BAD with the gold labels of their downloads)
    python3 -I code/data/assemble_populations.py --qian QIAN_DATA_DIR --cga CGA_CORPUS_DIR --bad BAD_DIR
    # 2. the Civil Comments texts of the 5% sample and the edit sample (from the CC0 download; written to data/ and results/)
    python3 -I code/data/rebuild_cc_texts.py CIVIL_COMMENTS_DIR
    # 3. every number, table and figure the arXiv source inputs: paper_arxiv/numbers_*.tex, paper_arxiv/tables/, paper_arxiv/figs/
    sh code/paper/x98z_build_arxiv.sh

`QIAN_DATA_DIR` holds `reddit.csv` and `gab.csv`, `CGA_CORPUS_DIR` is ConvoKit's `conversations-gone-awry-corpus` folder,
`BAD_DIR` holds `train.txt`, `valid.txt` and `test.txt`, and `CIVIL_COMMENTS_DIR` holds the three competition files. The
assembler checks each population against the digest of the file the paper was built from. `x98z_build_arxiv.sh` runs
`x98a`-`x98f`, which write `numbers_arxiv_{a,b,c,d,f}.tex`, 42 tables and two figures, and then `build_earlier_draft.sh`, which
writes the files of the earlier draft that the arXiv source also inputs: ten macro files (`numbers_fig`, `numbers_v3`,
`numbers_v4`, `numbers_v4b`, `numbers_v5`, `numbers_v5_examples`, `numbers_v6_tables`, `numbers_v6_eval`, `numbers_v6_dist`,
`numbers_v7_fix`), 18 tables (the arXiv source inputs two of them, `llms_v6.tex` and `confirm_v6.tex`) and two figures
(`figs/dose.pdf`, `figs/decomp.pdf`). Together these are every file the arXiv source inputs: 15 macro files, 40 tables and four
figures. With all inputs, they come out byte for byte as in the paper's build (we checked this from a copy of this repository
and the original downloads), with three exceptions that change nothing the paper prints: `numbers_v5.tex` is written as the
earlier draft had it, and the arXiv source's copy differs in two text macros that the arXiv text does not print
(`\vAppendixGeneral`, `\vOpenLLMSettings`); the first line of `numbers_v7_fix.tex`, a comment, names the script by its
released name `x99b_draft_numbers.py`; and `figs/dose.pdf` and `figs/decomp.pdf` differ only in the creation date matplotlib
writes into a PDF. All 107 macros that the arXiv text prints from the earlier draft's files are identical. The scripts stop at
the first one that fails; what each needs:

| Script | Writes | Needs besides this repository |
|---|---|---|
| `x98a_escalation.py` | `numbers_arxiv_a.tex`, 12 tables, `figs/oppoint_grid.pdf` | the Civil Comments download (step 2) |
| `x98b_size_contagion.py` | `numbers_arxiv_b.tex`, 12 tables | the downloads of Qian et al., CGA-Wiki and BAD (step 1) |
| `x98c_agree_eval.py` | `numbers_arxiv_c.tex`, 12 tables, `figs/kappa_distortion.pdf` | the BAD download (step 1) |
| `x98d_arxiv_numbers.py` | `numbers_arxiv_d.tex`, `tables/xe_pops.tex` | nothing |
| `x98e_split_tables.py` | five tables split from those of `x98a` and `x98b` | `x98a` and `x98b` run first |
| `x98f_arxiv_review.py` | `numbers_arxiv_f.tex` | the CGA-Wiki download (step 1) and the Civil Comments download (step 2) |
| `x15_paper_numbers.py` | `numbers_v3.tex`, five tables; reruns `code/analysis/x10_gap.py`, which rewrites `results/x15_paper/gap_*.json` (identically) | nothing |
| `x16_paper_numbers.py` | `numbers_v4.tex`, two tables | step 1 |
| `x17_numbers_v4b.py` | `numbers_v4b.tex`, three tables | step 1, and `x15` run first |
| `x18_numbers_v5.py` | `numbers_v5.tex` | step 1 |
| `x18_tables_v5.py` | `numbers_v5_examples.tex`, `numbers_v6_tables.tex`, seven tables (among them `llms_v6.tex`, and `examples_v5.tex` with three Civil Comments pairs, which it also prints) | step 2 |
| `x95_eval_v6.py` | `numbers_v6_eval.tex`, `tables/confirm_v6.tex` | nothing |
| `x96_dist_v6.py` | `numbers_v6_dist.tex` | step 1 |
| `x99b_draft_numbers.py` | `numbers_v7_fix.tex` | step 1 |
| `f201_paper_fig.py` | `numbers_fig.tex` (and `figs/escalation.pdf`, which the arXiv version does not use) | nothing |
| `p31_fig_dose.py` | `figs/dose.pdf` (Figure 1) and its data `figs/dose.json` | nothing |
| `p20b_fig_decomp.py` | `figs/decomp.pdf` (Figure 2) | nothing |

The provenance files that `x98a`-`x98f` write (`paper_arxiv/provenance_x*.md`) map every macro to its source file and key.
They also check their values against the macros of the earlier draft (`results/reference/numbers_*.tex`). The docstrings of the
earlier draft's generators name the folders that draft used (`paper_v3/`, `paper_v4/`, `paper_v5/`); here they all write to
`paper_arxiv/`. `x98z_build_arxiv.sh` compiles the PDF when the paper's LaTeX source is put into `paper_arxiv/`;
`x99c_flatten.py` and `x99d_package_arxiv.py` work on that source.

## Rebuilding results from the labels

The scripts in `code/analysis/` wrote the files in `results/`; each script's docstring gives its usage, inputs and seeds, and
`results/README.md` lists the script behind every file. After step 1, these 33 rerun from this repository and reproduce
their stored files byte for byte (we checked each): `f13_depth_slope_ci`, `x74_bounds`, `x74_crowd`, `x74_decomp`,
`x75_decomp`, `x75_onerater`, `x75_onerater_decomp`, `x75_within`, `x76_rethreshold`, `x76_select`, `x77_correct`,
`x77_rr_noise`, `x77_units`, `x78_frechet`, `x78_pubs`, `x79_agree`, `x80_magnitude`, `x80_matched`, `x80_mech2`, `x80_noise`,
`x83_synth` (both protocols), `x84_families`, `x85_human_ref`, `x86_decisions`, `x86_fresh`, `x87_within_null`,
`x88_breakdown`, `x88_manyraters`, `x88_noisefree_depth`, `x88_rerun`, `x92_matched`, `x97_decomp_replies`, `x99_ctx_replies`;
`x86_within_contagion` reproduces its file except for the last digit of one value, and `x77_or_check` except for the run
times it records. With the text shards below, `x98g_qwen_dose` reproduces its file too, and with the reverse sample rebuilt
by `x74_edit.py build`, so do `x75_edit_gaps` and `x79_songap`. The others need per-item files of the pipeline that this
repository does not include (for example the 0-100 ratings with the parent, the raw label shards behind `labels/`, the
per-cell files of the unit regime), or call the APIs and models again. Scripts that read comment texts take a folder of text
shards as an argument (`code/data/f01_full_items.py` writes them from the Civil Comments download); `x90_dose.py`,
`x92_construct.py`, `x89_other_tox.py` and `x98g_qwen_dose.py` read them from `$YOU_IS_NOT_TOXIC_ROOT/scratch/full`. Scripts
that call the Claude API need `ANTHROPIC_API_KEY` (and optionally `ANTHROPIC_WORKSPACE_ID`); the open models need a GPU with
transformers, torch and vLLM; the classifiers need detoxify and transformers.

Environment variables: `YOU_IS_NOT_TOXIC_ROOT` (repository root), `ANTHROPIC_API_KEY`, `ANTHROPIC_WORKSPACE_ID` (optional,
sent as the `anthropic-workspace-id` header only when set), `YOU_IS_NOT_TOXIC_PRIVATE_STRINGS` (optional, comma-separated
strings that `x99d_package_arxiv.py` refuses to package), `YOU_IS_NOT_TOXIC_VENUE_NOTE` (optional, a regular expression
for a submission note that `x99d_package_arxiv.py` refuses). The batch runners' optional `--bucket` mirrors their state to a
cloud bucket with `gcloud storage cp`.

## Licenses

- Code (`code/`, `prompts/`): MIT, see `LICENSE`.
- Our labels and scores (`labels/`, `results/`): CC BY 4.0, see `LICENSE-DATA.md`. The fields that come from an original
  dataset keep its license (Civil Comments: CC0; CAD: CC BY 4.0); `LICENSE-DATA.md` lists them.

## How to cite

Paper: https://arxiv.org/abs/2610.23446

```bibtex
@misc{yue2026you,
  title         = {``{Y}ou'' Is Not Toxic: How {LLM} Labelers Rewrite the Web's Conversation Laws},
  author        = {Yue, Liang and Vance, Eric A.},
  year          = {2026},
  eprint        = {2610.23446},
  archivePrefix = {arXiv},
  primaryClass  = {cs.SI},
  doi           = {10.48550/arXiv.2610.23446},
  url           = {https://arxiv.org/abs/2610.23446}
}
```
