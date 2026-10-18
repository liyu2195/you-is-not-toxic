# results/

The stored results that the paper scripts in `code/paper/` read: aggregate JSON files of the analyses, the per-cell outputs
of the simulated audits, and the per-item label files the scripts recompute from. Paths are the pipeline's,
so the scripts read them unchanged. Every file is listed with the script that wrote it below; scripts named `scripts/<name>`
in the docstrings and provenance notes are `code/analysis/`, `code/data/` or `code/paper/<name>` here, and `code/<name>` is
`code/labeling/<name>`.

What differs from the pipeline's files, to keep comment text out and the size small:

- **Population files** `results/x14_pops/*.parquet` are not here; `code/data/assemble_populations.py` rebuilds them from
  `labels/` (with the gold labels of the original downloads for Qian et al., CGA-Wiki and BAD), and with them
  `results/f10_population_thread_items.parquet`, the reply-thread table of `code/data/f10_population.py` that
  `x99b_draft_numbers.py` and `f13_depth_slope_ci.py` read.
- **Per-item label files** (`*.jsonl`) keep only the fields the scripts read: the answer letter, rating or option
  probabilities, the comment or request id and any error. The raw answer text (`raw`, its first 20 characters), the token
  counts (`in_tok`, `out_tok`, `cache_read`) and the open models' top-20 token lists (`top`, `mass`) were dropped, and the
  lines are written without spaces.
- `f51_items_haiku.parquet` keeps only the columns `id`, `article`, `z` (gold) and `y` (Haiku's flag), the ones `x98d`,
  `f201_paper_fig.py` and `code/data/f121_cc_samples.py` read.
- Three text files that `x98a`, `x98f` and `x18_tables_v5.py` read are not here (`data/second/s5/shard_*.jsonl`,
  `results/x74_edit/texts.jsonl`, `results/x74_edit/label_items/shard_000.jsonl`); `code/data/rebuild_cc_texts.py` writes
  the records the scripts use from the Civil Comments download. Nor is the reverse sample's `results/x74_edit/meta_rev.parquet`,
  which `code/analysis/x74_edit.py build` rebuilds (see the main README).

`data/` (next to `results/`) holds four inputs the scripts read from the pipeline's `data/` folder: `cc_meta.parquet`
(the Civil Comments metadata of all three competition files, 1,999,514 comments, as `code/data/p01_load.py` builds it: `id`,
`tox` (toxicity fraction), `created_date`, `publication_id`, `parent_id`, `article_id`, `rating` (the platform's decision),
`toxicity_annotator_count`, `split` and `Z` (gold); no text), `cc_attr_frac.parquet` (only the ids of the comments matched to the
subtype votes of `google/civil_comments`), `cga_meta.parquet` (ids, reply-to ids and the verified flags of the CGA-Wiki
population; no labels, no text, no speakers) and `you_edit_meta.parquet` (the edit sample: ids, stratum, toxicity fraction,
depth).

`reference/` holds the number macros (`numbers_*.tex`) and seven number tables of the earlier draft, which `x98a`-`x98e` read
only to check their own output against (`same()`, `chk()`) and in which `x96_dist_v6.py` looks up a few macros, and
`figs/dose.json`, the data of Figure 1b (written by `code/paper/p31_fig_dose.py`), which `x98f` reads. The ten macro files of
the earlier draft that the arXiv source inputs are copies here; `code/paper/build_earlier_draft.sh` regenerates them into
`paper_arxiv/`.

| Path | Contents | Written by |
|---|---|---|
| `full/haiku_tox_ctx0`, `full/haiku_tox_ctx1` | Haiku's answers to the main prompt on every Civil Comments comment, without and with the parent | `code/labeling/batch_manager.py` |
| `second/s5/sonnet_tox_ctx0`, `_ctx1`; `second/s5_articles.txt`, `second/s5_ids.txt` | Sonnet's answers on the 5% sample's articles; the sampled article ids and comment ids | `batch_manager.py`; `code/data/f121_cc_samples.py` |
| `b84/<model>/shard_*_tox_ctx{0,1}.jsonl`, `b84/x84_families.json` | OLMo-2, Granite and Llama option probabilities on the 5% sample's articles; their flags and laws | `code/labeling/label_vllm.py`; `code/analysis/x84_families.py` |
| `x81/` | the relabeling of the 5% sample in later batches and the earlier note (`haiku*.jsonl`, `sonnet*.jsonl`), `prn.json`, `x81_eval.json` | `x81_youprompt_rep.py`, `x81_prn.py`, `x81_youprompt_rep_eval.py` |
| `x75_rating/` | 0-100 ratings of Haiku and Sonnet; `x75_rating_eval.json` | `x75_rating.py`, `x75_rating_eval.py` |
| `x77_youfree/` | Haiku on the 5% sample with every "you" rewritten; `x77_youfree.json` | `x77_youfree.py`, `batch_manager.py` |
| `x74_edit/` | Haiku's and Sonnet's answers on the edit and reverse samples; `x74_edit.json` | `x74_edit.py` |
| `x89/labels/`, `x89/rewrite/out_s`, `x89/rewrite/out_c` | the toxicity question on the other corpora, without and with the parent (`*_ctx1`); Sonnet's corpus-wide rewrite; the rewrite with the parent shown | `x89_other_tox.py`, `x89_rewrite.py`, with `code/labeling/x89_batch.py` |
| `x89/*.json` | `x89_other_tox.json`, `x89_cc_reference.json` (`x89_other_tox.py` with `cc_threads cc_s5`), `x89_rewrite.json`, `x89_rewrite_other.json`, `x89_tuned.json` | `x89_*.py` |
| `x74_placebo/`, `x74_decomp/`, `x74_crowd/`, `x74_bounds/`, `x74_ccc/`, `x74_eval/` | placebo statistics and laws, decompositions, crowd checks, kappa bounds, CCC, audit evaluations | `x74_placebo.py`, `x74_decomp.py`, `x74_crowd.py`, `x74_bounds.py`, `x74_ccc.py`, `x74_eval_A.py`, `x74_eval_B.py` |
| `x74_b1b3/`, `x74_b2/`, `x11_conf_item_all/`, `x11_conf_edge_all/`, `x11_conf_unit/`, `x72_item_cga/`, `x10_dev_r0/`, `x10_dev_rA5/`, `x10_dev_rA7/`, `x10_dev_rE4/`, `x14_dev_rE5/`, `x10_ceiling_I4/`, `x10_edge_ceiling/`, `x15_paper/`, `x72_eval/` | simulated audits per (population, labeler, design) cell, information ceilings, and their evaluations (`x10_dev_r0` is only listed by `x98c`; `x11_conf_unit/` holds only the unit regime's evaluation `eval_H71e.json`; `x15_paper/gap_*.json` are rewritten by `x15_paper_numbers.py`) | `x09_audit_sim.py`, `x11_combined.py`, `x10_round_eval.py`, `x10_gap.py`, `x10_ceiling.py`, `x10_edge_ceiling.py`, `x12_unit_eval.py`, `x13_confirm_eval.py`, `x13_tables.py` |
| `x55_placebo/`, `x72_placebo/` | population sizes and gold prevalences (`x08_placebo.json`) | `x08_placebo.py` |
| `x75_*`, `x76_*`, `x77_*`, `x78/`, `x79/`, `x80/`, `x82/`, `x83/`, `x85/`, `x86/`, `x87/`, `x88/`, `x90/`, `x91/`, `x92/`, `x97/`, `x98g/`, `x99/` | one JSON file per analysis, named after its script | the script of the same name in `code/analysis/` (`x85_rrinf.json` and `x85_construct_{10,20,40}.json`: `x85_converged.py`; `x83/synthd_*.json`: `x83d_synth.py`; `x83/synth_binary.json`, `synth_cat.json`: `x83_synth.py`, the second with `PROTO=cat`; `x83/x83d_par.json`: `x83d_parallel.py`; `x86/x86_within.json`: `x86_within_contagion.py`; `x77_or_check.json`: `x77_or_check.py`, with functions of `f213_or_machine_check.py`; `x80_lopo_registered.json`: `x80_lopo.py --edge-registered`) |
| `f10_population.json`, `f51_items_haiku.parquet` | the Civil Comments population before sampling | `code/data/f10_population.py`, `code/data/f51_units.py` |
| `f122_qian.json`, `f124_prompts.json`, `f129_open_model.json`, `f13_depth_slope_ci.json`, `f14_regions_haiku.json`, `f31_ccc_compare*.json`, `f81_you_edit.json`, `f96_avalle.json` | earlier analyses still quoted | `f122_qian.py`, `f124_prompts.py`, `f129_open_model.py`, `f13_depth_slope_ci.py`, `f14_regions.py`, `f31_ccc_compare.py`, `f81_you_edit_analysis.py`, `f96_avalle.py` |

The audit outputs record their arguments, including the output path on the virtual machine that ran them (`/opt/...`).
