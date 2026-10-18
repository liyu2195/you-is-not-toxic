# Data license

## The labels and scores we produced: CC BY 4.0

The labelers' flags and scores in `labels/*/labels_*.parquet`, the stored results in `results/`, and the other files we
computed (structure columns such as thread ids and depths, `labels/manifest.json`) are licensed under the
[Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
Please cite the paper (see `README.md`) when you use them.

The code, including the prompts in `prompts/` and `code/labeling/prompts.py`, is under the MIT license in `LICENSE`. The main
prompt follows the question that the Civil Comments raters answered, and the 0-100 rating prompt and the yes/no prompt quote
Perspective's definition of toxicity (both by Jigsaw).

## The original datasets keep their own licenses

This repository contains no comment text of any corpus. Where a file repeats fields of an original dataset, those fields
keep the dataset's license.

| Corpus | License | What this repository contains from it |
|---|---|---|
| Civil Comments (Borkan et al. 2019; Jigsaw Unintended Bias in Toxicity Classification) | CC0 (stated by its TensorFlow Datasets release) | comment, parent, article and publication ids, creation dates, toxicity fractions, rater counts, platform ratings and the gold labels derived from them (`labels/cc_threads`, `labels/cc_s5`, `data/`) |
| CCC (Xenos et al. 2022) | CC0 (stated by its GitHub release) | aggregate results only |
| CAD, the Contextual Abuse Dataset (Vidgen et al. 2021; Zenodo record 4881008) | CC BY 4.0 | entry ids, parent and thread ids, primary labels and the gold labels derived from them (`labels/cad`); attribution to Vidgen et al. is required |
| Qian et al. (2019), Reddit and Gab | no license found | conversation row numbers, post numbers, post ids and depths only (`labels/qian_reddit`, `labels/qian_gab`); no text and no labels |
| CGA-Wiki, Conversations Gone Awry (Zhang et al. 2018; ConvoKit) | no license found | utterance, conversation and reply-to ids and positions only (`labels/cga`, `data/cga_meta.parquet`); no text, no labels, no speaker names |
| BAD, Bot-Adversarial Dialogue (Xu et al. 2021; ParlAI) | no license found | split, dialogue and turn numbers and the speaker role (human or bot) only (`labels/bad`); no text and no labels |

For the last three corpora, `code/data/rebuild_gold.py` reads the gold labels from your own download of the original
files; their use is governed by the terms of those releases.
