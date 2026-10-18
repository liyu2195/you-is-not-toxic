# labels/

One folder per population of the paper (Table 1). Every file is Parquet (zstd-compressed) and has one row per comment, in the
same order in every file of a folder; this order is the populations' order in the paper's pipeline and the analysis scripts
depend on it (seeded tie-breaking, bootstrap clusters), so keep it. No file contains comment text, user names or handles.

| Folder | Population | Rows | Gold labels here |
|---|---|---|---|
| `cc_threads` | Civil Comments reply threads | 1,251,487 | yes (CC0) |
| `cc_s5` | the 5% sample of Civil Comments articles (a subset of `cc_threads`) | 60,668 | yes (CC0) |
| `cad` | CAD, Contextual Abuse Dataset | 26,550 | yes (CC BY 4.0) |
| `qian_reddit` | Qian et al.'s Reddit conversations | 22,309 | no: rebuilt from the original download |
| `qian_gab` | Qian et al.'s Gab conversations | 33,776 | no: rebuilt from the original download |
| `cga` | CGA-Wiki, verified conversations | 5,340 | no: rebuilt from the original download |
| `bad` | BAD, Bot-Adversarial Dialogue | 78,874 | no: rebuilt from the original download |

`code/data/assemble_populations.py` joins a folder's files into the population file the scripts read,
`results/x14_pops/<folder>.parquet` (for the last four with the gold labels read by `code/data/rebuild_gold.py` from the
original files), and checks the result against `manifest.json`; from `cc_threads` it also writes
`results/f10_population_thread_items.parquet` (checked against the digest under `derived` in `manifest.json`).

## structure.parquet

Every folder has `id` (int64), the key of all its files, and `depth` (int64), the comment's reply depth (0 for a thread's root,
a conversation's first post or comment, a dialogue's first turn).

**`cc_threads`, `cc_s5`** (Civil Comments, train file of Jigsaw's Unintended Bias competition)

| Column | Type | Meaning |
|---|---|---|
| `id` | int64 | the comment's `id` in the Jigsaw files |
| `parent_id` | int64, nullable | the comment's `parent_id` in the Jigsaw files; null for a root |
| `thread_id` | int64 | `id` of the thread's root comment (the unit) |
| `article_id` | int64 | the comment's `article_id` (the cluster for bootstrap intervals and the article size law) |
| `depth` | int64 | number of parent links to the root |
| `gold` | int8 | 1 when `toxicity` >= 0.5 (at least half of the raters chose toxic or very toxic) |
| `toxicity` | float64 | the share of raters who chose toxic or very toxic (`target` in the train file) |
| `toxicity_annotator_count` | int64 | the comment's number of raters |

**`cad`** (CAD v1.1, `cad_v1_1.tsv`)

| Column | Type | Meaning |
|---|---|---|
| `id` | int64 | the pipeline's integer id of the entry (1 to 26,550) |
| `info_id` | string | the entry's `info_id` in CAD (a submission's title and body are `<id>-title` and `<id>-post`) |
| `parent_info_id` | string, nullable | `info_id` of the entry it replies to; null for titles, bodies and comments whose parent is not in the release |
| `thread_id` | string | `info_thread.id` (the unit and the cluster) |
| `depth` | int64 | reply depth: 0 for a submission's title and body and for the few comments without a parent in the release, 1 for a top-level comment |
| `gold` | int8 | 1 when the entry's primary labels include IdentityDirectedAbuse, AffiliationDirectedAbuse or PersonDirectedAbuse |
| `primary` | string | the entry's distinct `annotation_Primary` labels, sorted and joined by `|` |

**`qian_reddit`, `qian_gab`** (Qian et al. 2019, `reddit.csv` and `gab.csv`)

| Column | Type | Meaning |
|---|---|---|
| `id` | int64 | the pipeline's id: 10,000,000 (Reddit) or 20,000,000 (Gab) + 100 x `conversation` + `post` |
| `conversation` | int64 | 0-based row of the conversation in the CSV file (the unit and the cluster) |
| `post` | int64 | the post's number in the conversation (1, 2, ...) |
| `post_id` | string | the post's id as listed in the conversation's `id` column (a post can appear in several conversations) |
| `parent_post` | int64, nullable | number of the post it replies to (the nearest earlier post one level up); null for top-level posts |
| `depth` | int64 | the post's number of tabs minus the conversation's smallest |

**`cga`** (ConvoKit's `conversations-gone-awry-corpus`, conversations with `verified` and `pair_verified` set)

| Column | Type | Meaning |
|---|---|---|
| `id` | int64 | the pipeline's id: rank of the utterance among the corpus's non-header utterances sorted by conversation, timestamp and utterance id |
| `utterance_id` | string | ConvoKit utterance id |
| `conversation_id` | string | ConvoKit conversation id (the unit and the cluster) |
| `reply_to` | string, nullable | the utterance it replies to, when that is another non-header utterance of the corpus; a comment whose `reply_to` is itself gets no parent in the population |
| `position` | int64 | 1-based position in the conversation, in order of timestamp |
| `depth` | int64 | `position` - 1 |

**`bad`** (ParlAI `bot_adversarial_dialogue_datasets_with_persona`, all three splits)

| Column | Type | Meaning |
|---|---|---|
| `id` | int64 | 0-based line number over `train.txt`, `valid.txt` and `test.txt`, in that order (one line per turn) |
| `split` | string | train, valid or test |
| `dialogue` | int64 | 0-based number of the dialogue within its split (a line whose text holds one turn starts a dialogue) |
| `turn` | int64 | 1-based turn number (the number of turns in the line's text) |
| `speaker` | string | `speaker_to_eval` of the line: human or bot (a role, not a name) |
| `depth` | int64 | `turn` - 1 |

## labels_*.parquet

Each file has `id` and the flags `y:<labeler>` and scores `s:<labeler>` (all float64) of a group of labelers, as stored in the
population files: `labels_llm` (Claude and open LLMs), `labels_detoxify`, `labels_imsypp` and `labels_other_classifiers`.
Values are missing only for BAD's `haiku_ctx` (2,295 turns without a usable answer).

**Flags `y:`** are 1 (flagged) or 0. **Scores `s:`**: for Haiku and Sonnet the answer level (Civil Comments and BAD: 3 very toxic
/ definitely not OK, 2 toxic / probably not OK, 1 hard to say / probably OK, 0 not toxic / definitely OK; an answer without an
option letter counts as 0; the yes/no questions of CAD, Qian et al. and CGA-Wiki: 1 yes, 0 no); for the open LLMs the probability
of the toxic answers at the first answer position (p(A) + p(B); p(A) on the yes/no questions); for the classifiers their score
(Detoxify: toxicity; IMSYPP: 1 - p(acceptable); the other four: probability of the positive class).

| Labeler | Meaning | Populations |
|---|---|---|
| `haiku` | Claude Haiku 5.5 without the parent; flag = toxic or very toxic (own cut) on Civil Comments, yes on the corpora's own questions | all but `bad` |
| `haiku_ctx` | the same with the parent shown for replies whose parent has text (roots keep `haiku`) | all; on `bad` the first turn without context, later turns with the previous turn, flag = definitely or probably not OK |
| `sonnet`, `sonnet_ctx` | Claude Sonnet 5.5, the same settings | `cc_s5` (both); `cad` (`sonnet`) |
| `qwen7b` | Qwen2.5-7B-Instruct, own cut (toxic probability above 0.5) | `cc_threads` |
| `qwen7b_m`, `qwen7b_m_ctx` | Qwen2.5-7B-Instruct at gold's flag rate, without and with the parent | `cc_threads`, `cc_s5`, `cga` |
| `gemma_m`, `gemma_m_ctx` | Gemma 3 27B (4-bit) at gold's flag rate, without and with the parent | `cc_s5` (both); `qian_reddit`, `qian_gab` (`gemma_m`) |
| `detoxify_original@0.6`, `detoxify_original@m` | Detoxify `original` at its usual threshold 0.6 and at gold's flag rate (same score) | all |
| `detoxify_unbiased@0.6`, `detoxify_unbiased@m` | Detoxify `unbiased`, the same two cuts | all |
| `imsypp@argmax`, `imsypp@m` | IMSyPP/hate_speech_en: most probable class not acceptable, and gold's flag rate | all |
| `sn_rob@m` | s-nlp/roberta_toxicity_classifier at gold's flag rate | all |
| `mh_dist@m` | martin-ha/toxic-comment-model at gold's flag rate | all |
| `fb_hate@m` | facebook/roberta-hate-speech-dynabench-r4-target at gold's flag rate | all |
| `cn_off@m` | cardiffnlp/twitter-roberta-base-offensive at gold's flag rate | all |

On Civil Comments the LLMs answer the main prompt (the raters' toxicity question); on CAD, Qian et al.'s corpora, CGA-Wiki and
BAD the labels here answer each corpus's own question (abuse, hate speech, personal attack, unsafe turn). The prompts are in
`prompts/`; the paper's Appendix B gives every setting. Per-item labels of further settings are in `results/`, where the paper
scripts read them: Haiku's answers on every Civil Comments comment (`full/`), Sonnet's on the 5% sample's articles
(`second/s5/`), the 0-100 ratings of Haiku and Sonnet without the parent (`x75_rating/`), OLMo-2, Granite and Llama
(`b84/`), the toxicity question on the other corpora without the parent (`x89/labels/`), the rewrites (`x77_youfree/`,
`x74_edit/`, `x89/rewrite/`) and the relabeling in later batches (`x81/`). Other settings are in `results/` as summaries only.

## manifest.json

Per population: the number of rows, the full column list (name and dtype) of `results/x14_pops/<population>.parquet` in order,
the files of the folder, whether gold labels are included, and `digest_sha256`, a SHA-256 over the row count and every column's
name, dtype and little-endian values of the population file the paper was built from (computed and checked by
`table_digest` in `code/data/assemble_populations.py`).
