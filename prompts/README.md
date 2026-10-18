# prompts/

Every prompt the paper uses, as sent. The prompt files were rendered from the code with the placeholders `{comment}` and
`{parent}` in place of the texts, and each is identical to the corresponding box of the paper's Appendix B. A text enters a
prompt with its leading and trailing whitespace removed and cut to its first 2,000 characters plus ` [...]` when longer
(`_q` in `code/labeling/prompts.py`); the paraphrase requests use the first 2,000 characters without stripping or marker.

| File | Prompt | Source in the code |
|---|---|---|
| `system_main.txt` | system prompt of every request except those listed with their own | `SYSTEM`, `code/labeling/prompts.py` |
| `main_no_parent.txt`, `main_with_parent.txt` | main prompt: the Civil Comments raters' four-level question | `tox_prompt` |
| `request_body_haiku.txt` | the request body (Haiku; Sonnet: `"model": "claude-sonnet-5-5"`, `"thinking": {"type": "between_tools"}`) | `code/labeling/label_claude*.py`, `batch_manager.py` |
| `rating_0_100_no_parent.txt`, `rating_0_100_with_parent.txt` | 0-100 rating under Perspective's definition (Haiku-100, Sonnet-100) | `prompt`, `code/analysis/x75_rating.py` |
| `tuned_T8_messages.txt` | T8: eight example turns before the main prompt (the example slots; the paper's box) | `shots`, `code/analysis/x89_tuned.py` |
| `tuned_examples.json` | the Civil Comments ids and answer letters of the T8 and T16 examples, in order, and the development set's ids | `results/x89/tuned/build.json` of the pipeline |
| `tuned_TR_settings.txt` | TR: the main prompt with adaptive thinking at high effort | `code/analysis/x89_tuned.py` |
| `mitigation_M1_no_parent.txt` | M1: the main prompt with an instruction about "you" (main system prompt) | `m1_prompt`, `code/analysis/x91_mitigate.py` |
| `mitigation_M2_system.txt`, `mitigation_M2_no_parent.txt` | M2: no second person in the instructions | `M2_SYSTEM`, `m2_prompt` |
| `earlier_note_no_parent.txt` | the earlier note about "you" | `toxyou_prompt` |
| `yes_no_no_parent.txt`, `yes_no_with_parent.txt` | yes/no question with Perspective's definition (flag: A) | `toxbin_prompt` |
| `reversed_scale_system.txt`, `reversed_scale_no_parent.txt`, `reversed_scale_with_parent.txt` | reversed answer scale (flag: C or D) | `SYSTEMS["tox_rev"]`, `toxrev_prompt` |
| `paraphrase_system.txt`, `paraphrase_keep_you.txt`, `paraphrase_third_person.txt` | Sonnet's paraphrases (E3 keeps "you", E4 uses the third person) | `PARA`, `PERSON`, `code/analysis/x74_edit.py` |
| `corpus_cad_abuse_no_parent.txt`, `corpus_cad_abuse_with_parent.txt` | CAD's own question, abuse (flag: A) | `cad_prompt` |
| `corpus_qian_hate_system.txt`, `corpus_qian_hate_no_parent.txt`, `corpus_qian_hate_with_parent.txt` | Qian et al.'s own question, hate speech (flag: A) | `SYSTEMS["hate"]`, `hate_prompt` |
| `corpus_cga_attack_no_parent.txt`, `corpus_cga_attack_with_parent.txt` | CGA-Wiki's own question, personal attack (flag: A) | `pa_prompt` |
| `corpus_bad_unsafe_system.txt`, `corpus_bad_unsafe_first_turn.txt`, `corpus_bad_unsafe_later_turn.txt` | BAD's own question, unsafe turn (flag: A or B); later turns with the previous turn as `{parent}` | `SYSTEMS["unsafe"]`, `unsafe_prompt` |
| `you_detector_regex.txt` | the "you" detector, a Python regular expression compiled case-insensitively | `YOU`, `code/analysis/f80_you_edit_lib.py` |
| `rewrite_rule_second_to_third.txt` | the rule that rewrites "you" in the third person (the paper's Appendix A, LaTeX markup removed) | `edit`, `code/analysis/f80_you_edit_lib.py` |
| `rewrite_rule_third_to_second.txt`, `reverse_rule_regex.txt` | the reverse rule and its pattern | `reverse_edit`, `THEY`, `code/analysis/x74_edit.py` |

The tuned prompts' few-shot examples are Civil Comments comments; this repository gives them by comment id
(`tuned_examples.json`), not as text. Each example is a user turn holding the main prompt with the example's text in place of
`{comment}`, followed by an assistant turn with its letter (B for a gold positive, D for a clearly non-toxic comment); the
last example's turn carries `"cache_control": {"type": "ephemeral"}`. T16 starts with T8's eight examples, unmarked. The
texts are in the Civil Comments train file (`comment_text`), which is released under CC0.

The main prompt is also asked, unchanged, on the other corpora ("toxicity question"), and the open LLMs receive the same
system prompts and prompts through their chat templates (Appendix B).
