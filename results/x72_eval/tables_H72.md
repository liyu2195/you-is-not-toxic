### Items: validity at B ≥ 1,000 (coverage of the full-gold value)

| Design | Laws: mean | Laws: min | Placebo: mean | Placebo: min | Share of entries ≥ 0.90 | Entries |
|---|---|---|---|---|---|---|
| v3 items (`pps_seq4_ba`) | 0.949 | 0.916 | 0.949 | 0.916 | 1.000 | 892 |
| v2 (`upgrade_lw`) | 0.948 | 0.888 | 0.948 | 0.890 | 0.993 | 892 |
| PPI++ | 0.939 | 0.754 | 0.937 | 0.684 | 0.942 | 892 |
| Uniform | 0.939 | 0.802 | 0.945 | 0.862 | 0.964 | 892 |

### Items: median width ratio of v3 at B ≥ 1,000

| Metric | vs v2 | vs PPI++ | vs uniform | Entries |
|---|---|---|---|---|
| Depth law | 0.87 | 0.72 | 0.60 | 206 |
| Thread size law | 0.71 | 0.62 | 0.45 | 206 |
| Article size law | 0.73 | 0.41 | 0.33 | 68 |
| Placebo depth slope | 0.89 | 0.51 | 0.35 | 206 |
| Placebo size slope | 0.75 | 0.43 | 0.25 | 206 |

### Edges: validity at B ≥ 2,000 (coverage of the full-gold value)

| Design | Laws: mean | Laws: min | Placebo: mean | Placebo: min | Share of entries ≥ 0.90 | Entries |
|---|---|---|---|---|---|---|
| v3 edges (`edge_m3_c15`) | 0.949 | 0.914 | 0.953 | 0.926 | 1.000 | 384 |
| v2 (`upgrade_lw`) | 0.949 | 0.910 | 0.953 | 0.918 | 1.000 | 384 |
| PPI++ | 0.950 | 0.916 | 0.951 | 0.922 | 1.000 | 384 |
| Uniform | 0.950 | 0.926 | 0.951 | 0.920 | 1.000 | 384 |

### Edges: median width ratio of v3 at B ≥ 2,000

| Metric | vs v2 | vs PPI++ | vs uniform | Entries |
|---|---|---|---|---|
| Reply risk ratio | 0.91 | 0.85 | 0.80 | 192 |
| Placebo risk ratio | 0.89 | 0.40 | 0.40 | 192 |

### The price of the diagnostic: n80, the smallest budget at which ≥ 80% of audits detect the labeler's distortion

Rows where some design reaches 80% within the budget grid; '>' means not reached by the largest budget.

| Population | Labeler | Law | Labeler's value | Gold value | v3 | v2 | PPI++ | Uniform |
|---|---|---|---|---|---|---|---|---|
| bad | cn_off@m | Reply risk ratio | 1.02 | 1.21 | 4,000 | 4,000 | 4,000 | 4,000 |
| bad | cn_off@m | Thread size law | +0.0306 | +0.0591 | 8,000 | >8,000 | >8,000 | >8,000 |
| bad | detoxify_original@m | Reply risk ratio | 0.92 | 1.21 | 2,000 | 2,000 | 2,000 | 2,000 |
| bad | detoxify_original@m | Depth law | +0.0034 | +0.0102 | 4,000 | 4,000 | 4,000 | 8,000 |
| bad | fb_hate@m | Reply risk ratio | 0.99 | 1.21 | 4,000 | 4,000 | 4,000 | 4,000 |
| bad | fb_hate@m | Depth law | -0.0040 | +0.0102 | 1,000 | 1,000 | 1,000 | 2,000 |
| bad | fb_hate@m | Thread size law | +0.0208 | +0.0591 | 4,000 | >8,000 | >8,000 | >8,000 |
| bad | haiku_ctx | Reply risk ratio | 0.82 | 1.21 | 1,000 | 500 | 1,000 | 1,000 |
| bad | haiku_ctx | Depth law | +0.0044 | +0.0102 | 4,000 | 4,000 | 4,000 | 8,000 |
| bad | imsypp@m | Reply risk ratio | 0.98 | 1.21 | 4,000 | 4,000 | 4,000 | 4,000 |
| bad | imsypp@m | Thread size law | +0.0014 | +0.0591 | 2,000 | 8,000 | >8,000 | >8,000 |
| bad | mh_dist@m | Reply risk ratio | 1.09 | 1.21 | 16,000 | 16,000 | 16,000 | 16,000 |
| bad | mh_dist@m | Depth law | +0.0036 | +0.0102 | 4,000 | 4,000 | 4,000 | 8,000 |
| bad | mh_dist@m | Thread size law | +0.0317 | +0.0591 | 8,000 | >8,000 | >8,000 | >8,000 |
| bad | sn_rob@m | Reply risk ratio | 0.84 | 1.21 | 1,000 | 1,000 | 1,000 | 1,000 |
| bad | sn_rob@m | Depth law | +0.0032 | +0.0102 | 4,000 | 4,000 | 4,000 | 8,000 |
| bad | sn_rob@m | Thread size law | +0.0311 | +0.0591 | 8,000 | >8,000 | >8,000 | >8,000 |
| cad | cn_off@m | Reply risk ratio | 1.58 | 2.32 | 4,000 | 2,000 | 2,000 | 4,000 |
| cad | cn_off@m | Thread size law | +0.0241 | +0.0164 | 4,000 | 8,000 | >8,000 | >8,000 |
| cad | detoxify_original@0.6 | Reply risk ratio | 1.55 | 2.32 | 4,000 | 2,000 | 2,000 | 2,000 |
| cad | detoxify_original@0.6 | Thread size law | +0.0212 | +0.0164 | 8,000 | >8,000 | >8,000 | >8,000 |
| cad | fb_hate@m | Reply risk ratio | 1.65 | 2.32 | 4,000 | 4,000 | 4,000 | 4,000 |
| cad | fb_hate@m | Thread size law | +0.0215 | +0.0164 | 8,000 | 8,000 | >8,000 | >8,000 |
| cad | haiku | Reply risk ratio | 1.60 | 2.32 | 4,000 | 2,000 | 2,000 | 4,000 |
| cad | haiku | Thread size law | +0.0277 | +0.0164 | 2,000 | 2,000 | 8,000 | 8,000 |
| cad | imsypp@m | Reply risk ratio | 1.59 | 2.32 | 4,000 | 4,000 | 2,000 | 4,000 |
| cad | imsypp@m | Thread size law | +0.0223 | +0.0164 | 8,000 | 8,000 | >8,000 | >8,000 |
| cad | mh_dist@m | Reply risk ratio | 1.55 | 2.32 | 4,000 | 2,000 | 2,000 | 2,000 |
| cad | mh_dist@m | Thread size law | +0.0238 | +0.0164 | 4,000 | 8,000 | >8,000 | >8,000 |
| cad | sn_rob@m | Reply risk ratio | 1.57 | 2.32 | 2,000 | 2,000 | 4,000 | 2,000 |
| cad | sn_rob@m | Thread size law | +0.0249 | +0.0164 | 4,000 | 4,000 | 8,000 | >8,000 |
| cad | sonnet | Reply risk ratio | 1.62 | 2.32 | 2,000 | 2,000 | 4,000 | 4,000 |
| cad | sonnet | Thread size law | +0.0286 | +0.0164 | 2,000 | 2,000 | 4,000 | 8,000 |
| cc_s5 | cn_off@m | Reply risk ratio | 2.82 | 2.28 | 16,000 | 16,000 | >16,000 | >16,000 |
| cc_s5 | fb_hate@m | Article size law | +0.0031 | +0.0010 | 8,000 | >8,000 | >8,000 | >8,000 |
| cc_s5 | gemma_m | Article size law | +0.0043 | +0.0010 | 8,000 | >8,000 | >8,000 | >8,000 |
| cc_s5 | mh_dist@m | Reply risk ratio | 2.86 | 2.28 | 8,000 | 16,000 | 16,000 | >16,000 |
| cc_s5 | sn_rob@m | Reply risk ratio | 2.69 | 2.28 | 16,000 | 16,000 | >16,000 | >16,000 |
| cc_s5 | sonnet | Reply risk ratio | 1.81 | 2.28 | 16,000 | 16,000 | >16,000 | >16,000 |
| cc_s5 | sonnet | Article size law | +0.0167 | +0.0010 | 500 | 1,000 | 2,000 | 4,000 |
| cc_s5 | sonnet | Depth law | +0.0124 | -0.0016 | 500 | 500 | 1,000 | 1,000 |
| cc_s5 | sonnet | Thread size law | +0.0205 | +0.0008 | 1,000 | 2,000 | 2,000 | 2,000 |
| cc_s5 | sonnet_ctx | Reply risk ratio | 1.64 | 2.28 | 8,000 | 16,000 | 16,000 | 16,000 |
| cc_s5 | sonnet_ctx | Article size law | +0.0180 | +0.0010 | 500 | 500 | 2,000 | 2,000 |
| cc_s5 | sonnet_ctx | Depth law | +0.0152 | -0.0016 | 500 | 500 | 1,000 | 1,000 |
| cc_s5 | sonnet_ctx | Thread size law | +0.0232 | +0.0008 | 1,000 | 1,000 | 2,000 | 2,000 |
| cc_threads | haiku_ctx | Reply risk ratio | 1.53 | 2.39 | 4,000 | 8,000 | 8,000 | 8,000 |
| cc_threads | haiku_ctx | Article size law | +0.0222 | +0.0032 | 500 | 500 | 2,000 | 2,000 |
| cc_threads | haiku_ctx | Depth law | +0.0203 | -0.0012 | 250 | 500 | 500 | 500 |
| cc_threads | haiku_ctx | Thread size law | +0.0365 | +0.0015 | 500 | 500 | 500 | 500 |
| cc_threads | imsypp@argmax | Reply risk ratio | 1.72 | 2.39 | 8,000 | 16,000 | 16,000 | 16,000 |
| cc_threads | imsypp@argmax | Article size law | +0.0158 | +0.0032 | 1,000 | 1,000 | 4,000 | 4,000 |
| cc_threads | imsypp@argmax | Depth law | -0.0081 | -0.0012 | 2,000 | 2,000 | 4,000 | 4,000 |
| cc_threads | imsypp@m | Depth law | -0.0050 | -0.0012 | 4,000 | 8,000 | 8,000 | >8,000 |
| cc_threads | mh_dist@m | Reply risk ratio | 2.85 | 2.39 | 16,000 | >16,000 | >16,000 | >16,000 |
| cc_threads | qwen7b_m | Reply risk ratio | 1.80 | 2.39 | 16,000 | >16,000 | >16,000 | >16,000 |
| cc_threads | qwen7b_m_ctx | Reply risk ratio | 1.13 | 2.39 | 2,000 | 4,000 | 4,000 | 4,000 |
| cc_threads | qwen7b_m_ctx | Article size law | +0.0207 | +0.0032 | 500 | 500 | 2,000 | 2,000 |
| cc_threads | qwen7b_m_ctx | Depth law | +0.0269 | -0.0012 | 250 | 250 | 250 | 250 |
| cc_threads | qwen7b_m_ctx | Thread size law | +0.0462 | +0.0015 | 500 | 500 | 250 | 500 |
| cc_threads | sn_rob@m | Reply risk ratio | 2.76 | 2.39 | 16,000 | >16,000 | >16,000 | >16,000 |
| cga | cn_off@m | Depth law | +0.0332 | +0.0374 | 1,000 | 1,000 | 2,000 | >2,000 |
| cga | fb_hate@m | Depth law | +0.0180 | +0.0374 | 250 | 500 | 500 | 500 |
| cga | fb_hate@m | Thread size law | -0.0160 | -0.0251 | 2,000 | >2,000 | >2,000 | >2,000 |
| cga | haiku | Depth law | +0.0306 | +0.0374 | 500 | 1,000 | 2,000 | >2,000 |
| cga | haiku | Thread size law | -0.0162 | -0.0251 | 2,000 | >2,000 | >2,000 | >2,000 |
| cga | imsypp@m | Depth law | +0.0300 | +0.0374 | 500 | 500 | 1,000 | >2,000 |
| cga | mh_dist@m | Depth law | +0.0305 | +0.0374 | 1,000 | 1,000 | 2,000 | >2,000 |
| cga | mh_dist@m | Thread size law | -0.0310 | -0.0251 | 2,000 | >2,000 | >2,000 | >2,000 |
| cga | qwen7b_m | Depth law | +0.0237 | +0.0374 | 250 | 500 | 500 | 1,000 |
| cga | qwen7b_m | Thread size law | -0.0146 | -0.0251 | 2,000 | >2,000 | >2,000 | >2,000 |
| cga | sn_rob@m | Depth law | +0.0327 | +0.0374 | 1,000 | 1,000 | 1,000 | >2,000 |
| qian_gab | cn_off@m | Reply risk ratio | 0.89 | 0.69 | 2,000 | 2,000 | 2,000 | 2,000 |
| qian_gab | cn_off@m | Depth law | -0.0025 | -0.0195 | 500 | 500 | 1,000 | 2,000 |
| qian_gab | cn_off@m | Thread size law | -0.0625 | -0.1198 | 500 | 500 | 1,000 | 2,000 |
| qian_gab | detoxify_original@0.6 | Reply risk ratio | 0.88 | 0.69 | 2,000 | 2,000 | 2,000 | 2,000 |
| qian_gab | detoxify_original@0.6 | Depth law | -0.0040 | -0.0195 | 500 | 500 | 1,000 | 2,000 |
| qian_gab | detoxify_original@0.6 | Thread size law | -0.0706 | -0.1198 | 500 | 1,000 | 1,000 | 2,000 |
| qian_gab | fb_hate@m | Reply risk ratio | 0.76 | 0.69 | 8,000 | 8,000 | 8,000 | 16,000 |
| qian_gab | fb_hate@m | Depth law | -0.0068 | -0.0195 | 500 | 1,000 | 2,000 | 4,000 |
| qian_gab | fb_hate@m | Thread size law | -0.0773 | -0.1198 | 500 | 1,000 | 1,000 | 4,000 |
| qian_gab | gemma_m | Reply risk ratio | 0.83 | 0.69 | 2,000 | 4,000 | 4,000 | 4,000 |
| qian_gab | gemma_m | Depth law | -0.0132 | -0.0195 | 2,000 | 4,000 | 4,000 | >8,000 |
| qian_gab | gemma_m | Thread size law | -0.0917 | -0.1198 | 1,000 | 2,000 | 2,000 | 8,000 |
| qian_gab | haiku | Reply risk ratio | 0.92 | 0.69 | 1,000 | 1,000 | 2,000 | 2,000 |
| qian_gab | haiku | Depth law | -0.0101 | -0.0195 | 1,000 | 2,000 | 4,000 | 8,000 |
| qian_gab | haiku | Thread size law | -0.0678 | -0.1198 | 500 | 500 | 1,000 | 2,000 |
| qian_gab | imsypp@m | Reply risk ratio | 0.95 | 0.69 | 1,000 | 1,000 | 2,000 | 1,000 |
| qian_gab | imsypp@m | Depth law | -0.0005 | -0.0195 | 500 | 500 | 1,000 | 2,000 |
| qian_gab | imsypp@m | Thread size law | -0.0401 | -0.1198 | 250 | 500 | 500 | 1,000 |
| qian_gab | mh_dist@m | Reply risk ratio | 1.08 | 0.69 | 500 | 500 | 500 | 1,000 |
| qian_gab | mh_dist@m | Depth law | -0.0009 | -0.0195 | 500 | 500 | 1,000 | 2,000 |
| qian_gab | mh_dist@m | Thread size law | -0.0313 | -0.1198 | 250 | 500 | 500 | 1,000 |
| qian_gab | sn_rob@m | Reply risk ratio | 0.90 | 0.69 | 2,000 | 2,000 | 2,000 | 2,000 |
| qian_gab | sn_rob@m | Depth law | -0.0036 | -0.0195 | 500 | 500 | 1,000 | 2,000 |
| qian_gab | sn_rob@m | Thread size law | -0.0687 | -0.1198 | 500 | 1,000 | 1,000 | 2,000 |
| qian_reddit | cn_off@m | Reply risk ratio | 1.51 | 1.14 | 4,000 | 4,000 | 4,000 | 4,000 |
| qian_reddit | cn_off@m | Depth law | -0.0119 | -0.0252 | 1,000 | 1,000 | 1,000 | 2,000 |
| qian_reddit | cn_off@m | Thread size law | -0.0517 | -0.1030 | 500 | 1,000 | 2,000 | 2,000 |
| qian_reddit | detoxify_original@0.6 | Reply risk ratio | 1.30 | 1.14 | 16,000 | 16,000 | 16,000 | 16,000 |
| qian_reddit | detoxify_original@0.6 | Depth law | -0.0158 | -0.0252 | 1,000 | 2,000 | 2,000 | 4,000 |
| qian_reddit | detoxify_original@0.6 | Thread size law | -0.0638 | -0.1030 | 1,000 | 1,000 | 2,000 | 4,000 |
| qian_reddit | fb_hate@m | Depth law | -0.0190 | -0.0252 | 2,000 | 4,000 | 4,000 | 8,000 |
| qian_reddit | fb_hate@m | Thread size law | -0.0689 | -0.1030 | 1,000 | 2,000 | 2,000 | 4,000 |
| qian_reddit | gemma_m | Reply risk ratio | 1.26 | 1.14 | 16,000 | 16,000 | 16,000 | 16,000 |
| qian_reddit | gemma_m | Depth law | -0.0223 | -0.0252 | 8,000 | 8,000 | >8,000 | >8,000 |
| qian_reddit | gemma_m | Thread size law | -0.0805 | -0.1030 | 2,000 | 4,000 | 4,000 | 8,000 |
| qian_reddit | haiku | Reply risk ratio | 1.49 | 1.14 | 4,000 | 4,000 | 4,000 | 8,000 |
| qian_reddit | haiku | Depth law | -0.0119 | -0.0252 | 1,000 | 1,000 | 1,000 | 2,000 |
| qian_reddit | haiku | Thread size law | -0.0467 | -0.1030 | 500 | 1,000 | 1,000 | 2,000 |
| qian_reddit | imsypp@m | Reply risk ratio | 1.26 | 1.14 | 16,000 | 16,000 | 16,000 | 16,000 |
| qian_reddit | imsypp@m | Depth law | -0.0125 | -0.0252 | 1,000 | 1,000 | 1,000 | 2,000 |
| qian_reddit | imsypp@m | Thread size law | -0.0566 | -0.1030 | 500 | 1,000 | 2,000 | 2,000 |
| qian_reddit | mh_dist@m | Reply risk ratio | 1.50 | 1.14 | 4,000 | 4,000 | 8,000 | 8,000 |
| qian_reddit | mh_dist@m | Depth law | -0.0067 | -0.0252 | 500 | 500 | 1,000 | 1,000 |
| qian_reddit | mh_dist@m | Thread size law | -0.0230 | -0.1030 | 250 | 500 | 1,000 | 1,000 |
| qian_reddit | sn_rob@m | Reply risk ratio | 1.53 | 1.14 | 4,000 | 4,000 | 4,000 | 4,000 |
| qian_reddit | sn_rob@m | Depth law | -0.0137 | -0.0252 | 1,000 | 1,000 | 2,000 | 2,000 |
| qian_reddit | sn_rob@m | Thread size law | -0.0503 | -0.1030 | 500 | 1,000 | 1,000 | 2,000 |
