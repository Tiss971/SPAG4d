# benchmarks/ — layout

One subfolder per benchmark campaign, named `<topic>_<date>`. Read
[`BENCHMARK_RULES.md`](BENCHMARK_RULES.md) before adding to any of them — it is the only file
that stays at this level, because it applies to all campaigns.

| folder | what it holds |
|---|---|
| [`baseline_2026-07-27/`](baseline_2026-07-27/) | Shipped-production reference over the 10-clip set: `baseline_2026-07-27.json` + its writeup. A magnitude reference only — never diff a new run against it (rule 1). |
| [`b1_vram_2026-07-31/`](b1_vram_2026-07-31/) | B1 VRAM/time levers: bf16 × `SPAG_SAM3_MAXSIZE` × `SPAG_SAM3_SCALE`, plus the per-clip colored-depth videos. **Its "target res" columns are mislabeled** — see `BENCHMARK_RULES.md` §10, the maxsize factor was applied to the WAFT-shrunk shape. |
| [`confdecay_2026-08-17/`](confdecay_2026-08-17/) | Confidence-decay reactivation + flow-warped propagation state (T1 §4.1/§4.2 fixes). Three matched arms per clip: `legacy`, `decay`, `noprop`. |
| *(not in this folder — see note below)* `freeze_bg_live_color` validation, 2026-08-21 | `freeze_bg`/`bglock`/`freeze_bg_live_color` 3-arm comparison on `circulation_site_1_edit_coupe` (shadows) + `boutique1_HQ` (screens). Output at repo-root `validate_freeze_bg_live_color_out{,_x2}/`, run via `validate_freeze_bg_live_color.py` — not yet relocated under `benchmarks/` per the convention below. See `BENCHMARK_RULES.md` §17 and `bglock_open_questions.md` §8.6. |

Conventions:

- **Each campaign folder is self-contained**: result JSONs, the `RESULTS.md` writeup, and the run
  logs the numbers were parsed from.
- **Matched arms are sibling subfolders** inside the campaign, one per config, run back-to-back on
  the same GPU (rule 1). The arm name is the config, not the date.
- Per-frame `depth_maps/` dumps and `*.npy` are gitignored — they are regenerable and run to
  hundreds of MB per arm. Keep the JSON/MD/logs, delete the dumps once the metrics are extracted.
