# benchmarks/ — layout

One subfolder per benchmark campaign, named `<topic>_<date>`. Read
[`BENCHMARK_RULES.md`](BENCHMARK_RULES.md) before adding to any of them — it is the only file
that stays at this level, because it applies to all campaigns.

| folder | what it holds |
|---|---|
| [`baseline_2026-07-27/`](baseline_2026-07-27/) | Shipped-production reference over the 10-clip set: `baseline_2026-07-27.json` + its writeup. A magnitude reference only — never diff a new run against it (rule 1). |
| [`b1_vram_2026-07-31/`](b1_vram_2026-07-31/) | B1 VRAM/time levers: bf16 × `SPAG_SAM3_MAXSIZE` × `SPAG_SAM3_SCALE`, plus the per-clip colored-depth videos. **Its "target res" columns are mislabeled** — see `benchmarks/T1_T2_P0_AUDIT.md`, the maxsize factor was applied to the WAFT-shrunk shape. |
| [`confdecay_2026-08-17/`](confdecay_2026-08-17/) | Confidence-decay reactivation + flow-warped propagation state (T1 §4.1/§4.2 fixes). Three matched arms per clip: `legacy`, `decay`, `noprop`. |

Conventions:

- **Each campaign folder is self-contained**: result JSONs, the `RESULTS.md` writeup, and the run
  logs the numbers were parsed from.
- **Matched arms are sibling subfolders** inside the campaign, one per config, run back-to-back on
  the same GPU (rule 1). The arm name is the config, not the date.
- Per-frame `depth_maps/` dumps and `*.npy` are gitignored — they are regenerable and run to
  hundreds of MB per arm. Keep the JSON/MD/logs, delete the dumps once the metrics are extracted.
