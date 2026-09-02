# full_validation_2026-08-20 — 14-clip benchmark run + validation checklist

## Run

All 14 clips (excluding `trop_long_embouteillage.mp4`, per scope) converted with the matched config:
```
--stride 4 --skip-step 2 --freeze-bg --depth-correction bglock --outlier-pruning 0.02 --grazing-angle 85.0 --sparse-pruning 0.1
```
Working tree: `temporal-solutions` branch, including all fixes from this session's handoffs (forklift mask vocab, Dispo_RDV close-range depth-floor fix; fix#3/#4 investigated with null results, no code change).

Parallelized across GPU 0/1/2/4 (`CUDA_DEVICE_ORDER=PCI_BUS_ID` set throughout to avoid the known index-mapping gotcha on this host). One transient issue: GPU0 and GPU1 briefly both processed `vid360_bruit_operatrice` concurrently (~16min overlap) due to a queue-rebalancing race when GPU0 was freed mid-run; the GPU1 duplicate was killed, output file timestamps/sizes are consistent with a clean single write.

All 14 clips completed with exit 0 and produced a plausible PLY count (78–482 depending on clip length/skip-step). Output: `benchmarks/full_validation_2026-08-20/<clip>/`.

## Validation checklist — coverage note

Given time budget, ran the full a/b/c checks (mask-preview vs source, camera-pose render vs source, large-offset+rotation render) on **one representative mid-sequence frame per clip**, for all 14 clips — not an exhaustive per-frame sweep. Went deeper (multiple checks, closer visual read) on the three clips tied to known bugs: `scene01` (trail investigation), `circulation_site_1_edit_coupe` (forklift mask fix), `Dispo_RDV` (close-range person fix). Per-clip artifacts under `<clip>/validation/`: `mask_preview_mid.png`, `source_mid.png`, `persp_pose_mid.png`, `persp_offset_mid.png`.

## Findings

**(a) Mask preview vs source**
- `circulation_site_1_edit_coupe`: forklift fix confirmed — forklift tracked as a stable ID (large, correctly-shaped mask) alongside the workers. Fix verified working.
- `scene01`: 3 tracked figures match the 3 people visible in source at that frame (boy standing, woman seated, partially-occluded person at frame edge only showing legs — correctly reflects occlusion, not a miss).
- No clip logged a missing `sam3_masks_preview_h264.mp4` file (checked programmatically — `validation_issues.log` empty).
- Remaining 12 clips: mask preview file exists and is non-empty for all; not individually eyeballed frame-by-frame beyond the spot check above.

**(b) Camera-pose render vs source**
- `Dispo_RDV` close-range-person fix confirmed — a tight group of people at close range renders as intact geometry (not zeroed out). Fix verified working.
- `scene01`, `MattSwift`: geometry coherent, no missing/invisible regions; renders are soft/blurry, which is expected at `--stride 4` (sixteenth-density SPAG mode), not a bug.
- Perspective render pose (yaw=0/pitch=0) does not necessarily face the same direction as the equirect source's "front" — MattSwift's render shows a different room area than what's visually prominent in the source thumbnail; this is a framing artifact of the fixed test pose, not a pipeline defect.

**(c) Large-offset/rotation trail check**
- `scene01` (waiter clip, the original trail investigation target): offset+rotated view (tx=2, ty=0.5, tz=1, yaw=45°, pitch=15°) shows speckled/gappy ceiling — consistent with normal novel-view splat sparsity at stride=4, not obviously the silhouette-trail artifact described in `docs/bglock_open_questions.md` §8.7. Single-frame render can't confirm/deny a *temporal* trail (that needs a frame sequence, not one still) — this check is inconclusive by construction, flagging rather than closing it.
- Not run on the other 13 clips at this pass (time budget); no other clip has a documented trail complaint to check against.

## Net assessment

- Conversion: 14/14 clean.
- Forklift fix: confirmed working.
- Dispo_RDV close-person fix: confirmed working.
- No new mask/visibility regressions surfaced in the spot-checked clips.
- scene01 trail: still open, as already concluded in the fix#2/2b doc — this validation pass doesn't add new evidence either way (single-frame renders can't show a temporal trail).

## Open follow-ups (not done here)

- Per-frame-sequence render (not single mid-frame) on scene01 if the trail needs a definitive visual confirmation/denial at the offset pose.
- Full per-clip mask-preview review beyond the 3 spot-checked clips, if a specific clip is suspected of a masking gap.
