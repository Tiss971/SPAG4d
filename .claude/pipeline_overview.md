# Pipeline: Vidéo 360° fixe → Nuage de points 3D Gaussien dynamique

> Compacté 2026-08-25. Mécanismes détaillés (bglock, freeze_bg_live_color, etc.) sont
> maintenant couverts plus en profondeur dans `docs/WORK_LOG_DYNAMIC_360_RECONSTRUCTION.md` ;
> ce doc reste la référence pour l'architecture fichier-par-fichier et le diagramme d'étapes.

## Vue d'ensemble

Deux chemins dans `run_video()` :
- **Chemin principal** (`active_generator` = `da360`/`dap`/`pager`) : depth par frame +
  stabilisation temporelle (bg-lock ou affine) + conversion SPAG.
- **Chemin UniSHARP 360** (`active_generator="unisharp360"`) : process externe par frame
  (reconstruction 3DGS complète, pas de depth map exposée), incompatible avec
  bg-lock/affine/SAM3/WAFT.

```
[Vidéo 360°]
    ↓
[1] Extraction frames (skip sampling) → extract_video_frames
    ↓
    ├─ Si unisharp360 → branche séparée, sinon:
    ↓
[2] Dense optical flow (WAFT) → masque de mouvement + (si bglock) flow bidirectionnel seam-padded
    ↓
[3] SAM3 video tracking → masques de mouvement affinés M_i (objet-aware)
    ↓
[4] Médiane temporelle masquée → Master Background + activity_mask
    ↓
[5] DA360/PaGeR depth sur Background → D_ref (référence rigide, calculée UNE FOIS)
    ↓
[6] Pour chaque frame i:
    ├─ DA360/PaGeR depth → D_i (brut)
    ├─ Alignement affine: D_i ← s·D_i + t (pixels statiques, vs D_ref)
    ├─ depth_correction:
    │   ├─ "bglock" (défaut): propagate_depth_via_flow → composite_bg_locked
    │   │   (fond = D_ref exact, objet = depth propagée dans masque dilaté+feathered)
    │   └─ "affine" (legacy): FGDepthStabilizer2 (médiane glissante fg)
    ├─ Scene defaults fixes (une fois sur D_ref)
    ├─ (option) freeze_bg / freeze_bg_live_color
    ├─ (option) depth_npy_dir: dump depth/mask/flow .npy
    ├─ depth_to_gaussians(image_i, D_i_final)
    ├─ Post-filtres: outlier/grazing-angle/sparse-region pruning
    └─ save_ply_gsplat → gaussians/frame_i.ply
    ↓
[∞] Séquence de N .ply
```

## Détail des étapes

**[1] Extraction (`extract_video_frames`, video.py:40-93)** — OpenCV, skip sampling, retourne
`(frames, meta)`. Pas de downscaling ici (survient plus tard pour WAFT si résolution >1024px).

**[2] WAFT flow (video.py:473-524)** — flow monodirectionnel t→t+1 sur frames downscalées
(≤1024px) → seuil magnitude → masques de mouvement. Si `depth_correction=="bglock"`: deuxième
passe flow **bidirectionnel** seam-padded (`compute_bidirectional_flow`), upscalé natif —
c'est ce flow qui alimente `propagate_depth_via_flow` à l'étape [6]. Échec WAFT → fallback
propre sur `segment_with_sam` sans flow.

**[3] SAM3 tracking** — DIoU + prompts SAM3 si flow ok, fallback détection générique sinon.
Résultat fusionné en `fused_mask` (`alignement_mask`: `"sam"`, `"sam_and_activity"`,
`"nothing"`, `"all"`). `depth_correction=="bglock"` **exige** `alignement_mask in ("sam",
"sam_and_activity")` (ValueError sinon) — sous `SPAG_LOCK_ACTIVITY=1` (défaut), le compositing
lui-même clé sur `sam_mask` seul, pas `fused_mask` (voir CLAUDE.md).

**[4] Fond de référence (video.py:96-173)** — `compute_temporal_median2`, médiane temporelle
NaN-masquée (ignore pixels mobiles) → `master_background` + `activity_mask`.

**[5] Profondeur de référence** — `D_ref` estimée une seule fois sur le master background.
`temporal_consistency=True` (disponible, pas la valeur par défaut) désactive le rescale par
médiane indépendant par frame ; depuis bglock, ce n'est plus le mécanisme anti-drift
principal — bglock neutralise le drift en figeant le fond directement.

**[6.1] Alignement affine (video.py:760-867)** — `align_depth_frame` : `s,t` sur pixels
statiques (`fused_mask==0`) via `lstsq`/`ransac`/`median`, appliqué à toute la frame. Toujours
la première passe, que `depth_correction` soit `affine` ou `bglock`.

**[6.2] `bglock` vs `affine`** — sélectionné par `depth_correction` (défaut `"bglock"`).
`affine` (legacy) : `FGDepthStabilizer2` (médiane glissante fg), fond jamais explicitement
figé. `bglock` (video.py:842-870) : `propagate_depth_via_flow` (warp t-1→t, blend confiance
décroissante, gère couture/pôles/désocclusion) puis `composite_bg_locked` (fond = `D_ref`
exact, depth propagée seulement dans le masque dynamique dilaté+feathered). Résultat mesuré
(MattSwift 150f) : bg_temporal_std 0.097→0.0001m, fg_temporal_std 0.354→0.237m (voir
`HOW_TO_HAVE_TEMPORAL_STABILITY.md` pour le tableau complet A/B/C).

**[6.3] Scene defaults fixes** — `depth_min/max/sky_thr` calculés une fois sur `D_ref`. **Piège
corrigé (Dispo_RDV, 2026-08-20)** : `depth_min` dérivé de `D_ref` (fond) ne doit **pas** servir de
plancher de validité pour le foreground — un fond fixe n'a par construction jamais de sujet proche
de la caméra, donc réutiliser tel quel ce `depth_min` supprimait silencieusement toute personne plus
proche que ~2.65m. Fix au point d'appel foreground uniquement : `fg_depth_min = min(depth_min, 0.15)
if depth_min is not None else 0.15` ; `gaussians_bg` (construit depuis `D_ref` lui-même) inchangé.
Détails : `docs/DISPO_RDV_CLOSE_RANGE_PRUNING_FIX.md`.

**[6.4] `freeze_bg` (video.py:878-911)** — Gaussiennes de fond calculées une fois (frame 0),
concaténées au foreground régénéré chaque frame (masque SAM). ~1.5× plus rapide, ~70% moins de
splats. Bug corrigé : `aligned_depth_np2` copiée avant NaN-masking pour éviter que le masque
NaN fuite dans `depth_prev_final` par aliasing. `freeze_bg_live_color` (voir CLAUDE.md) re-
échantillonne la couleur du fond figé depuis l'image live chaque frame là où `sam_mask==0`.

**[6.5] `depth_npy_dir` (video.py:894-895)** — dump `depth_{idx}.npy`/`mask_{idx}.npy`/
`flow_{idx}.npy` (post-stabilisation) + `depth_ref.npy` (une fois). Pour reprojeter sans trous
d'occlusion : composer `depth_{idx}` (là où `mask_{idx}==0`) + repli sur `depth_ref` là où
`mask_{idx}==1` — même principe que `freeze_bg` appliqué aux dumps bruts.

**[6.6] Conversion & post-filtres** — `depth_to_gaussians` + pruning, stateless/recalculé par
frame (source résiduelle de popping, non résolue par bg-lock qui stabilise la profondeur, pas
l'identité des Gaussiennes).

**[7] Export PLY** — `save_ply_gsplat`, pas de correspondance inter-frames, popping résiduel
accepté par design.

## Chemin alternatif : UniSHARP 360 (video.py:422-471)

Branche séparée : chaque frame → JPEG temp → `convert_unisharp360` (subprocess externe,
`third_party/UniSHARP`) → reconstruction 3DGS complète par frame en ERP natif directement (pas
de cubemap, pas de couture). SAM3/WAFT/depth-compositing/bglock ne s'appliquent pas. Options :
`unisharp_repo`, `unisharp_python`, `unisharp_checkpoint`, `unisharp_scale_align`,
`unisharp_format_mode`, `unisharp_max_gaussians`. Mergé, fonctionnel, pas encore positionné
comme remplaçant/complément de `bglock`.

## Métriques de stabilité temporelle

`ConversionResult.depth_metrics` (voir `A1_DEPTH_METRICS_IMPLEMENTATION.md`) : `bg_depth_cv`
(primaire), `bg_delta_max/mean`, `bg_spike_count`, `fg_delta_std`/`fg_depth_cv` (doit être plus
élevé que le fond). Calculé sur la depth alignée, avant bg-lock.

## Reste ouvert

1. Cohérence FB du flow non branchée sur le gel de mémoire SAM (existe en interne dans
   `propagate_depth_via_flow` pour la désocclusion, mais rien ne gèle explicitement SAM3).
2. Pas d'identité/tracking Gaussien inter-frames (popping résiduel, "dernier recours").
3. Utilité de `depth_smoothing` une fois `bglock` en place : **confirmée** (voir
   `bglock_sol1_median_w5` dans `HOW_TO_HAVE_TEMPORAL_STABILITY.md`) — pas redondante,
   se combine et améliore fg_depth_cv au-delà de chaque mécanisme seul.
4. UniSHARP vs bglock — pas de comparaison directe tranchée.
5. Filtres de pruning stateless par frame — source connue non traitée de popping résiduel.
6. Trail de depth-bleed scene01 ("waiter") sous bglock — **rouvert** (2026-08-21) : bleed de
   profondeur au bord de silhouette pendant le warp-flow, confirmé comme mécanisme réel mais le
   fix testé (`SPAG_BLEED_REJECT`, seuil global) supprimait aussi les points de contact fond
   légitimes (pieds au sol) → reverté. Prochain fix doit distinguer bleed vs. contact réel via
   structure spatiale locale, pas un seuil global. Détails : `docs/bglock_open_questions.md` §8.7,
   `docs/SPAG_BLEED_REJECT_SCENE01_VALIDATION.md`.

## Fichiers clés

| Fichier | Fonctionnalité | Statut |
|---|---|---|
| `spag4d/video.py` | Pipeline principal (`run_video`) | bglock défaut, unisharp360 branché, freeze_bg fix aliasing, depth_npy_dir, depth_metrics |
| `spag4d/flow_depth_propagation.py` | `propagate_depth_via_flow`, `composite_bg_locked` | cœur du gain de stabilité |
| `spag4d/detect_opticalflow.py` | WAFT wrapper (mono + bidirectionnel) | complet |
| `spag4d/da360_model.py` / `pager_model.py` | Depth estimation | `temporal_consistency` disponible, pas le levier principal depuis bglock |
| `spag4d/scene_analysis.py` | Scene defaults fixes | actif |
| `spag4d/scene_filter.py` | Pruning filters | stateless par design, popping résiduel connu |
| `spag4d/spag_converter.py` | SPAG Gaussians | pas d'identité inter-frames |
| `spag4d/unisharp360.py` | Chemin UniSHARP 360 | branche séparée |
| `spag4d/ply_writer.py` | Export PLY | pas de correspondance inter-frames (par design) |
| `spag4d/core.py` | `ConversionResult` | `depth_metrics` field |
