# Pipeline: Vidéo 360° fixe → Nuage de points 3D Gaussien dynamique

> Mise à jour 2026-07-16, après merge `d7678d0` (bg-locked depth stability +
> unisharp360). Remplace la version du 2026-07-09 : le mode `depth_correction`
> par défaut est désormais **`bglock`** (fond figé sur la référence), pas
> l'alignement affine seul. Voir [TEMPORAL_STABILITY_SUMMARY.md](TEMPORAL_STABILITY_SUMMARY.md)
> pour la chronologie complète des méthodes testées.

## Vue d'ensemble

La pipeline transforme une **vidéo 360° monoculaire** (caméra fixe) en une **séquence de nuages de points 3D Gaussian Splat** (un PLY par frame). Deux familles de chemins existent dans `run_video()` :

- **Chemin principal** (`active_generator` = `da360`/`dap`/`pager`, tous partagent la même machinerie) : depth par frame + stabilisation temporelle (bg-lock ou affine) + conversion SPAG.
- **Chemin UniSHARP 360** (`active_generator="unisharp360"`) : branche séparée, tourne un process externe par frame (une reconstruction 3DGS complète par frame, pas de depth map exposée), **incompatible avec bg-lock/affine/SAM3/WAFT** — voir section dédiée plus bas.

```
[Vidéo 360°]
    ↓
[1] Extraction frames (skip sampling) → extract_video_frames
    ↓
    ├─ Si active_generator == "unisharp360" → branche séparée (voir plus bas), sinon:
    ↓
[2] Dense optical flow (WAFT) → masque de mouvement + (si bglock) flow bidirectionnel seam-padded
    ↓ (optionnel, si flow réussit)
[3] SAM3 video tracking → masques de mouvement affinés M_i (objet-aware)
    ↓
[4] Médiane temporelle masquée sur les frames → Master Background + activity_mask
    ↓
[5] DA360/PaGeR depth estimation sur Background → D_ref (référence rigide, calculée UNE FOIS)
    ↓
[6] Pour chaque frame i:
    ├─ DA360/PaGeR depth estimation → D_i (brut)
    ├─ Alignement affine: D_i ← s·D_i + t (sur pixels statiques, vs D_ref)
    ├─ Stabilisation de la profondeur (choix depth_correction):
    │   ├─ "bglock" (défaut): propagate_depth_via_flow (warp objet depuis frame précédente)
    │   │   → composite_bg_locked (fond = D_ref exact, objet = depth propagée, dans un masque
    │   │      dynamique dilaté+feathered) → variance de fond ≈ 0 par construction
    │   └─ "affine" (legacy): FGDepthStabilizer2 (médiane glissante sur la depth foreground)
    ├─ Scene defaults fixes (depth_min/max/sky_thr calculés une fois sur D_ref, pas par frame)
    ├─ (option) freeze_bg: réutilise les Gaussiennes de fond calculées à la frame 0, ne régénère
    │    que le foreground (masque SAM) → moins de Gaussiennes, plus rapide
    ├─ (option) depth_npy_dir: dump la depth métrique brute (post-stabilisation) en .npy float32
    ├─ Conversion Gaussienne: depth_to_gaussians(image_i, D_i_final)
    ├─ Post-filtres: outlier/grazing-angle/sparse-region pruning
    └─ Export: save_ply_gsplat → gaussians/frame_i.ply
    ↓
[∞] Séquence de N fichiers .ply (N = nombre de frames capturées)
```

---

## Détail des étapes

### [1] Extraction frames (`extract_video_frames`) — video.py:40–93

```python
def extract_video_frames(video_path: str, output_folder, skip_step: int = 10, export: bool = True)
    → tuple[np.ndarray, dict]
```

- Lit la vidéo au complet avec OpenCV, **skip sampling** : ignore `skip_step-1` frames de suite pour chaque frame conservée.
- Retourne `(frames [N,H,W,3] uint8 BGR, meta dict)` où `meta` contient `W`, `H`, `fps`, `total`.
- Pas de downscaling ici ; le downscaling pour WAFT survient plus tard dans `run_video` si résolution > 1024px.

**État**: Fonctionnel, stable.

---

### [2] Dense optical flow (WAFT) — video.py:473–524, detect_opticalflow.py

```python
class WAFTWrapper:
    def infer_pair(frame1, frame2) → flow [H,W,2]
    def run(frames, meta, out_dir, threshold) → dict
```

- **Checkpoint**: `/raid/mb273924/_DATASETS/uptale/tar-c-t.pth`, config `WAFT/config/a1/tar-c-t.json`.
- `model.run(...)` : flow monodirectionnel (t→t+1) sur les frames downscalées (≤1024px), seuille magnitude → masques de mouvement `flow_masks`, exporte `flows.mp4` (diagnostics).
- **Si `depth_correction == "bglock"`** : une **deuxième passe** calcule le flow **bidirectionnel** (`compute_bidirectional_flow`, forward + backward, avec padding de couture `flow_seam_pad`) pour chaque paire consécutive, upscalé à résolution native si besoin. C'est ce flow bidirectionnel (pas le flow monodirectionnel de la première passe) qui alimente `propagate_depth_via_flow` à l'étape [6].
- Les deux bugs historiques (`masks[-1]` non initialisé, `meta["new_H"/"new_W"]` manquants) sont **corrigés** : `meta["new_W"]`/`meta["new_H"]` sont désormais toujours définis avant l'appel WAFT, que le downscale ait lieu ou non.
- Échec WAFT (exception) : `flow_masks` reste `None`, la pipeline retombe proprement sur `segment_with_sam` (fallback SAM sans flow) sans planter.

**État**: Fonctionnel. Pas de forward-backward consistency branchée sur le gel de mémoire SAM (reste un point ouvert, voir plus bas) — mais la vérification fb est bien utilisée en interne par `propagate_depth_via_flow` (bglock) pour détecter les zones de désocclusion.

---

### [3] SAM3 video tracking — video.py (segment_with_flows / segment_with_sam)

Comportement inchangé vs version précédente : tracking DIoU + prompts SAM3 si le flow a réussi, fallback détection cible générique ("human") sinon. Résultat : masques de mouvement affinés objet-aware, fusionnés avec l'activity mask en `fused_mask` (contrôlé par `alignement_mask`: `"sam"`, `"sam_and_activity"`, `"nothing"`, `"all"`).

**Contrainte nouvelle** : `depth_correction == "bglock"` **exige** `alignement_mask in ("sam", "sam_and_activity")` — un masque dynamique réel est obligatoire pour savoir où faire confiance à la depth propagée plutôt qu'au fond figé (`run_video` lève une `ValueError` sinon).

---

### [4] Fond de référence (médiane temporelle masquée) — video.py:96–173

`compute_temporal_median2(frames_array, masks_dict, quantile)` : médiane temporelle **NaN-masquée** (ignore les pixels mobiles selon `masks_dict`), retourne `master_background` + `activity_mask`. Comportement inchangé depuis 07-09, toujours l'implémentation utilisée (pas `compute_temporal_median`, inutilisée).

---

### [5] Profondeur de référence — da360_model.py / pager_model.py

`D_ref` = depth estimée **une seule fois** sur le master background. Mode `temporal_consistency=True` (disponible mais **pas la valeur par défaut** dans les configs de production actuelles) désactive le rescale-par-médiane indépendant par frame ; sinon chaque frame recalcule sa propre médiane→5m (source de drift si non compensé plus loin par bg-lock/affine).

**Depuis 07-09, ce n'est plus la mitigation principale** : c'est `bglock` (étape 6) qui neutralise ce drift en figeant le fond directement, indépendamment de ce que fait `temporal_consistency`.

---

### [6.1] Alignement affine de profondeur — video.py:760–867 (inchangé)

`align_depth_frame` : estimation `s, t` sur les pixels statiques (`fused_mask == 0`) via `lstsq`/`ransac`/`median`, application `depth = s·depth + t` sur toute la frame. Toujours la première passe de stabilisation, **que `depth_correction` soit `affine` ou `bglock`** — bglock affine encore la sortie du modèle brut avant de la composer avec `D_ref`.

---

### [6.2] Stabilisation de la profondeur : `bglock` vs `affine` — **nouveau, cœur du changement**

Deux stratégies, sélectionnées par `depth_correction` (`video.py:361`, défaut `"bglock"`) :

#### **`affine` (legacy)** — comportement de la version précédente

`FGDepthStabilizer2` (médiane glissante sur `buffer_size` valeurs de profondeur médiane foreground) corrige les sauts sur la depth déjà alignée affine. Le **fond** n'est jamais explicitement figé : il reste soumis au bruit de ré-estimation + alignement à chaque frame.

#### **`bglock` (défaut actuel)** — video.py:842–870

Principe central : **ne pas ré-estimer le fond, le figer sur `D_ref`**, et ne faire confiance à la depth fraîche que dans le masque d'objet mobile.

1. `propagate_depth_via_flow(depth_prev_final, aligned_depth_np, depth_ref_np, flow_fwd, flow_bwd, ...)` : warp la depth stabilisée de la frame précédente vers la frame courante via le flow bidirectionnel, blend avec la depth monoculaire fraîche par une confiance qui décroît (`decay=0.85`) — gère la couture ERP horizontale (padding circulaire), la singularité aux pôles (marge de méfiance `pole_margin_frac`) et la désocclusion (retombe sur `D_ref`, déjà connu). Frame 0 : pas de propagation, `depth_object = aligned_depth_np`.
2. `composite_bg_locked(depth_object, depth_ref_np, fused_mask, dilate_px=bg_lock_dilate_px, feather_px=bg_lock_feather_px)` : pixels statiques = `D_ref` **exactement** (variance temporelle nulle par construction) ; la depth propagée n'est utilisée que dans le masque dynamique, dilaté et feathered pour absorber le lag/halo du masque.

**Résultat mesuré** (`9_MattSwift.mp4`, 150 frames, cf. [depth_stability_benchmark.md](depth_stability_benchmark.md)) :

| method | bg_temporal_std (m) | bg_flicker p2p (m) | fg_temporal_std (m) |
|---|---:|---:|---:|
| affine (legacy) | 0.0974 | 0.0322 | 0.3536 |
| flow-prop seul | 0.0847 | 0.0059 | 0.2797 |
| **bglock (défaut)** | **0.0001** | **0.0000** | **0.2367** |

Fond quasi-parfaitement stable (~1000× moins de variance), et le foreground est *aussi* moins bruyant que l'affine seul (la depth de l'objet vient du flow-propagated, pas d'une ré-estimation brute).

**État**: Fonctionnel, mergé, défaut de production. `FGDepthStabilizer`/`FGDepthStabilizer2` (EMA/médiane glissante) restent le chemin utilisé uniquement quand `depth_correction="affine"`.

---

### [6.3] Scene defaults fixes — inchangé depuis 07-09

`depth_min`/`depth_max`/`sky_threshold` calculés une fois sur `D_ref` (`compute_scene_defaults`), réutilisés pour toutes les frames — élimine la variation per-frame des seuils de filtrage Gaussien. Toujours actif.

---

### [6.4] `freeze_bg` (optionnel) — video.py:878–911

Si activé : les Gaussiennes de fond sont calculées **une seule fois** (frame 0) et concaténées à chaque frame aux Gaussiennes du foreground (régénérées depuis le masque SAM). Réduit fortement le nombre de Gaussiennes et le temps de calcul (~1.5× plus rapide, ~70% moins de splats dans les benchmarks C.1).

**Bug corrigé récemment** (commenté dans le code, video.py:882-888) : `aligned_depth_np2` est copiée avant le NaN-masking pour éviter que le masque NaN ne fuite dans `depth_prev_final` par aliasing — sinon, avec `bglock+freeze_bg`, la région masquée (humain) s'érodait en NaN au fil des frames et disparaissait progressivement.

---

### [6.5] Export depth brute (`depth_npy_dir`, optionnel) — video.py:894–895

Si fourni, dump `depth_{idx}.npy` (float32, la depth métrique réellement utilisée pour générer les Gaussiennes, **après** alignement/bglock/freeze_bg — pas un JPEG normalisé irréversible). Sert notamment de source pour la supervision de profondeur côté FreeTimeGsVanilla (voir `/raid/mb273924/FreeTimeGsVanilla/docs/FLOW_DEPTH_MASK_TRAINING_PLAN.md`) et pour les métriques de stabilité temporelle ci-dessous.

---

### [6.6] Conversion Gaussienne & post-filtres — inchangés

`depth_to_gaussians` (SPAG spherical projection) + `prune_outliers`/`prune_grazing_angle`/`prune_sparse_regions`, toujours stateless/recalculés par frame (source résiduelle de popping, cf. section causes plus bas — non résolue par bg-lock, qui stabilise la *profondeur*, pas l'*identité* des Gaussiennes).

---

### [7] Export PLY — inchangé

`save_ply_gsplat`. Toujours pas de correspondance/ID inter-frames — popping résiduel du PLY accepté par design (voir "Reste ouvert" ci-dessous).

---

## Chemin alternatif : UniSHARP 360 (`active_generator="unisharp360"`) — video.py:422–471

Branche complètement séparée, ajoutée en parallèle du travail bg-lock (commit `23eeccf`) :

- Pas de depth map interne exposée : chaque frame est écrite en JPEG temporaire puis passée à `convert_unisharp360` (subprocess externe, `third_party/UniSHARP`), qui produit **une reconstruction 3DGS complète par frame** directement en ERP natif (pas de projection cubemap/face, donc pas de couture).
- Toute la machinerie SAM3/WAFT/depth-compositing/bglock/affine **ne s'applique pas** à ce chemin — architecturalement incompatible (pas de depth map à composer).
- Options dédiées : `unisharp_repo`, `unisharp_python`, `unisharp_checkpoint`, `unisharp_scale_align`, `unisharp_format_mode`, `unisharp_max_gaussians` (sous-échantillonnage optionnel post-génération).
- Comparé à `pager`/`da360` via les nouvelles métriques de stabilité de fond (`bg_depth_cv`, voir [DEPTH_METRICS_IMPLEMENTATION.md](DEPTH_METRICS_IMPLEMENTATION.md)) — premiers tests favorables à UniSHARP sur le CV de fond, mais pas encore de comparaison directe contre `bglock` (bglock rend le fond quasi-parfait par construction, ce que UniSHARP n'a pas structurellement).

**État**: Mergé, fonctionnel, chemin de génération alternatif — pas encore positionné comme remplaçant ou complément de `bglock`.

---

## Métriques de stabilité temporelle (nouveau) — voir [DEPTH_METRICS_IMPLEMENTATION.md](DEPTH_METRICS_IMPLEMENTATION.md)

`run_video()` retourne désormais `ConversionResult.depth_metrics`, calculé sur la depth alignée (avant bg-lock) :
- `bg_depth_cv` : coefficient de variation (std/mean) de la profondeur médiane du fond — métrique primaire, plus bas = plus stable.
- `bg_delta_max`/`bg_delta_mean`/`bg_spike_count` : amplitude et fréquence des sauts frame-à-frame sur le fond.
- `fg_delta_std`/`fg_depth_cv` : équivalents foreground (doit être *plus élevé* que le fond — sinon les objets ne sont probablement pas bien suivis).

Utilisé par `batch_compare_generators.py` pour classer les générateurs (pager vs unisharp vs da360) par stabilité de fond plutôt que par nombre de Gaussiennes.

---

## Reste ouvert (post bg-lock)

1. **Cohérence forward-backward du flow non branchée sur le gel de mémoire SAM** : la vérification fb existe en interne dans `propagate_depth_via_flow` (pour la désocclusion), mais rien ne "gèle" explicitement le tracking SAM3 pendant une occlusion détectée.
2. **Pas d'identité/tracking persistant des Gaussiennes entre frames** : le nombre et l'ordre des Gaussiennes varient librement par frame (post-pruning) → popping résiduel du PLY, indépendant de la qualité de la depth. Reporté comme "dernier recours" (effort/risque élevés).
3. **`depth_smoothing`** (lissage temporel médiane/gaussienne glissante, option indépendante) : son utilité une fois `bglock` en place n'est **pas confirmée** — possiblement redondant ou même conflictuel. À trancher par un benchmark bglock seul vs bglock+smoothing.
4. **UniSHARP vs bglock** : pas de comparaison directe tranchée; UniSHARP a un meilleur CV de fond que les *autres* générateurs testés sans bg-lock, mais bg-lock rend le fond quasi-parfait structurellement pour da360/pager/dap — comparaison encore à faire.
5. **Filtres de pruning (outlier/grazing-angle/sparse)** : toujours stateless par frame, source connue mais non traitée de popping résiduel indépendant de la depth.

---

## Fichiers clés (statut, 2026-07-16)

| Fichier | Fonctionnalité | Statut |
|---|---|---|
| `spag4d/video.py` | Pipeline principal (`run_video`) | ✓ bglock défaut, unisharp360 branché, freeze_bg fix aliasing, depth_npy_dir, depth_metrics |
| `spag4d/flow_depth_propagation.py` | `propagate_depth_via_flow`, `composite_bg_locked` | ✓ nouveau, cœur du gain de stabilité |
| `spag4d/detect_opticalflow.py` | WAFT wrapper (mono + bidirectionnel) | ✓ complet, bugs 07-09 corrigés |
| `spag4d/da360_model.py` / `pager_model.py` | Depth estimation | ✓ `temporal_consistency` disponible (pas le levier principal depuis bglock) |
| `spag4d/scene_analysis.py` | Scene defaults fixes | ✓ actif |
| `spag4d/scene_filter.py` | Pruning filters | ✓ stateless par design, popping résiduel connu |
| `spag4d/spag_converter.py` | SPAG Gaussians | ✓ pas d'identité inter-frames (inchangé) |
| `spag4d/unisharp360.py` | Chemin UniSHARP 360 | ✓ nouveau, branche séparée |
| `spag4d/ply_writer.py` | Export PLY | ✓ pas de correspondance inter-frames (par design) |
| `spag4d/core.py` | `ConversionResult` | ✓ `depth_metrics` field ajouté |
