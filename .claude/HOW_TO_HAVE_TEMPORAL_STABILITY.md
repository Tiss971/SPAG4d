# Résumé : stabilisation temporelle fond fixe / premier plan mobile

Vue d'ensemble de toutes les méthodes testées pour éliminer le scintillement
("flicker") de la profondeur monoculaire sur une vidéo 360° ERP à caméra fixe,
en distinguant fond statique (doit être parfaitement stable) et objets mobiles
(doivent bouger correctement). Compacté 2026-08-25 — résumé exécutif équivalent
en anglais dans `docs/WORK_LOG_DYNAMIC_360_RECONSTRUCTION.md`.

## Chronologie des méthodes

| Date | Méthode | Statut | Résultat |
|---|---|---|---|
| 2026-07-09 | **`temporal_consistency`** : `scale_factor_to_5m` calculé une fois sur la frame de référence au lieu d'un rescale médiane par frame | ✅ mergé | Corrige la source primaire de dérive d'échelle |
| 2026-07-09 | **Scene defaults fixes** : `depth_min/max/sky_thr` sur `depth_ref` une seule fois | ✅ mergé | Seuils de filtrage Gaussien stables |
| 2026-07-09 | **Propagation de profondeur par flow** (`flow_depth_propagation.py`) : warp t-1→t via flow WAFT, blend pondéré par confiance décroissante ; gère couture ERP, singularité pôles, désocclusion (`D_ref`) | ✅ mergé | Gain ~11-29% sur écart-type temporel |
| 2026-07-09→15 | **Compositing bg-locked** (`composite_bg_locked`) : fond figé exactement à `D_ref` (variance nulle), depth propagée seulement dans le masque dynamique dilaté+feathered | ✅ mergé, **gagnant** | std fond 0.097→0.0001m (~1000×) ; fg aussi moins de flicker que l'affine (0.237 vs 0.354m) |
| — | **Lissage temporel médiane/gaussienne** (`depth_smoothing*`) | ✅ implémenté, combiné à bglock | `bglock_sol1_median_w5` = config de production |
| — | Référence multi-frame (médiane N frames, `sol4_ref7`) | Testé | **No-op** confirmé, abandonné |
| — | Alignement affine robuste (RANSAC/Siegel) | Disponible (`method=`) | Utilité non prouvée depuis bg-lock, jamais benchmarké systématiquement |
| — | Tracking de points / identité Gaussienne inter-frames | Reporté ("dernier recours") | Non tenté |
| 2026-07-21 | **GPU-resident depth chain + composite-on-GPU + single-pass WAFT** | ✅ mergé | Wall −40%, VRAM −45%, métriques quasi-identiques (détail : `B1_VRAM_TIME_REDUCTION_PLAN.md`) |
| 2026-08-17 | **Réactivation du decay de confiance** (audit P0, `docs/bglock_open_questions.md` §4.1/4.2) : `decay=0.85` n'avait jamais compté (reset à 1.0 chaque frame → poids fixe {0,0.85}, pas décroissance). État désormais un **âge par pixel warpé le long du flow**, confiance = `max(floor, decay**age)` | ✅ mergé, défaut ON | MattSwift fg motion-fidelity 0.692→0.923 vs monoculaire brut ; l'ancienne "stabilité" (`fg_depth_cv` 0.131 vs 0.158) supprimait ~31% du mouvement réel. Mesures antérieures au 2026-08-17 = ancien chemin |
| 2026-08-20→21 | **Investigation trail scene01 "waiter"** sous bglock : smear de depth-bleed derrière le sujet mobile. #2 mask-scope gap (fixé, sans effet visible) → #3 retune dilate/feather (null) → #4 decay de confiance (non-contributeur) → #5 `SPAG_BLEED_REJECT` (seuil global de rejet du bleed silhouette) | 🔴 rouvert | #5 a mesurablement réduit le trail (diff ×10-100 vs #2-#4) mais son seuil global supprimait aussi les points de contact fond légitimes (pieds au sol) → reverté. Mécanisme confirmé (bleed au bord de silhouette pendant le warp-flow) mais pas de fix shippable. Détails : `docs/bglock_open_questions.md` §8.7, `docs/SPAG_BLEED_REJECT_SCENE01_VALIDATION.md` |

## Où on en est

`run_video()` expose tous les leviers ensemble : `temporal_consistency`, scene defaults
fixes, `depth_smoothing*`, `depth_correction="bglock"` (`composite_bg_locked`), chemin
générateur alternatif `unisharp360`. Le hot-path GPU (align+smoother+propagate+composite)
s'active automatiquement pour la config gagnante.

**Idée centrale** : ne pas ré-estimer + corriger le fond à chaque frame — le figer sur
`D_ref`, ne faire confiance à la depth fraîche/propagée que dans le masque d'objet mobile.

**Config de production** : `bglock_sol1_median_w5` (`depth_correction="bglock"` +
`depth_smoothing=True, depth_smoothing_window=5, depth_smoothing_method="median"`) —
meilleure stabilité fond ET premier plan simultanément.

---

## Benchmark A. affine vs B. flow-prop vs C. bg-locked

`9_MattSwift.mp4` (clip supprimé du repo depuis, résultats conservés ici), 150 frames, 2048×1024,
PLY stride 4 (masque dynamique ≈3%).

| method | bg_temporal_std (m) | bg_flicker p2p (m) | fg_temporal_std (m) | spatial_rough |
|--------|--------:|--------:|--------:|--------:|
| A. affine | 0.0974 | 0.0322 | 0.3536 | 0.02535 |
| B. flow-prop | 0.0847 | 0.0059 | 0.2797 | 0.02248 |
| **C. bg-locked** | **0.0001** | **0.0000** | **0.2367** | 0.03095 |

Caveat : masque dynamique ici = proxy magnitude flow WAFT, pas SAM3 (le pipeline réel
utilise SAM3, déjà dilaté+feathered). Rugosité spatiale légèrement plus haute pour
bg-locked = artefact de bord au feather, compensable en élargissant `feather_px`.

Repro : clip source supprimé (voir note ci-dessus) ; commande historique
`benchmark_depth_stability.py --video TestImage/9_MattSwift.mp4 --max-frames 150 --ply-stride 4`
n'est plus rejouable telle quelle.

---

## Benchmark solutions.py — configs de stabilité

`benchmark_solutions.py`, 8 configs × 4 vidéos (`accident_electrique_02`, `atelier_1`,
`boutique1_HQ`, `circulation_site_1_edit_coupe`), generator `da360`.

### Round 1 (moyenne 4 vidéos)

| config | bg_depth_cv ↓ | bg_spikes/frame ↓ | fg_depth_cv ↓ | fg_delta_mean ↓ | time/video |
|---|---|---|---|---|---|
| baseline | 0.0132 | 0.0346 | 0.0510 | 0.0637 | 198s |
| bglock | 0.0132 | 0.0346 | 0.0577 | 0.0637 | 237s |
| **sol1_median_w5** | **0.0116** | **0.0205** | **0.0467** | **0.0340** | 203s |
| sol1_gaussian_w5 | 0.0126 | 0.0378 | 0.0475 | 0.0501 | 208s |
| sol4_ref7 | 0.0132 | 0.0346 | 0.0515 | 0.0526 | 194s |

Gagnant round 1 : `sol1_median_w5` (fenêtre médiane 5) — meilleur sur toutes les métriques
simultanément (bg_depth_cv −12%, spike rate ~÷2, fg_delta_mean ~÷2), ~5s/vidéo de coût.
Médiane bat noyau gaussien sur les outliers. `bglock` seul = identique au baseline sur les
métriques bg (attendu, change le compositing pas les valeurs) mais +20% temps et dégrade
fg — nécessite inspection visuelle pour juger seul. `sol4_ref7` = no-op confirmé.

### Round 2 — combinabilité (3 vidéos, `skip_step=4`)

| config | bg_depth_cv ↓ | bg_spikes/frame ↓ | fg_depth_cv ↓ | fg_delta_mean ↓ | time/video |
|---|---|---|---|---|---|
| baseline | 0.0078 | 0.0179 | 0.0317 | 0.0284 | 289s |
| bglock | 0.0078 | 0.0179 | 0.0125 | 0.0284 | 362s |
| sol1_median_w5 | 0.0074 | 0.0036 | 0.0281 | 0.0132 | 292s |
| **bglock_sol1_median_w5** | **0.0074** | **0.0036** | **0.0083** | **0.0132** | 376s |

**Gagnant retenu : `bglock_sol1_median_w5`** — hérite du meilleur des deux parents plutôt
que de moyenner : bg_depth_cv/spike/fg_delta_mean = `sol1_median_w5` seul (bglock ne
perturbe pas ce gain), fg_depth_cv (0.0083) *meilleur* que chaque parent seul (bglock
0.0125, sol1 0.0281) — bénéfices cumulatifs car ils touchent des parties différentes du
pipeline. Coût : +30% vs baseline, la config la plus chère testée. `sol4_ref7` reconfirmé
no-op même combiné. **`sol1_median_w5` seul reste le fallback** si le +30% de temps compte
pour un batch complet (fg_depth_cv 0.028 vs 0.008, ~80s/vidéo moins cher).

## Idées explorées mais non retenues / non testées

- **Alignement affine robuste (RANSAC/Siegel)** — disponible, jamais benchmarké
  systématiquement ; utilité non prouvée depuis bg-lock (le fond n'est plus aligné, il est
  figé), resterait pertinent seulement pour l'alignement du masque objet mobile lui-même.
- **Diagnostics qualité de masque** (IoU, stabilité frame-à-frame) — jamais implémenté en
  outil séparé ; la cohérence FB interne à `propagate_depth_via_flow` couvre partiellement.
- **Réglage `FGDepthStabilizer2`** (`sol6_*`) — testé round 1, inconcluant, abandonné.
- **Tracking Gaussien inter-frames** — effort/risque élevés, dernier recours si le popping
  devient bloquant.

## Reste ouvert

- Pas de cohérence FB du flow branchée sur le gel de la mémoire SAM (occlusion basique).
- Pas d'identité/tracking par Gaussienne inter-frames — popping accepté, pas résolu.
- Alignement affine robuste non benchmarké — probablement sans intérêt tant que bg-lock
  gère le fond.
- Trail de depth-bleed scene01 (voir chronologie 2026-08-20→21) — mécanisme confirmé mais
  aucun fix shippable ; prochain essai doit distinguer bleed vs. contact fond réel via
  structure spatiale locale (ring contigu et non borné vs. petit footprint stable), pas un
  seuil global par frame.
