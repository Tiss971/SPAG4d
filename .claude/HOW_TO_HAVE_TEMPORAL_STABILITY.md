# Résumé : stabilisation temporelle fond fixe / premier plan mobile

Vue d'ensemble de toutes les méthodes testées pour éliminer le scintillement
("flicker") de la profondeur monoculaire sur une vidéo 360° ERP à caméra fixe,
en distinguant fond statique (doit être parfaitement stable) et objets mobiles
(doivent bouger correctement). Fusionne l'ancien
`BENCHMARK_SOLUTIONS_RESULTS.md`, `V2_depth_stability_benchmark.md` et
`V3_OTHERS_SOLUTIONS_TO_TEST.md` en un seul document (2026-07-21).

## Chronologie des méthodes

| Date | Méthode | Statut | Résultat |
|---|---|---|---|
| 2026-07-09 | **B1 — mode `temporal_consistency`** : désactive le rescale par médiane par frame dans DA360/PaGeR ; calcule `scale_factor_to_5m` une seule fois sur la frame de référence, réutilisé pour toutes les frames ([B1_temporal_consistency_fix.md](B1_temporal_consistency_fix.md)) | ✅ mergé (`temporal_consistency=True`) | Corrige la source primaire de dérive d'échelle |
| 2026-07-09 | **C.1 — scene defaults fixes** : `depth_min/max/sky_thr` calculés une seule fois sur `depth_ref` au lieu d'être recalculés par frame | ✅ mergé | Supprime une source secondaire de scintillement (seuils de filtrage Gaussien stables) |
| 2026-07-09 | **Propagation de profondeur par flow** (prototype, mergé dans `spag4d/flow_depth_propagation.py`) : warp la depth stabilisée de t-1 vers t via le flow WAFT dense, au lieu de ré-estimer + recaler affine chaque frame. Blend pondéré par une confiance qui décroît (`decay=0.85`) pour éviter la dérive sur mouvement radial. Gère les pièges spécifiques ERP fixe : couture horizontale (padding circulaire), singularité aux pôles (bande de méfiance 8%), disocclusion = fond déjà connu (`D_ref` réutilisé directement) | ✅ validé sur vidéo réelle, puis mergé (`spag4d/flow_depth_propagation.py`) | Gain ~11-29% sur l'écart-type temporel ; élimine les décrochages catastrophiques ponctuels de l'affine sur `MattSwift_03.mp4` (sujet mobile réel) |
| 2026-07-09 → 07-15 | **Compositing bg-locked** (`composite_bg_locked`, [détails ci-dessous](#benchmark-a--affine-vs-b--flow-prop-vs-c--bg-locked)) : ne plus "aligner" le fond, le figer exactement à `D_ref` (variance nulle par construction) ; la depth propagée par flow n'est utilisée que dans le masque dynamique (dilaté + feathered) | ✅ mergé (`bg_lock_dilate_px`, `bg_lock_feather_px` dans `run_video`) | **Gagnant** : std temporel fond 0.097m → 0.0001m (~1000×), flicker p2p 0.032→0.000m ; foreground aussi moins de flicker que l'affine seul (0.237 vs 0.354m) |
| — | **Lissage temporel de profondeur** (médiane/gaussienne glissante sur fenêtre de frames) | ✅ implémenté (`depth_smoothing`, `depth_smoothing_window`, `depth_smoothing_method`) ; benchmarké combiné à bg-lock, [voir ci-dessous](#benchmark-solutionspy--configs-de-stabilité) | **Gagnant combiné avec bg-lock** : `bglock_sol1_median_w5` = config de production |
| — | Cohérence forward-backward du flow (détection occlusion) | Partiellement couvert par la vérification fb interne à la propagation par flow ; pas branché séparément sur le gel de la mémoire SAM | Non fait en tant que tel |
| — | Alignement affine robuste (RANSAC / médiane Siegel vs lstsq) | Disponible (`method=`), utilité non prouvée depuis que bg-lock existe | Non testé systématiquement — voir [idées restantes](#idées-explorées-mais-non-retenues--non-testées) |
| — | Référence de profondeur multi-frame (médiane de N frames de référence) | Testé (`reference_frames_for_median` / `sol4_ref7`) | **No-op** confirmé sur bg et fg, seul ou combiné à bglock/sol1 — abandonné |
| — | Tracking de points / identité des Gaussiennes entre frames | ⭐⭐⭐ effort et risque élevés, explicitement reporté ("dernier recours") | Non tenté |
| parallèle | **Comparaison de générateurs** pager vs unisharp360, via nouvelles métriques (`bg_depth_cv`, spike count, [DEPTH_METRICS_IMPLEMENTATION.md](A1_DEPTH_METRICS_IMPLEMENTATION.md)) | ✅ chemin unisharp360 mergé en parallèle du bg-lock (commit `23eeccf`) | UniSHARP montrait un CV de fond plus faible sur les premiers tests |
| 2026-07-21 | **GPU-resident depth chain + composite-on-GPU + Tier-3 single-pass WAFT** ([B1_VRAM_TIME_REDUCTION_PLAN.md](B1_VRAM_TIME_REDUCTION_PLAN.md)) | ✅ mergé (`d23cb01`), composite-on-GPU par défaut | Wall 148.8→88.4s (−40%), VRAM 37,092→20,353MB (−45%), métriques quasi-identiques |
| 2026-07-31 | **Validation full-frame / full-stride** (`scripts/run_scene03_fullres_bglock.py`, `skip_step=1, stride=1`) sur `scene_03.mp4` (café intérieur, personnes, dataset jamais utilisé auparavant) — pas de sous-échantillonnage benchmark, config proche production | ✅ run complet | 617 frames, 2020s, VRAM peak 31,146MB — confirme que bglock passe à l'échelle en pleine résolution sans changement d'ordre de grandeur du coût VRAM/temps vs les runs sous-échantillonnés |
| 2026-08-17 | **Réactivation du decay de confiance + état propagé warpé** (audit P0 T1 §4.1/§4.2, [benchmarks/T1_T2_P0_AUDIT.md](../benchmarks/T1_T2_P0_AUDIT.md), résultats [benchmarks/confdecay_2026-08-17/](../benchmarks/confdecay_2026-08-17/RESULTS.md)) : le `decay=0.85` annoncé depuis 2026-07-09 **n'avait jamais compté** — un reset à 1.0 à chaque frame de confiance rendait `running_confidence` binaire {0, 0.85}, donc un poids de blend fixe, pas une décroissance. L'état porté est maintenant un **âge par pixel** (frames depuis le dernier ré-ancrage), **warpé le long du flow** comme la depth qu'il décrit (§4.2), et la confiance vaut `max(conf_floor, decay**age)` | ✅ mergé, défaut ON (`SPAG_CONF_DECAY=0.85`, `SPAG_CONF_FLOOR=0.5`) ; `SPAG_CONF_LEGACY=1` restitue l'ancien chemin à l'identique | MattSwift : la fidélité du mouvement fg passe de **0.692 → 0.923** vs l'estimation monoculaire brute — l'ancienne "stabilité" fg (`fg_depth_cv` 0.1307 vs 0.1582) était donc à ~31% du mouvement réel supprimé. `bg_depth_cv` inchangé, VRAM identique, temps dans le bruit. **Toute mesure antérieure au 2026-08-17 a été prise sous l'ancien chemin** |

## Où on en est (`spag4d/video.py`, config de production)

`run_video()` expose tous les leviers gagnants ensemble : `temporal_consistency`,
scene defaults fixes, `depth_smoothing*`, `depth_correction="bglock"`
(`bg_lock_dilate_px`/`bg_lock_feather_px`, appelle `composite_bg_locked`), et un
chemin générateur alternatif `unisharp360`. Le hot-path GPU (align + smoother +
propagate + composite) est utilisé automatiquement pour la config gagnante.

**Idée centrale qui a gagné** : ne pas ré-estimer + corriger le fond à chaque
frame — le figer sur la profondeur de référence, et ne faire confiance à la
depth fraîche/propagée que dans le masque d'objet mobile (dilaté, feathered).

**Config de production retenue** : `bglock_sol1_median_w5`
(`depth_correction="bglock"` + `depth_smoothing=True, depth_smoothing_window=5,
depth_smoothing_method="median"`) — meilleure stabilité fond ET premier plan
simultanément (voir benchmark ci-dessous).

---

## Benchmark A. affine vs B. flow-prop vs C. bg-locked

Vidéo `9_MattSwift.mp4`, 150 frames, native 2048×1024, PLY stride 4 (masque
dynamique ≈ 3% de l'image, une personne bougeant dans une pièce).

| method | bg_temporal_std (m) | bg_flicker p2p (m) | fg_temporal_std (m) | spatial_rough |
|--------|--------:|--------:|--------:|--------:|
| A. affine (current) | 0.0974 | 0.0322 | 0.3536 | 0.02535 |
| B. flow-prop        | 0.0847 | 0.0059 | 0.2797 | 0.02248 |
| **C. bg-locked (new)** | **0.0001** | **0.0000** | **0.2367** | 0.03095 |

- **Le fond est maintenant fixe** : std temporel 0.097 m → 0.0001 m (~1000×), p2p
  flicker 0.032 m → 0.000 m.
- **Les objets bougent toujours** : le std du premier plan reste non nul (0.24 m)
  — et *plus bas* que l'affine (0.35 m), car la depth objet est propagée par flow.
- La rugosité spatiale légèrement plus élevée de bg-locked est un artefact de
  bord : la région statique de la métrique inclut le bord flou du masque feathered
  où le `D_ref` lisse rencontre la depth objet plus bruitée. Élargir `feather_px`
  compense ce compromis.

Repro : `conda run -n spag4d python benchmark_depth_stability.py --video
/raid/mb273924/SPAG4d/TestImage/9_MattSwift.mp4 --output
./benchmark_depth_stability_mattswift --max-frames 150 --ply-stride 4
--ply-export-count 5` — sort `stats.json`, `comparison.png`, et séquences PLY
stride-4 pour `affine` et `bglock` (comparaison A/B visualiseur du fond immobile
vs mouvement objet). Flow tourne à ≤1024px puis upscalé ; depth / compositing /
PLY tournent à la résolution native.

Caveats : le masque dynamique ici est un proxy magnitude de flow WAFT, pas SAM3
— bg-locking n'est bon que si le masque l'est (un pixel d'objet manqué est gelé
au fond). Dans le pipeline réel, le masque SAM3 alimente `composite_bg_locked`
(déjà dilaté+feathered pour absorber le lag/halo du masque). Le mouvement radial
(objet approchant droit sur la caméra) a un flow quasi-nul ; la confiance
décroissante de flowprop re-ancre sur la depth monoculaire, donc bg-locked
hérite de cette correction dans le masque.

---

## Benchmark solutions.py — configs de stabilité

`benchmark_solutions.py` — 8 configs pipeline, chacune sur les 4 mêmes vidéos
(`accident_electrique_02`, `atelier_1`, `boutique1_HQ`,
`circulation_site_1_edit_coupe`), generator = `da360`.

Base config partagée : `skip_step=8, stride=8, temporal_consistency=False,
freeze_bg=True, outlier_pruning=0.3, grazing_angle=85.0, sparse_pruning=0.1`.

Exécuté en parallèle sur 3 GPU (1, 2, 4) via
`.sandbox/run_parallel_benchmark.sh`. ~50 min pour les 32 runs vidéo/config.
Métriques via `calculate_temporal_stability()` à partir de `depth_metrics` de
`run_video()` — pas besoin de lire les PLY.

**Bug corrigé en route** : `benchmark_solutions.py` importait un module
inexistant `batch_compare_generators` (renommé en `benchmark_generators.py`
sans mise à jour de l'import) — corrigé à `benchmark_solutions.py:26`.

**Piège GPU pinning** : `CUDA_VISIBLE_DEVICES=4` sélectionnait initialement le
GPU Display physique du DGX (index 3) au lieu du 4e A100 voulu, car
l'énumération CUDA par défaut ("fastest first") ne correspond pas aux index
PCI-bus de `nvidia-smi`. Corrigé en exportant `CUDA_DEVICE_ORDER=PCI_BUS_ID`
avant `CUDA_VISIBLE_DEVICES`.

### Round 1 — résultats (moyenne sur 4 vidéos)

| config | bg_depth_cv ↓ | bg_spikes/frame ↓ | fg_depth_cv ↓ | fg_delta_mean ↓ | time/video |
|---|---|---|---|---|---|
| baseline | 0.0132 | 0.0346 | 0.0510 | 0.0637 | 198s |
| bglock | 0.0132 | 0.0346 | 0.0577 | 0.0637 | 237s |
| **sol1_median_w5** | **0.0116** | **0.0205** | **0.0467** | **0.0340** | 203s |
| sol1_gaussian_w5 | 0.0126 | 0.0378 | 0.0475 | 0.0501 | 208s |
| sol4_ref7 | 0.0132 | 0.0346 | 0.0515 | 0.0526 | 194s |
| sol6_b5_j0.2 | 0.0132 | 0.0346 | 0.0586 | 0.0637 | 189s |
| sol6_b11_j0.5 | 0.0132 | 0.0346 | 0.0538 | 0.0637 | 199s |
| sol6_b15_j1.0 | 0.0132 | 0.0346 | 0.0548 | 0.0637 | 195s |

Données complètes : `benchmark_solutions/SUMMARY.json`, détail par vidéo dans
`benchmark_solutions/<config>/results.json`.

Takeaways round 1 :
- **Gagnant : `sol1_median_w5`** (lissage temporel de profondeur, filtre médian,
  fenêtre=5). Meilleur sur toutes les métriques de stabilité simultanément —
  bg_depth_cv −12%, taux de spike quasi divisé par 2, fg_delta_mean quasi divisé
  par 2 — pour seulement ~5s/vidéo de coût additionnel.
- `sol1_gaussian_w5` (même idée, fenêtre gaussienne) aide mais moins que la
  variante médiane, et a même un taux de spike *supérieur* au baseline — le
  lissage médian gère mieux les outliers de saut de profondeur qu'un noyau
  gaussien.
- `bglock` seul donne des chiffres de stabilité fond identiques au `baseline`
  (attendu — il change le compositing, pas les valeurs de depth qui nourrissent
  ces métriques) mais coûte ~20% de temps en plus et dégrade fg_depth_cv. Pas
  justifié sur cette seule preuve — nécessite une inspection visuelle/PLY.
- `sol4_ref7` (référence sur 7 frames) est un no-op sur les métriques bg ici —
  identique au baseline — avec un petit changement de fg_delta_mean au niveau du
  bruit.
- Les runs `sol6_*` (réglage du foreground-stabilizer) ne bougent pas les
  métriques bg du tout (ils ne touchent que le compositing FG) et ne montrent
  pas de tendance monotone claire sur fg_depth_cv selon la taille du buffer —
  inconcluant à partir de ces seules métriques ; nécessiterait une revue
  visuelle.

### Round 2 (2026-07-16→17) — test de combinabilité

Suivi pour tester si `bglock` et `sol1_median_w5`/`sol4_ref7` se combinent.
Changements vs round 1 : `atelier_1` retirée (3 vidéos restantes :
`accident_electrique_02`, `boutique1_HQ`, `circulation_site_1_edit_coupe`),
`skip_step` réduit de 8 à 4 (plus de frames/vidéo, runtime ~50% plus long),
`sol1_gaussian_w5` et tous les `sol6_*` retirés (pas de signal clair round 1).
Nouvelles configs combo : `bglock_sol1_median_w5`, `sol1_median_sol4` (lissage
médian + ref 7 frames), `bglock_sol4_ref7`.

| config | bg_depth_cv ↓ | bg_spikes/frame ↓ | fg_depth_cv ↓ | fg_delta_mean ↓ | time/video |
|---|---|---|---|---|---|
| baseline | 0.0078 | 0.0179 | 0.0317 | 0.0284 | 289s |
| bglock | 0.0078 | 0.0179 | 0.0125 | 0.0284 | 362s |
| sol1_median_w5 | 0.0074 | 0.0036 | 0.0281 | 0.0132 | 292s |
| sol4_ref7 | 0.0078 | 0.0179 | 0.0314 | 0.0288 | 247s |
| sol1_median_sol4 | 0.0074 | 0.0036 | 0.0279 | 0.0134 | 290s |
| **bglock_sol1_median_w5** | **0.0074** | **0.0036** | **0.0083** | **0.0132** | 376s |
| bglock_sol4_ref7 | 0.0078 | 0.0179 | 0.0123 | 0.0288 | 341s |

(Valeurs absolues non comparables au round 1 — set de vidéos et skip_step
différents — mais les deltas relatifs dans ce round sont valides.)

**Oui, ça se combine, et c'est le nouveau gagnant.** `bglock_sol1_median_w5`
hérite du meilleur des deux parents plutôt que de moyenner ou d'interférer :
- bg_depth_cv / taux de spike / fg_delta_mean correspondent à `sol1_median_w5`
  seul (0.0074 / 0.0036 / 0.0132) — bglock ne perturbe pas le gain de stabilité
  fond du lissage médian.
- fg_depth_cv descend à 0.0083, *meilleur* que chaque parent seul (bglock:
  0.0125, sol1_median_w5: 0.0281) — les deux agissent sur des parties
  différentes du pipeline (compositing fond vs lissage des valeurs de depth) et
  leurs bénéfices fg se cumulent.
- Coût : 376s/vidéo, la config la plus chère testée (+30% vs baseline, +29% vs
  sol1_median_w5 seul, +4% vs bglock seul) — le surcoût de compositing de bglock
  domine le coût ajouté, le lissage lui-même est bon marché.

`sol4_ref7` (référence multi-frame) confirme le round 1 : no-op sur toutes les
métriques, seul ou combiné à bglock ou sol1 — `bglock_sol4_ref7` est
statistiquement identique à `bglock` seul, `sol1_median_sol4` identique à
`sol1_median_w5` seul. Sûr d'abandonner `sol4`/`reference_frames_for_median`.

**Décision retenue** : `bglock_sol1_median_w5` en production — meilleure
stabilité bg ET fg simultanément. `sol1_median_w5` seul reste le fallback si le
+30% de temps compte pour un batch complet (fg_depth_cv 0.028 vs 0.008, mais
~80s/vidéo moins cher).

---

## Assets de présentation + comparaisons additionnelles (2026-07-31)

Pour un support PPT expliquant le mécanisme bglock (D_ref calculé une seule
fois avant la boucle par-frame, puis réutilisé deux fois : comme **cible de
régression** de l'alignement affine par frame sur les pixels statiques, et
comme **valeur figée exacte** du compositing final) :

- `.sandbox/bglock_explainer/bglock_pipeline_diagram.png` — diagramme
  matplotlib avec une zone "COMPUTED ONCE" (D_ref) séparée visuellement d'une
  zone "EVERY FRAME t" (alignement + propagation + compositing), flèche
  explicite D_ref → Affine align labellisée "regression target".
- `.sandbox/bglock_explainer/bglock_bg_flicker_evidence.png` — graphe
  flicker fond réel affine vs bglock sur `accident_electrique_02`.
- `.sandbox/bglock_explainer/bglock_bullets.md` — notes orateur, avec une
  section détail sur la mécanique de `propagate_depth_via_flow` (warp
  backward via `flow_fwd`, confiance = FB-consistency × pole-margin ×
  decay temporel 0.85, disocclusion = masque SAM *de la frame courante*, pas
  de la précédente) et `composite_bg_locked` (alpha = masque dilaté+feathered,
  cas limite `D_ref` NaN).

Nouveaux clips de comparaison (scripts existants réutilisés tels quels, voir
[Note] dans `CLAUDE.md`) : `fg_depth_flicker.mp4` (affine vs bglock) sur
`atelier_1`, `da360_vs_pager_indepscale_fg_flicker.mp4` (da360 vs pager) sur
`circulation_site_1_edit_coupe` — ce dernier via un nouveau script pérennisé
`scripts/render_indepscale_fg_flicker.py` (percentiles 2-98 **indépendants**
par côté, nécessaire car da360 = profondeur métrique et pager = profondeur
scale-invariant ; un percentile partagé y écraserait artificiellement le côté
à plus petite échelle).

## Idées explorées mais non retenues / non testées

Issues de l'exploration initiale (avant que bg-lock + sol1 median ne soient
identifiés comme gagnants). Statut mis à jour ; celles déjà couvertes
ci-dessus ne sont pas répétées.

- **Alignement affine robuste (RANSAC / médiane Siegel)** — disponible via
  `alignement_method=`, jamais benchmarké systématiquement. Utilité non prouvée
  depuis que bg-lock existe (le fond n'est plus aligné du tout, il est figé) ;
  ne resterait pertinent que pour l'alignement du masque d'objet mobile lui-même.
  Effort faible, risque faible — à faire seulement si un problème d'alignement
  du premier plan est identifié.
- **Diagnostics qualité de masque** (couverture flow/SAM, IoU, stabilité
  frame-à-frame) — jamais implémenté en tant qu'outil séparé ; utile seulement
  si on soupçonne les masques d'être le goulot d'étranglement. La cohérence
  forward-backward interne à `propagate_depth_via_flow` couvre partiellement ce
  besoin en production.
- **Réglage fin de `FGDepthStabilizer2`** (`buffer_size`, `jump_threshold`) —
  testé en round 1 (`sol6_*`), inconcluant (pas de tendance monotone claire) ;
  nécessiterait une revue visuelle pour trancher. Abandonné faute de signal.
- **Tracking de points / identité des Gaussiennes entre frames** — effort et
  risque élevés (⭐⭐⭐), explicitement mis de côté comme "dernier recours".
  Ne vaut le coup que si le popping du PLY entre frames devient un problème
  bloquant pour un usage avec interpolation ; le popping actuel est accepté,
  pas résolu.

## Reste ouvert

- Pas de cohérence forward-backward du flow branchée sur le gel de la mémoire
  SAM (gestion des occlusions encore basique).
- Pas d'identité/tracking par Gaussienne entre frames — le popping du PLY
  d'une frame à l'autre est accepté, pas résolu.
- Alignement affine robuste (RANSAC/médiane) non benchmarké — probablement
  sans intérêt tant que bg-lock gère le fond, à revisiter seulement si le
  masque d'objet mobile pose problème.
