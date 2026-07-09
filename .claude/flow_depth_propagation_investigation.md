# Investigation: Flow-based depth propagation (fallback si l'alignement affine ne suffit pas)

Contexte: caméra 360° **fixe**, ERP. Alternative/complément à `align_depth_frame`
(`spag4d/video.py`) qui ré-estime la profondeur monoculaire à chaque frame puis
la recale statistiquement (s, t) sur les pixels statiques.

## Idée

Au lieu de ré-estimer + recaler chaque frame indépendamment, on **propage** la
depth déjà stabilisée de la frame t-1 vers t via le flow dense WAFT déjà
calculé dans le pipeline (warp), et on ne retombe sur l'estimation monoculaire
que là où le warp n'est pas fiable.

```
depth_final[t] = confiance · warp(depth_final[t-1], flow[t-1→t])
               + (1-confiance) · depth_affine[t]   (méthode actuelle, fallback)
```

où `confiance` vient de :
- **cohérence forward-backward** du flow (détecte occlusions/disocclusions),
- une **bande de méfiance aux pôles** (voir ci-dessous),
- une **décroissance temporelle** (evite la dérive sur mouvement radial, voir ci-dessous).

Pour les pixels de disocclusion (masque dynamique redevenu statique) : comme la
caméra est fixe, la zone révélée EST le fond statique déjà connu →
on réutilise directement `D_ref` plutôt qu'une estimation monoculaire fraîche.

## Pourquoi ce n'est pas juste "appliquer WAFT" — 4 pièges spécifiques à l'ERP fixe

1. **Couture horizontale (seam)** : WAFT est un modèle de flow entraîné sur de
   la vidéo perspective, sans notion que la colonne 0 et la colonne W-1 sont
   adjacentes sur la sphère. Un objet qui traverse la couture ressemble à une
   téléportation (flow horizontal énorme et faux). **Mitigation implémentée** :
   padding circulaire horizontal des deux frames avant inférence WAFT (défaut
   64px), crop après — la fenêtre de corrélation locale peut alors "voir à
   travers" la couture pour un déplacement inter-frame < `seam_pad`.
   Le warp lui-même échantillonne aussi `x mod W` (pas de clamp) pour rester
   cohérent.

2. **Singularité aux pôles** : près des rangées du haut/bas, une ligne entière
   de pixels ERP peut représenter un quasi-point sur la sphère. Un petit
   mouvement angulaire y produit un flow pixel énorme/instable, et WAFT n'a
   pas été entraîné pour cette distorsion. **Mitigation implémentée** :
   bande de non-confiance (`pole_margin_frac`, défaut 8%) en haut/bas — dans
   cette bande, on ne fait jamais confiance à la propagation, fallback
   systématique sur l'alignement affine actuel.

3. **Angle mort — mouvement radial pur** : le flow optique ne capture que le
   déplacement 2D apparent. Un objet qui s'approche/s'éloigne le long du rayon
   caméra (fréquent avec une caméra fixe : quelqu'un qui marche vers/depuis la
   caméra) a un flow quasi nul mais une vraie variation de profondeur. Un pur
   remplacement par la depth propagée figerait ce genre d'objet à une
   profondeur obsolète. **Mitigation implémentée** : jamais de remplacement
   pur — blend pondéré par une confiance qui **décroît à chaque frame propagée
   sans ré-ancrage** (`PropagationState.decay`, défaut 0.85), donc la depth
   revient progressivement vers l'estimation monoculaire (bruitée mais non
   biaisée) au lieu de dériver indéfiniment.

4. **Disocclusion = fond déjà connu (spécifique caméra fixe)** : contrairement
   à une caméra mobile où une zone révélée est vraiment nouvelle, ici toute
   zone qui redevient statique après passage d'un objet est nécessairement du
   fond déjà capturé dans `D_ref` (médiane temporelle). On l'utilise
   directement plutôt qu'une estimation monoculaire fraîche (qui, elle,
   redérive à chaque frame comme documenté dans `pipeline_overview.md`).

## Implémentation prototype

- `spag4d/flow_depth_propagation.py` — module autonome (pas de dépendance
  SAM3), pensé pour être testé isolément puis branché dans `video.py` à la
  place du corps de boucle qui appelle `align_depth_frame`.
  - `compute_bidirectional_flow` : flow t→t+1 ET t+1→t avec padding circulaire.
  - `warp_backward` : `grid_sample` avec échantillonnage circulaire en x,
    clampé en y (pas de wrap aux pôles).
  - `fb_consistency_error` : erreur de cohérence forward-backward, sert à
    détecter occlusions/zones non fiables.
  - `pole_trust_mask` : bande de méfiance aux pôles.
  - `propagate_depth_via_flow` : orchestration + état de confiance persistant
    (`PropagationState`) entre frames.

- `benchmark_flow_propagation.py` (racine, même style que
  `benchmark_realistic.py`) — script de comparaison empirique, autonome
  (masque foreground approximé par seuillage de la magnitude du flow WAFT,
  pas besoin de SAM3). Gère les sources 4K : `--work-max-size` (défaut 1024)
  downscale tout de suite après extraction — DA360 redimensionne de toute
  façon en interne à 518×1036, donc travailler à 4K n'ajoute aucun détail de
  profondeur réel, juste du temps de calcul WAFT/DA360. Export PLY optionnel
  (`--ply-export-count`, `--ply-stride`) pour quelques frames espacées
  régulièrement, avec un stride spatial élevé (défaut 8) pour garder des
  fichiers gérables à ces résolutions de travail. Calcule pour chaque frame :
  - la depth actuelle (`align_depth_frame`, méthode en prod),
  - la depth propagée par flow (ce prototype),
  et compare l'écart-type temporel sur les pixels foreground (proxy direct du
  scintillement, même métrique que `_temporal_depth_std.jpg` déjà utilisée
  dans le pipeline).

## Résultat empirique

| Vidéo | Résolution native | Contenu | Std temporel — affine (actuel) | Std temporel — flow prop. (prototype) | Gain |
|---|---|---|---|---|---|
| `temp_fast_track_h264.mp4` | 1024×512 | Entrepôt, ~statique | 0.497 m | 0.445 m | ~11% |
| `accident_electrique_fast5.mp4` | 3840×1920 (4K) | Entrepôt, ~statique | 0.598 m | 0.511 m | ~15% |
| `MattSwift_03.mp4` | 2048×1024 | **2 personnes assises, gestes de la main** | 0.299 m | 0.213 m | ~29% |

(toutes les 3 tournées avec `--work-max-size 1024`, downscale WAFT/DA360
appliqué avant tout calcul — cf. section précédente.)

Les deux premières vidéos n'ont pas de sujet clairement mobile : le masque
foreground (seuillage flow) ne capte qu'un petit cluster de bruit/flicker
(`_debug_fg_mask.jpg` dans chaque dossier de sortie), pas un objet qui
traverse la scène — ces deux chiffres mesurent donc surtout un lissage du
bruit de fond du monoculaire, pas un vrai cas d'usage.

**`MattSwift_03.mp4` est le test qui compte** : la scène montre deux personnes
assises qui discutent, avec des gestes de main détectés par le masque flow.
Sur `comparison.png` de ce run, la courbe orange (alignement affine actuel)
présente des décrochages catastrophiques et ponctuels — jusqu'à **0.05 m**
autour de la frame 119, plusieurs chutes sous 1 m ailleurs — typiques d'une
frame où l'estimation monoculaire brute dérape et où le recalage affine (basé
sur un petit nombre de pixels statiques) ne suffit pas à corriger. La courbe
verte (propagation par flow) ignore complètement ces décrochages et reste
dans une plage cohérente (~1.5–3.5 m) sur toute la séquence : c'est
exactement le mode de défaillance que la propagation par flow est censée
éliminer — un pixel dont l'estimation monoculaire ponctuelle est aberrante
hérite quand même d'un prior temporel stable tant que le flow le suit
correctement.

**Limite restante** : le masque foreground reste un proxy (seuillage flow),
pas les masques SAM3 réels. Aucune de ces 3 vidéos n'exerce non plus les cas
limites propres à l'ERP (objet qui traverse la couture, sujet qui s'approche
radialement de la caméra) — à vérifier avant intégration en production, mais
le mécanisme central (rejet des décrochages ponctuels du monoculaire) est
maintenant validé sur un sujet réellement mobile.

## Plan d'intégration dans `video.py` (si validé)

1. Dans la boucle principale (`video.py:566-654`), garder le calcul de
   `depth_affine` tel quel (sert toujours de fallback + ré-ancrage radial).
2. Calculer `flow_fwd, flow_bwd` entre frame `idx-1` et `idx` — déjà
   quasi-disponible : `WAFTWrapper.run()` ne renvoie aujourd'hui que le
   masque binaire, il faudrait soit stocker le flow dense brut (`stats["flows"]`),
   soit ré-appeler `infer_pair` bidirectionnellement dans la boucle.
3. Remplacer `aligned_depth_np = align_depth_frame(...)` par un appel à
   `propagate_depth_via_flow(...)`, avec `fused_mask` de la frame précédente
   comme `disocclusion_mask`.
4. Exposer `fb_err_threshold`, `pole_margin_frac`, `decay` comme paramètres
   de `run_video` (même esprit que `alignement_method`), pour pouvoir A/B
   tester comme `temporal_consistency` l'a été pour B1.
5. Le coût CPU/GPU supplémentaire est ~2x le flow WAFT (il faut le sens
   retour en plus de l'aller, déjà utilisé pour les masques de mouvement) —
   négligeable comparé au coût DA360 + SAM3.
