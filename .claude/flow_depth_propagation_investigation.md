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
  pas besoin de SAM3), qui calcule pour chaque frame :
  - la depth actuelle (`align_depth_frame`, méthode en prod),
  - la depth propagée par flow (ce prototype),
  et compare l'écart-type temporel sur les pixels foreground (proxy direct du
  scintillement, même métrique que `_temporal_depth_std.jpg` déjà utilisée
  dans le pipeline).

## Résultat empirique (préliminaire)

Testé sur `temp_fast_track_h264.mp4` (1024×512 ERP, 75 frames @ 30fps, scène
d'entrepôt) :

| Méthode | Écart-type temporel moyen (pixels foreground) |
|---|---|
| Alignement affine (actuel) | 0.497 m |
| Propagation par flow (prototype) | 0.445 m |

→ **~11% de réduction du bruit temporel** sur les pixels foreground, et la
courbe de profondeur médiane du foreground est visiblement plus lisse (moins
de pics ponctuels frame-à-frame — voir `benchmark_flow_prop/comparison.png`).

**Limite importante de ce test** : `temp_fast_track_h264.mp4` ne contient pas
de sujet clairement mobile — le masque foreground (seuillage flow) ne capte
qu'un petit cluster de bruit/flicker (`_debug_fg_mask.jpg`), pas un objet
traversant la scène. Le chiffre ci-dessus mesure donc surtout un
**lissage du bruit de fond du monoculaire**, pas encore un cas d'usage réel
(personne qui marche, objet qui bouge). **Prochaine étape recommandée avant
d'intégrer** : rejouer ce même benchmark sur une vidéo avec un vrai sujet en
mouvement (et idéalement les masques SAM3 réels plutôt que le proxy par
seuillage de flow) pour confirmer que le gain se maintient — et surtout pour
vérifier le comportement aux limites (objet traversant la couture ERP, objet
s'approchant radialement de la caméra) qui ne sont pas exercées par ce clip.

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
