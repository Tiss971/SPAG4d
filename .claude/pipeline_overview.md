# Pipeline: Vidéo 360° fixe → Nuage de points 3D Gaussien dynamique

## Vue d'ensemble

La pipeline transforme une **vidéo 360° monoculaire** (caméra fixe) en une **séquence de nuages de points 3D Gaussian Splat** (un PLY par frame) avec stabilisation temporelle du fond et anti-flickering sur la profondeur.

```
[Vidéo 360°]
    ↓
[1] Extraction frames (skip sampling)
    ↓
[2] Dense optical flow (WAFT) → binary motion mask M_i
    ↓ (optionnel, si flow réussit)
[3] SAM3 video tracking → refined motion masks M_i (objet-aware)
    ↓
[4] Médiane temporelle sur frames (NaN masking) → Master Background
    ↓
[5] DA360/PaGeR depth estimation sur Background → D_ref (référence rigide)
    ↓
[6] Pour chaque frame i:
    ├─ DA360/PaGeR depth estimation → D_i (brut, scale-drifted)
    ├─ Alignement affine: D_i ← s·D_i + t  (sur pixels statiques seulement)
    ├─ Stabilisation FG: médiane glissante + jump detection sur masque foreground
    ├─ Scene defaults (optionnel): compute_scene_defaults(D_i) → depth_min/max/sky_thr
    ├─ Conversion Gaussienne: depth_to_gaussians(image_i, D_i_aligned)
    ├─ Post-filtres: outlier/grazing-angle/sparse-region pruning
    └─ Export: save_ply_gsplat → gaussians/frame_i.ply
    ↓
[∞] Sequence de N fichiers .ply (N = nombre de frames capturées)
```

---

## Détail des étapes

### [1] Extraction frames (`extract_video_frames`) — video.py:40–93

```python
def extract_video_frames(video_path: str, skip_step: int = 10) 
    → tuple[np.ndarray, int]
```

- Lit la vidéo au complet avec OpenCV, **skips sampling** : ignore `skip_step-1` frames de suite pour chaque frame conservée (ex: skip_step=10 → 10% des frames).
- Retourne `(frames [N,H,W,3] uint8 BGR, meta dict)` où `meta` contient `W`, `H`, `fps`, `total` (nombre total de frames du fichier source).
- **Pas de downscaling ici**; le downscaling optionnel pour WAFT survient plus tard dans `run_video` si la résolution > 1024 px max.

**État**: Fonctionnel, simple. Pas de issues connues à ce stade.

---

### [2] Dense optical flow (WAFT) — detect_opticalflow.py:152–230

```python
class WAFTWrapper:
    def __init__(self, checkpoint: str, config: str)
    def infer_pair(frame1, frame2) → flow [H,W,2]
    def run(frames, meta, out_dir, threshold) → dict
```

- **Checkpoint**: modèle WAFT entraîné (ex: `/raid/mb273924/_DATASETS/uptale/tar-c-t.pth`).
- **Config**: paramètres d'inférence JSON (ex: `config/a1/tar-c-t.json`).
- `infer_pair(frame1, frame2)` : convertit deux frames BGR uint8 en tensors CUDA (RGB, sans normalisation), appelle `model.calc_flow(t1, t2)`, retourne flow pixel-displacement `(H, W, 2)`.
- `run(frames, meta, out_dir, threshold)` : boucle sur paires consécutives, seuille magnitude → masque binaire `(mag > threshold) * 255`, exporte `flow_mag.mp4` (HSV colorwheel) et `flow_mask.mp4`.
- Retourne dict `{"timings": [...], "mags": [...mean magnitude...], "masks": [N, H, W] uint8}`.

**État**: Fonctionnel mais **2 bugs potentiels identifiés**:
1. **Bug: masks[-1] non initialisé** (ligne 205). Boucle `range(len(frames)-1)` ne remplit que les indices `0..len(frames)-2`. La dernière frame a un masque de garbage mémoire, pas des zéros.
2. **Bug: KeyError `meta["new_H"]/["new_W"]`** (ligne 206). Ces clés ne sont définies dans `run_video` (video.py:339-341) **que si un downscale a lieu**. Si la vidéo est déjà ≤1024 px, elles n'existent pas → crash attendu.

**Absence**: Aucune vérification bidirectionnelle forward-backward n'existe (contrairement à la spec `goal.md`). Pas de détection d'occlusion via flow consistency.

---

### [3] SAM3 video tracking — video.py:941 & 1261

Deux variantes selon le statut du flow WAFT:

#### **3a. Avec flow (segment_with_flows)** — video.py:941–1259

```python
def segment_with_flows(video_path, output_path, viz_frames, flow_masks, meta, 
                       n_total_frames, skip_step, ...)
    → outputs_per_frame: dict[frame_idx: dict[obj_id: mask]]
```

**Étapes**:
1. **DIoU Tracking** (video.py:982–1150): Détecte contours dans `flow_masks`, applique nettoyage morpho (open/close/erode/dilate avec kernels fixe), filtre contours par aire/aspect/confiance.
2. Associe chaque région de mouvement **détectée à cet instant** aux **tracks existants** via **IoU overlap** (priorité 1) ou **distance euclidienne normalisée** (fallback si IoU=0).
3. Crée une track ID neuve si aucun match.
4. Génère des vidéos diagnostic (boxes, masks) et prépare prompts SAM3 (points ou boxes).
5. **SAM3 propagation** (video.py:1172–1259): Lance une session SAM3 permanente pour toute la vidéo, enregistre les tracks détectés en tant que prompts, appelle `propagate_in_video` mode "both" (avant/arrière), accumule masques.
6. Retourne `outputs_per_frame` = `dict[frame_idx] → frame_output` où `frame_output` est la réponse SAM3 (binary_masks, obj_ids, boxes).

**État**: Fonctionnel. Tracking DIoU robuste, SAM3 bien intégré. **Manque**: pas de gel de mémoire SAM lors d'occlusion (lié au bug bidirectionnel du flow).

#### **3b. Sans flow (segment_with_sam)** — video.py:1261–1362

```python
def segment_with_sam(video_path, viz_frames, n_total_frames, skip_step, enable_viz=False)
    → outputs_per_frame: dict
```

**Étapes**:
1. Lance session SAM3 permanente.
2. Cherche une cible (ex: "human") dans les frames jusqu'à en trouver une.
3. Appelle `propagate_in_video` mode "both" sans prompts supplémentaires.

**État**: Fallback simple. Pas de multi-objet ni de tracking explicite.

---

### [4] Fond de référence (médiane temporelle) — video.py:96–173

```python
def compute_temporal_median(frames_array, quantile=0.33)
    → (median_frame, activity_mask)

def compute_temporal_median2(frames_array, masks_dict: list[dict], quantile=0.33)
    → (median_frame, activity_mask)
```

**compute_temporal_median** (simple, inutilisée dans video.py):
- Médiane naïve channel-by-channel via GPU.
- Retourne aussi une `activity_mask` basée sur std channel.

**compute_temporal_median2** (utilisée, ligne 375):
- **Médiane masquée**: NaN out les pixels mobiles (selon `masks_dict[idx]`), puis `torch.nanmedian` pour ignorer les NaN.
- Retourne `master_background` (médiane des pixels statiques) et `activity_mask` (seuillage du std, quantile par défaut 0.33).
- Gère le cas où tous les pixels d'un endroit sont masqués → NaN → remplacés par 0 post-calcul.

**État**: Fonctionnel. La médiane masquée élimine bien les objets mobiles du fond de référence.

**Impact sur goal.md**: ✓ Étape 1 (Geometry Initialization) implémentée. Pas de divergence vs spec.

---

### [5] Profondeur de référence — da360_model.py:195–238 & pager_model.py:116–155

#### **DA360Model.predict()** — da360_model.py:195–238

```python
def predict(self, image_tensor) → (depth [H,W], meta dict)
```

1. Normalise ImageNet (BGR→RGB, subtract mean/std).
2. Redimensionne à (518, 1036) en bilinear.
3. Inférence → disparity (scale-invariant, unitless).
4. Disparity → depth: `depth = 1 / (disparity + eps)`.
5. **Renormalisation par médiane par frame** (ligne 228-231):
   ```python
   for i in range(B):
       median_depth = depth[i].median()
       depth[i] = depth[i] * (5.0 / median_depth)  # set median to ~5m
   ```
   **Cette renormalisation est indépendante par frame, sans aucune référence temporelle.**
6. Upsample à résolution d'entrée (bilinear).

**Analyse**: Chaque frame recalcule sa médiane et rescale à 5m indépendamment. **C'est une source majeure de scale drift d'une frame à l'autre**, même sur un objet statique. Le *but* est d'approximer une profondeur métrique raisonnable (5m = intérieur/extérieur mipoint), mais le *coût* est que deux frames d'une même scène statique peuvent avoir des médians différents → diferentes rescales → depth_aligned essaie de corriger ça après coup.

**État**: Par conception, produit du drift de scale. C'est prévisible et accepné; l'alignement affine est le remède.

#### **PaGeRModel.predict()** — pager_model.py:116–155

```python
def predict(self, image_tensor, metric=False) → (depth [H,W], ...)
```

1. Si `metric=False` (défaut): pas de rescale supplémentaire, retourne la disparity inverse brute (scale-invariant, même comportement que DA360 brut).
2. Si `metric=True`: utilise un routeur CLIP indoor/outdoor classifiant chaque image, charge le head de scale correspondant, rescale la profondeur. **Mais**: la classification CLIP est **indépendante par frame**, aucune mémoire temporelle → même risque de drift.

**État**: Même profil que DA360 — scale-drifted par frame, mitigé par alignement affine.

---

### [6.1] Alignement affine de profondeur — video.py:760–867 & 870–939

```python
def align_depth_frame(depth_frame, depth_ref, mask_moving, method="lstsq", ...) 
    → aligned_depth [H,W]

def _estimate_scale_shift(x, y, method, ...) 
    → (s, t)  where y ≈ s*x + t
```

**Principes**:
1. **Masque statique**: pixels où `mask_moving == 0` (fond de la scène, pas d'objet mobile).
2. Extrait valeurs `x = depth_frame[static_mask]`, `y = depth_ref[static_mask]` (cibles).
3. **Normalisation**: x/y par médiane de y pour robustesse aux échelles différentes.
4. **Estimation selon method**:
   - `"lstsq"` (défaut): moindres carrés classiques, sensible aux outliers.
   - `"ransac"`: RANSAC (scikit-learn), robuste aux ~50% outliers.
   - `"median"`: Siegel simplifié (médiane de ratios y/x), très rapide et robuste.
5. **Dénormalisation et clipping**: applique `s = np.clip(s, [0.5, 2.0])`, recalcule t si s a clippé.
6. **Application complète**: `depth_frame = s * depth_frame + t` (partout, pas juste sur static_mask), puis clip ≥0.

**État**: Implémentation complète et bien documentée. Paramètres ajustables.

**Dépendance critique**: **La qualité de l'alignement dépend entièrement de la qualité de `mask_moving`**. Si le masque est incomplet ou en retard d'une frame, des pixels mobiles se glissent dans `static_mask` → estimations `s, t` polluées.

---

### [6.2] Stabilisation du foreground — video.py:435–554

Deux classes coexistent; **seule FGDepthStabilizer2 est actuellement utilisée** (ligne 554).

#### **FGDepthStabilizer (EMA baseline, code mort?)** — video.py:435–503

```python
class FGDepthStabilizer:
    def __init__(self, jump_threshold=0.5, ema_alpha=0.3, max_outlier_run=4, ...)
    def __call__(self, depth, mask) → depth_corrected
```

- Maintient une baseline EMA (exponential moving average) de la profondeur médiane foreground.
- Détecte les jumps > `jump_threshold` et les corrige via scale relative à la baseline.
- Accepte un changement réel après `max_outlier_run` frames consécutives de jump.

**État**: Fonctionnel mais **pas utilisé** (l'instance à ligne 504 est écrasée par FGDepthStabilizer2 à ligne 554).

#### **FGDepthStabilizer2 (médiane glissante, actuel)** — video.py:508–554

```python
class FGDepthStabilizer2:
    def __init__(self, buffer_size=11, jump_threshold=0.5, scale_clip=(0.5, 2.0))
    def __call__(self, depth, mask) → depth_corrected
```

- Maintient un **deque de buffer_size** valeurs brutes de profondeur médiane foreground.
- Baseline = médiane glissante du buffer (ignores peaks < buffer_size/2).
- À chaque frame: si |current - baseline| > jump_threshold, applique scale correction (clippée).
- Sinon, retourne la depth inchangée.

**Avantages vs EMA**:
- Pas d'EMA alpha à tuner; la médiane s'adapte automatiquement.
- Robuste par défaut à ~50% de pics outliers (la médiane les ignore).
- Pas de compteur de "outlier run" à tracker.

**État**: Fonctionnel et actuellement actif. Utilisé à video.py:601–606.

---

### [6.3] Scene defaults (optionnel) — scene_analysis.py:8–60

```python
def compute_scene_defaults(depth_map, image_height=None, sky_mask=None) → dict
```

**Calcul statistique pur**, **independant par frame**:
- Valides: `depth > 0.01 & isfinite`, optionnel moins sky_mask.
- Percentiles: p1, p50, p95, p99 des profondeurs valides.
- **Retourne**:
  - `sky_threshold = max(p95, p1 + 1.0)` ← seuil de détection du ciel.
  - `depth_min = max(0.01, p1 * 0.8)` ← marge 20% sous le 1er percentile.
  - `depth_max = p99 * 1.1` ← marge 10% au-dessus du 99e percentile.
  - `orbit_radius = max(0.05, p50 * 0.05)` ← 5% de la profondeur médiane.

**Utilisé dans `to_gaussians`** (video.py:176-189):
```python
if depth_min is None:  # i.e., not passed as argument to run_video
    depth_min = scene_defaults["depth_min"]
```

**Impact sur le scintillement**: Si `depth_min`/`max` sont `None` au démarrage de `run_video`, ils sont **recalculés indépendamment à chaque frame** dans `to_gaussians`. Cela signifie que les seuils `filter_gaussian_candidates` (qui décide quels pixels deviennent des Gaussiennes) varient légèrement frame-par-frame, indépendamment de la stabilisation de la profondeur elle-même.

**State**: Fonctionnel mais source potentielle de scintillement (voir section C des causes, ci-dessous).

---

### [6.4] Conversion Gaussienne — spag4d/spag_converter.py:37–181 & core.py:322–354

```python
def depth_to_gaussians(erp_image, depth_map, params=None, device=None) → dict

def _run_spag_pipeline(image_tensor, depth, depth_min, depth_max, sky_threshold, stride) → dict
```

**Pipeline SPAG (Spherical Projection)**:
1. **SPAGParams**: stride (pixel sampling), depth_min/max, sky_threshold, pole_thinning, etc.
2. Crée une grille sphérique via `create_spherical_grid` (spherical_grid.py).
3. **Filter gaussian candidates** (scene_filter.py:197–364): applique masques:
   - Depth range: `depth_min ≤ depth ≤ depth_max`.
   - Sky detection (si mode "depth"): `depth > sky_threshold` → masqué.
   - Pole thinning (stochastique, seed=42): réduit les Gaussiennes aux pôles où la densité est suréchantillonnée.
4. **Projection sphérique**: chaque pixel survivant → Gaussienne 3D:
   - Position: `means = depth * ray_direction` (ray = direction eq→cartésienne).
   - Couleur: échantillonnée directement du pixel panorama (sRGB, pas de gamma).
   - Scale/opacité: dérivées de la géométrie locale (voir spag_converter.py:90–160).
5. **Retour**: dict `{means [N,3], scales [N,3], quats [N,4], colors [N,3], opacities [N,1]}`.

**État**: Fonctionnel, bien testé. Le design de base est sain.

**Propriétés**:
- **Chaque pixel survivant** devient une Gaussienne indépendante.
- **Pas de suivi inter-frames**: aucune correspondance pixelle-à-pixelle entre frames N et N+1.
- **Nombre de Gaussiennes variable** : dépend de quels pixels passent les filtres.

---

### [6.5] Post-filtres — scene_filter.py:536–633

Trois filtres appliqués indépendamment après `depth_to_gaussians`, chacun **stateless et recalculé par frame**:

#### **prune_outliers** — scene_filter.py:536–599

```python
def prune_outliers(gaussians, strength=0.5, k=16) → gaussians_pruned
```

- **Statistical Outlier Removal (SOR)** via k-NN: pour chaque Gaussienne, distance moyenne aux k voisins.
- Statistiques globales: mean & std des distances moyennes.
- Garde points où `dist < global_mean + std * std_ratio`.
- Mapping: `strength [0,1] → std_ratio [3.0, 0.5]` (higher strength = more aggressive).

**Effet**: Supprime les points isolés ("floaters"), mais les seuils dépendent de la distribution complète du nuage **de cette frame**. Même un point stationnaire peut être supprimé si le nuage change de densité globale (ex: objet mobile sort du champ).

#### **prune_grazing_angle** — scene_filter.py:368–458

```python
def prune_grazing_angle(gaussians, depth_map, stride=2, max_angle_deg=80.0, normals=None) 
    → gaussians_pruned
```

- Identifie les Gaussiennes à des angles rasants (surface presque de profil par rapport au rayon de vue).
- Calcule gradient de profondeur local, rapporte à la profondeur (relative gradient).
- Seuil: `relative_grad < tan(max_angle_deg) * angular_spacing`.
- Optionnel (PaGeR): utilise des normales de surface apprises au lieu du gradient.

**Effet**: Réduit les artéfacts de "stries" aux bords d'objets, mais les seuils basés sur gradient peuvent fluctuer frame-à-frame si la profondeur brute varie.

#### **prune_sparse_regions** — scene_filter.py:461–533

```python
def prune_sparse_regions(gaussians, min_neighbors=3, radius_multiplier=3.0, k=8) 
    → gaussians_pruned
```

- K-NN: voisins dans un rayon `= scale * radius_multiplier`.
- Garde points avec ≥ min_neighbors voisins.

**Effet**: Supprime les points isolés en cluster creux, mais le seuil dépend de la distribution locale du nuage **de cette frame**.

**Impact global**: Ces 3 filtres introduisent une **dépendance frame-par-frame sur la distribution globale du nuage**. Un point 3D identique peut survivre à la frame N (si le nuage est dense) et être supprimé à la frame N+1 (si le nuage s'amincit ailleurs) → **flicker/popping inévitable sans identité persistante**.

---

### [7] Export PLY — ply_writer.py:25–154

```python
def save_ply_gsplat(gaussians, path, sh_degree=0, colors_linear=True) → None
```

**Processus**:
1. Extrait means, scales, quats, colors, opacities des tensors Pytorch.
2. Log-encode scales.
3. Convertit colors en coefficients SH DC (ou SH band-1 si fourni).
4. Crée un PLY structuré compatible gsplat viewers.
5. Écrit sur disque.

**État**: Fonctionnel, standard.

**Limitation critique**: **Aucune mécanique de correspondance inter-frames**:
- Nombre de Gaussiennes varie librement à chaque frame.
- Ordre des points est simplement l'ordre d'iteration du array (row-major après pruning).
- Pas d'ID persistant, pas de tracking de point → un viewer qui interpole entre frames `frame_N.ply` et `frame_(N+1).ply` verra du **popping/scintillement** simplement dû à cette absence d'identité, indépendamment de la qualité de l'alignement de profondeur.

---

## Causes probables des 3 points de blocage

### **A. Qualité des masques dynamiques (WAFT + SAM3)**

**Problèmes identifiés**:

1. **Bugs latents bloquants** ✓ **CORRIGÉS (2026-07-09)**:
   - ~~`masks[-1]` non initialisé~~ → `np.zeros` au lieu de `np.empty`.
   - ~~`meta["new_H"]/["new_W"]` non définis~~ → définis systématiquement dans `run_video` (ligne 350-351), et `WAFTWrapper.run` tire les dimensions de `frames[0].shape[:2]`.

2. **Absence de forward-backward consistency**:
   - Le goal.md prévoyait: "If the WAFT error spikes, flag an occlusion and pause SAM's memory accumulation".
   - **Pas implémenté** : seul le flow avant (frame t → t+1) est calculé, jamais le backward (frame t+1 → t).
   - Conséquence: objets occlus ne sont pas détectés → SAM continue de tracker fantôme.
   - *Statut: optionnel si masques SAM suffisants.*

3. ~~**Morphologie de nettoyage appliquée tardivement**~~ **ANALYSE INCORRECTE**:
   - La morphologie (open/close/erode/dilate) est appliquée **avant** extraction des contours (ligne 987–990), sur le masque binaire de flow.
   - Pas de problème identifié ici.

### **B. Alignement affine (`align_depth_frame`)**

**Causes racine**:

1. **DA360/PaGeR rescale la profondeur indépendamment par frame** ✓ **CORRIGÉ (2026-07-09)**:
   - ~~Per-frame rescaling~~ **Remplacé**: Nouveau mode `temporal_consistency=True` désactive le rescale par frame.
   - **Nouveau workflow**: Calculer la médiane une seule fois sur la référence, appliquer ce scale_factor à toutes les frames.
   - **Code**: `video.py` ligne 407-415 calcule `scale_factor_to_5m` une seule fois, l'applique globalement.
   - **Benchmark**: Script `benchmark_temporal_consistency.py` mesure l'amélioration (variance de profondeur, alignement affine, flicker).

2. **L'alignement dépend de la qualité des masques SAM** (potentiel blocker si masques mauvais):
   - `fused_mask` = SAM ∪ `activity_mask` (ligne 580-591).
   - Si SAM est incomplet, des pixels mobiles contaminent le masque statique → estimations `s, t` biaisées.
   - **Atténuation**: Si foreground représente <10% de l'image, la contamination mineure affecte peu les statistiques robustes (RANSAC/médiane).
   - **Remarque**: B2 est un blocker réel **uniquement** si les masques SAM sont de mauvaise qualité (ce qui renvoie au point A).

3. **Masque global d'activité** (statique, calculé une fois):
   - `activity_mask` est calculé lors du `compute_temporal_median2` (ligne 375) : **une seule fois**, pas par frame.
   - C'est un seuillage du std channel-wise sur la médiane temporelle globale.
   - Pas de problème d'inconstance temporelle ici.

### **C. Stabilisation FG / scintillement résiduel**

**Causes racine**:

1. **`compute_scene_defaults` recalculé par frame sans lissage temporel**:
   - Si depth_min/depth_max sont `None` (défaut), chaque frame recalcule ses percentiles indépendamment.
   - Cela fait varier quels pixels passent les seuils `filter_gaussian_candidates`.
   - Solution: fixer depth_min/depth_max une fois sur `depth_ref` et les réutiliser pour toutes les frames.

2. **Filtres de pruning dépendent de la distribution du nuage de cette frame**:
   - `prune_outliers`: seuil basé sur KD-tree stats de cette frame → peut garder point X à la frame N, le supprimer à N+1.
   - `prune_grazing_angle`, `prune_sparse_regions`: même problème.
   - **Solution**: passer des seuils absolus/fixes plutôt que relatifs par frame.

3. **`save_ply_gsplat` n'a aucune correspondance inter-frames**:
   - Différente comptes de Gaussiennes d'une frame à l'autre.
   - Aucun ID persistant → viewers qui interpolent verront du popping inévitable.
   - **Solution**: soit tracker les points (complexe), soit accepter le scintillement et utiliser une post-production temporelle (flou optique, débruitage).

4. **Deux stabilisateurs coexistent**:
   - `FGDepthStabilizer` (EMA) à ligne 435 : jamais utilisé, code mort ou en cours de transition.
   - `FGDepthStabilizer2` (médiane glissante) à ligne 508 : actuel, mais recalcule sur la seule profondeur foreground brute; ne tient pas compte de comment `compute_scene_defaults` recalcule aussi.

---

## Ordre suggéré d'investigation & correction

### **1. Fixer bugs bloquants WAFT** ★★★ Urgent **✓ FAIT (2026-07-09)**
   - ✓ `masks` initialisé avec `np.zeros` plutôt que `np.empty`.
   - ✓ `meta["new_H"]/["new_W"]` défini systématiquement dans `run_video`.
   - ✓ `WAFTWrapper.run` tire dimensions de `frames[0].shape[:2]`.

### **2. Validez qualité des masques SAM** ★★★ Critique si foreground contamination élevée
   - **Diagnostic**: Visualisez les masques SAM versus flow masques. Chevauchement?
   - Si SAM échoue/est en retard → implémenter forward-backward flow consistency (étape 3).
   - Si SAM OK mais alignement imparfait → probablement dû à B1 (rescale drift), pas à B2.
   - **Remarque**: Si foreground < 10% de l'image, contamination mineure ne devrait pas affecter significativement les estimations robustes.

### **3. Ajouter forward-backward flow consistency** ★★ Si B2 (masques) reste problématique
   - Calculer flow t+1→t en plus de t→t+1.
   - Comparer: `consistency_error = ||flow_fwd(x) + flow_bwd(x + flow_fwd(x))||`.
   - Masquer les pixels où error > threshold (occlusions potentielles).
   - Geler SAM memory sur ces pixels.
   - **Impact**: Masques robustes pour objets occlus/rapides → meilleur alignement affine.

### **4. Fixer scene defaults** ★★ Haute priorité, bas effort (C1)
   - À `run_video`, si depth_min/depth_max/sky_threshold restent `None`, les calculer une seule fois sur `depth_ref` et les passer à tous les `to_gaussians` appels.
   - Élimine une source directe de scintillement (variation des seuils per-frame).
   - **Code change**: ~3 lignes.

### **5. Évaluer & stabiliser seuils de pruning** ★ Moyenne priorité (C2/C4)
   - **Question**: Est-ce que du popping/scintillement résiduel est acceptable ou critique?
   - Si acceptable → accepter et documenter (peu d'action nécessaire).
   - Si critique → optionnel: passer seuils absolus aux filtres, ou désactiver pruning (`outlier_pruning=0.0`).

### **6. Question ouverte: correspondance inter-frames** ★ Design decision (C3)
   - Le projet a-t-il besoin d'une **identité persistante des Gaussiennes** d'une frame à l'autre?
   - Ou est-ce que le **scintillement est acceptable** si la caméra est statique (peu de parallaxe, peu d'attente d'interpolation temporelle)?
   - **Si non requis**: accepter et documenter le scintillement.
   - **Si requis**: tracking de points 3D via optical flow dense ou correspondance → beaucoup plus complexe.

---

## Résumé: Hiérarchie des causes de scintillement

```
[Scintillement visuel observé]
  ↓
  ├─ Cause 1 (profondeur - PRIMAIRE): DA360 rescale indépendamment par frame
  │  └─ Mitigation: align_depth_frame (utilise masques SAM pour pixels statiques)
  │     └─ Efficace SI masques SAM corrects; peu sensible si foreground < 10%
  │
  ├─ Cause 2 (masques - DÉPENDANCE): WAFT+SAM3 incomplets/en retard
  │  └─ Bugs WAFT: ✓ CORRIGÉS (2026-07-09)
  │  └─ Mitigation si masques restent mauvais: forward-backward consistency
  │
  ├─ Cause 3 (seuils scene - SECONDAIRE): compute_scene_defaults per-frame
  │  └─ Mitigation: fixer depth_min/max une fois sur depth_ref (facile)
  │
  ├─ Cause 4 (filtres - SECONDAIRE): prune_* dépendent de distribution frame
  │  └─ Mitigation: seuils absolus ou désactiver pruning (optionnel)
  │
  └─ Cause 5 (identité PLY - FONDAMENTALE): Pas d'ID inter-frames
     └─ Popping inévitable sans tracking de points (complexe)
     └─ Mitigation: accepter ou implémenter tracking 3D
```

**Ordre de priorité**:
1. **Diagnostic**: Vérifier qualité masques SAM (causes 1 & 2).
2. **Quick wins**: Fixer scene_defaults (cause 3) → gain visuel potentiel.
3. **Analyse**: Évaluer si popping résiduel (cause 5) est acceptable ou doit être adressé.

---

## Fichiers clés (statut)

| Fichier | Fonctionnalité | Statut | Issues |
|---------|---|---|---|
| `spag4d/video.py` | Pipeline principal | ✓ Complet | ✓ temporal_consistency mode (2026-07-09), scale_factor_to_5m calculé une seule fois |
| `spag4d/detect_opticalflow.py` | WAFT wrapper | ✓ Complet | ✓ masks[-1] initialisé à 0, H/W tirés de frames (2026-07-09) |
| `spag4d/da360_model.py` | Depth estimation | ✓ Complet | ✓ temporal_consistency=True mode (2026-07-09), backward compatible |
| `spag4d/pager_model.py` | Depth estimation alt | ✓ Complet | ✓ temporal_consistency=True mode (2026-07-09), backward compatible |
| `spag4d/scene_analysis.py` | Scene defaults | ⚠ Opportunité | Recalculé per-frame → source flicker mineure (cf. C1) |
| `spag4d/scene_filter.py` | Pruning filters | ✓ Complet | Stateless → flicker par design (cf. C2/C4) |
| `spag4d/spag_converter.py` | SPAG Gaussians | ✓ Complet | Pas d'identité inter-frames (cf. C3) |
| `spag4d/ply_writer.py` | Export PLY | ✓ Complet | Pas de correspondance inter-frames (par design) |