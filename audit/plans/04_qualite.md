# Plan 04 — Qualité : invariants jamais cassés

## Invariant maître (non négociable)

> Pour tout prompt `p`, la sortie (logits, et donc tokens échantillonnés à graine fixée) du
> système optimisé est **identique** à celle du modèle de référence pleine précision exécuté
> dans son dtype natif, à une borne d'erreur flottante explicite et prouvée près.

## Les 4 invariants opérationnels

### I1 — Parité numérique des logits
- **Définition** : `‖logits_opt − logits_ref‖_∞ ≤ ε`, avec `ε` = borne d'arrondi flottant
  documentée (cf. §borne).
- **Cas idéal (bit-à-bit)** : pour les optimisations qui ne changent PAS l'ordre des
  opérations flottantes (mmap A2, KV-cache exact A3 sur même ordre, cache RAM A5), exiger
  `ε = 0` (égalité binaire des logits).
- **Cas « même maths, ordre différent »** (A1 si la sommation des experts change d'ordre) :
  `ε` borné par l'analyse d'erreur de la sommation (cf. §borne).

### I2 — Déterminisme
- Même entrée + même graine ⇒ même sortie, run après run. Impose : `torch.use_deterministic_
  algorithms(True)` en mode validation, désactivation de toute source non déterministe
  (threads de réduction, TF32). Le code actuel utilise `ThreadPoolExecutor`
  (`airllm_base.py:441`) **uniquement pour l'I/O** (charge les poids), pas pour le calcul →
  ne casse pas le déterminisme du forward. À **vérifier** : `move_layer_to_device` est appelé
  dans le thread principal (`airllm_base.py:469`), donc l'ordre de matérialisation est
  déterministe. ✔ `[à confirmer par test I2]`.

### I3 — Conservation du dtype (RISQUE IDENTIFIÉ)
- **Problème réel** : `set_module_tensor_to_device(..., dtype=self.running_dtype)`
  (`airllm_base.py:317-319`) **caste** tous les poids vers `running_dtype` (défaut `float16`,
  `airllm_base.py:57`). Si le modèle de référence est en **bf16**, le caster en fp16 **change
  les valeurs** (bf16 et fp16 ont des mantisses différentes) → **violation potentielle de
  l'invariant**.
- **Règle qualité** : `running_dtype` DOIT égaler le dtype natif des poids du modèle de
  référence (lu depuis `config.torch_dtype` / l'entête safetensors). Tout cast vers un dtype
  plus pauvre est **interdit** comme voie principale (c'est de la quantization déguisée).
- Cast vers un dtype **plus riche** (fp16→fp32) est sans perte *si* le calcul de référence est
  aussi défini en fp32 ; sinon il change l'arrondi → à traiter comme « même maths, ordre/precision
  différent » avec borne (I1).

### I4 — Équivalence structurelle du graphe
- Les optimisations ne doivent pas changer le graphe de calcul, seulement l'ordonnancement
  I/O et le set de poids chargés. A1 (MoE) est le seul item qui touche au graphe (il *omet*
  des branches à coefficient nul) → exige la démonstration du Plan 09.

## Borne d'erreur flottante admise

- **Optimisations sans réordonnancement** (A2, A4, A5, A6, mmap) : `ε = 0`. Aucune tolérance.
  Si un seul bit diffère, c'est un bug, pas de l'arrondi.
- **A1 (MoE)** : la sortie de couche est `Σ_{e∈S} g_e·E_e(x)`. Si l'on préserve **le même
  ordre de sommation** que la référence (itérer les experts dans l'ordre d'indice croissant,
  comme le fait le forward dense qui les somme tous mais avec `g_e=0` pour les absents), alors
  le résultat est **bit-à-bit identique** → `ε = 0`. La preuve : ajouter `0·E_e(x)` ne change
  pas une somme flottante (x+0.0 == x en IEEE-754, sauf `x = -0.0`/NaN — cas exclus pour des
  activations finies). Donc A1 vise **`ε = 0`**, pas seulement « borné ».  `[HYPOTHÈSE à
  prouver : le forward MoE de référence somme bien dans cet ordre ; sinon reproduire son ordre]`.
- **A3 (KV-cache)** : mémoïsation exacte ⇒ `ε = 0` si l'attention est calculée sur exactement
  les mêmes (k,v) dans le même ordre.

## Procédure de vérification (le « gardien »)

1. Choisir un **modèle de référence** exécutable intégralement (petit MoE + au moins un slice
   d'un grand modèle) — cf. Plan 05.
2. Construire l'**oracle** : exécuter HF Transformers standard (non-AirLLM) sur le même prompt,
   même dtype, même device, récupérer les logits de référence.
3. Comparer logits_opt vs logits_ref selon la borne de l'item.
4. **Test bloquant en CI** : tout commit qui fait passer `ε > ε_admis` est rejeté.

---

## Section transversale A — Justification

On distingue `ε = 0` (bit-à-bit) de `ε` borné parce que l'invariant « sans perte » du sujet
est strict mais admet « l'erreur d'arrondi flottant inhérente, bornée et prouvée ». On choisit
de viser `ε = 0` partout où c'est atteignable (la majorité des items) et de ne tolérer une
borne >0 que là où le réordonnancement est inévitable — et alors de la prouver. C'est plus
exigeant que « une petite perte est ok » et c'est volontaire.

## Section transversale B — Preuve sans perte

Ce plan EST le dispositif de preuve. Sa propre validité repose sur la justesse de l'oracle
(Plan 05). Auto-preuve : le gardien ne valide un item que si sa borne est respectée sur le
jeu de référence ET sur les stress tests adverses.

## Section transversale C — Tests

- `test_parity_bitexact` : pour A2/A4/A5/A6, assert `torch.equal(logits_opt, logits_ref)`.
- `test_parity_bounded` : pour A1/A3, assert `(logits_opt-logits_ref).abs().max() <= eps`.
- `test_dtype_preserved` : assert `running_dtype == config natif` (gardien I3), échoue si
  quelqu'un remet `float16` par défaut sur un modèle bf16.
- `test_determinism` : 2 runs, `torch.equal`.

## Section transversale D — Stress tests

- Parité maintenue sous : contexte long, RAM saturée (l'éviction `to("meta")` ne doit pas
  corrompre un poids), coupure/reprise (un shard partiellement écrit ne doit pas passer le
  marqueur `.done`, `safetensor_model_persister.py:23` — vérifier l'atomicité), throttling
  (la lenteur ne change pas la valeur — à prouver par run lent vs rapide → mêmes logits).

## Section transversale E — Mitigation

- **Risque I3 (dtype)** : c'est le piège silencieux le plus dangereux (perte invisible).
  **Parade** : gardien `test_dtype_preserved` bloquant + forcer `dtype` depuis l'entête.
  **Plan B** : si un modèle n'a pas de dtype natif clair, exécuter en fp32 (sur-précision) et
  comparer à la référence fp32.
- **Risque oracle faux** : si l'oracle lui-même est buggé, on « prouve » une fausse parité.
  **Parade** : oracle = HF Transformers vanille non modifié (autorité indépendante), sur
  petit modèle entièrement vérifiable à la main sur ≥1 token.
