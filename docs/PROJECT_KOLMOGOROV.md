# Project KOLMOGOROV — Plan de recherche & d'ingénierie

> **Thèse :** les poids d'un LLM n'ont pas besoin d'être *stockés et déplacés* ; ils
> peuvent être *générés à la volée* par un opérateur résident minuscule. Le goulot
> de l'inférence n'est pas le calcul ni la mémoire — c'est qu'on a confondu une
> **fonction** avec sa **table tabulée**.
>
> Sous-titre : *Inférence sans matérialisation des poids.*
> Statut : **PLAN — pré-exécution.** Aucun résultat ici n'est acquis ; tout est
> pré-enregistré et falsifiable.

Document type : RFC de recherche / dossier qualité. Public : équipe d'ingénierie,
revue scientifique, reviewers externes. Convention : tout seuil chiffré ci-dessous
est **pré-enregistré** (pre-registration) — on ne le bouge pas après avoir vu les
données, sous peine d'invalider la réfutation.

---

## 0. Résumé exécutif

On teste une seule hypothèse, décomposée en deux sous-claims **séparables** :

- **C1 (information / ML)** — Les poids d'un LLM entraîné ont une *longueur de
  description effective* très inférieure à leur taille nominale, conditionnellement
  à un générateur appris partagé entre couches. Mesurable **sans aucun argument
  système**, en pur ML.
- **C2 (systèmes)** — Sur la roofline réelle, **recalculer** une couche depuis ce
  générateur coûte moins de temps-mur que la **lire** depuis le tier de stockage le
  plus lent.

La **découverte** exige *C1 ∧ C2*. Mais **C1 seul** est déjà un résultat publiable
(énoncé sur le contenu informationnel intrinsèque des poids), et **C2 seul** est un
résultat systèmes. On les attaque donc indépendamment, C1 d'abord (moins cher).

**Decision gate central :** on ne dépense pas un GPU de classe 70B tant que C1 n'a
pas montré un signal de *scaling favorable* (cf. §6, la « loi d'échelle décisive »)
sur des modèles ≤ 1.4B.

---

## 1. Plan scientifique — hypothèse formelle

Soit un modèle `M` de `L` couches, poids `W = {W_ℓ}_{ℓ=1..L}`, `N = Σ_ℓ |W_ℓ|`
paramètres. Soit `G_θ` un **générateur résident** de paramètres `θ` (avec
`|θ| ≪ N` en bits) produisant des approximations `Ŵ_ℓ = G_θ(ℓ, c_ℓ)`, où `c_ℓ` est
un **code par couche** (petit) éventuellement nul.

**Hypothèse H1 (principale).** Il existe `(G_θ, {c_ℓ})` tel que :
1. **(qualité)** la perte de tâche de `M[Ŵ]` est dans un `ε` de celle de `M[W]` ;
2. **(information)** les bits effectifs par poids
   `b_eff = (|θ| + Σ_ℓ |c_ℓ|) / N` battent strictement la frontière de
   quantification SOTA à iso-qualité ;
3. **(systèmes)** le temps de recompute d'une couche est inférieur à son temps de
   reload sur au moins un profil matériel réaliste.

**Hypothèses nulles (à réfuter) :**
- `H0_C1` : aucun générateur avec `b_eff` sous la frontière de quantification SOTA
  n'atteint `Δppl ≤ ε`. (Les poids sont incompressibles au-delà du connu.)
- `H0_C2` : `∀` matériel réaliste, `t_recompute ≥ t_reload` pour tout générateur
  passant C1.

**Pourquoi ce n'est pas absurde (priors empiriques forts, cf. §3) :**
Denil et al. 2013 prédisent ~95 % des poids depuis ~5 % ; Aghajanyan et al. 2020
montrent une *dimension intrinsèque* basse du fine-tuning ; LoRA exploite ce
sous-espace ; le pruning >90 % et la quantification 1.58-bit (BitNet) tiennent ;
et des modèles de diffusion **génèrent déjà des poids fonctionnels** (Wang et al.
2024 *Neural Network Diffusion* ; Peebles et al. 2022 *G.pt* ; Schürholt et al.
2022). L'information intrinsèque des poids est manifestement basse. La question
n'est pas *si* mais *combien*, et *si recompute < reload.*

---

## 2. Fondements mathématiques

### 2.1 Codage conditionnel & MDL
On code `W` via le couple (générateur partagé `θ`, codes par couche `c_ℓ`). La
longueur de description (Rissanen MDL ; Hinton & van Camp 1993, *minimiser la MDL
des poids*) est :

```
DL(W) = |θ| + Σ_ℓ |c_ℓ| + Σ_ℓ DL( W_ℓ - Ŵ_ℓ )      (bits)
```

`θ` est **payé une fois** et reste **résident** (donc hors du bus lent à
l'inférence). Les bits qui **traversent réellement** la frontière lente par forward
sont donc :

```
B_read ≈ Σ_ℓ |c_ℓ|      (θ résident, amorti)   ≪   Σ_ℓ |W_ℓ| = N·bitwidth
```

C'est rigoureux par construction (codage conditionnel) ; la **magnitude** de
`Σ|c_ℓ|` à iso-qualité est la question empirique de C1.

### 2.2 Borne inférieure — la « loi » conjecturée
Pour une entrée `x`, soit `S(x)` la sous-computation réellement nécessaire à la
sortie `Y` (sparsité contextuelle). On conjecture une **limite de Shannon de
l'inférence bornée en mémoire** :

```
   B*(x)  ≥  I( Y ; S(x) | résident )          (bits devant traverser / token)
```

et `B*(x) ≪ N·bitwidth`. **Statut : conjecture.** La partie *prouvable* est la
borne supérieure constructive (le codage §2.1 atteint `Σ|c_ℓ|`). La partie *borne
inférieure* fait l'objet d'un travail théorique séparé (data-processing inequality
sur le canal stockage→calcul) ; on ne prétend pas la prouver dans ce projet, on la
*pose* et on mesure à quel point nos codes s'en approchent.

**Corollaire unificateur :** sparsité contextuelle, MoE, quantification extrême et
décodage spéculatif sont tous des **codes** réduisant `B_read` ; la génération de
poids est le code qui vise `θ` résident + `c_ℓ` minimal. Quatre « astuces »
deviennent quatre approximations d'une même borne.

### 2.3 Le crossover systèmes (cœur quantitatif de C2)
Reload d'une couche : `t_reload = Bytes(W_ℓ) / BW_slow`.
Recompute : `t_recompute = FLOPs(G_θ→W_ℓ) / (C_gpu · u)`, `u` = utilisation.

Recompute gagne ssi :

```
FLOPs(G_θ→W_ℓ)  <  Bytes(W_ℓ) · ( C_gpu · u / BW_slow )
```

Ordre de grandeur : `C_gpu/BW_slow ≈ (1e14 FLOP/s)/(5e9 B/s) ≈ 2·10⁴ FLOP/octet`.
On dispose donc d'un **budget d'environ 2·10⁴ FLOPs par octet de poids produit**
avant de toucher le mur. Un poids fp16 = 2 octets ; le générer via un opérateur de
rang `r` coûte `O(r)` FLOPs. **La marge est énorme** — c'est précisément l'écart
roofline qu'AirLLM laisse inutilisé. C2 est *a priori* plausible ; le simulateur
(§5) le tranche par profil matériel.

---

## 3. Sources / état de l'art *(à confirmer en revue de littérature §11.0)*

**Théorie de l'information & MDL**
- Shannon 1948, *A Mathematical Theory of Communication*.
- Landauer 1961, *Irreversibility and heat generation in computing*.
- Rissanen 1978 ; Grünwald 2007, *The MDL Principle*.
- Hinton & van Camp 1993, *Keeping neural networks simple by minimizing the
  description length of the weights*. ← fondation directe.

**Information intrinsèque des poids**
- Denil et al. 2013, *Predicting Parameters in Deep Learning*. ← prior le plus fort.
- Aghajanyan et al. 2020, *Intrinsic Dimensionality Explains the Effectiveness of
  Language Model Fine-Tuning*.
- Frankle & Carbin 2019, *The Lottery Ticket Hypothesis*.
- Hu et al. 2021, *LoRA*.

**Génération / représentation de poids**
- Ha et al. 2016, *HyperNetworks*.
- Sitzmann et al. 2020, *SIREN* (représentations neuronales implicites).
- Schürholt et al. 2022, *Hyper-Representations*.
- Peebles et al. 2022, *G.pt — Learning to Learn with Generative Models of NN
  Checkpoints*.
- Wang et al. 2024, *Neural Network Diffusion* (p-diff). ← preuve d'existence.

**Compression extrême (frontière baseline à battre)**
- Dettmers et al. 2022, *LLM.int8* ; 2023, *QLoRA* ; *SpQR*.
- Egiazarian et al. 2024, *AQLM* ; Tseng et al. 2024, *QuIP#*.
- Ma et al. 2024, *The Era of 1-bit LLMs (BitNet b1.58)*.

**Sparsité / streaming (corollaires de la borne)**
- Liu et al. 2023, *Deja Vu* ; Song et al. 2023, *PowerInfer* ;
  Alizadeh et al. 2024 (Apple), *LLM in a Flash*.
- Leviathan et al. 2023 / Chen et al. 2023, *Speculative decoding*.
- Rajbhandari et al. 2021, *ZeRO-Infinity* (offload NVMe — l'art antérieur côté systèmes).

> ⚠️ Discipline : chaque référence est **re-vérifiée** (existence, claim exact,
> chiffres) au jalon §11.0 avant toute citation dans un livrable externe. On ne
> cite pas de mémoire.

---

## 4. Plan technique & architecture logicielle

Paquet nouveau, **isolé** d'`air_llm` jusqu'à intégration (Stage 3) :

```
kolmogorov/
  data/            # chargement checkpoints HF, extraction layerwise, datasets eval
  generators/
    base.py        # interface WeightGenerator: fit(W) -> θ,c ; generate(ℓ,c)->Ŵ
    lowrank.py     # G1: décomposition bas-rang + résidu structuré (baseline)
    hypernet.py    # G2: hyper-réseau / coord-MLP (INR) conditionné (ℓ, position)
    diffusion.py   # G3: diffusion/flow sur poids (borne haute de qualité)
    procedural.py  # G4: bases pseudo-aléatoires figées + coefficients appris
    hybrid.py      # G5: dictionnaire résident + product-codes par couche
  eval/
    perplexity.py  # WikiText-2/103, C4 (hold-out)
    harness.py     # wrapper lm-eval-harness (MMLU subset, déterministe)
    reconstruction.py # erreur Frobenius relative, spectre, par-couche
  systems/
    roofline_sim.py   # simulateur analytique + Monte-Carlo (§5)
    microbench.py     # mesures réelles reload vs recompute par tier
    airllm_bridge.py  # intégration dans AirLLMBaseModel.forward (Stage 3)
  registry/        # configs pré-enregistrées, seeds, manifestes d'expériences
  report/          # figures, tables, export auto
tests/             # unitaires hermétiques + tests de propriété
```

**Interface pivot** (contrat stable, testé) :
```python
class WeightGenerator(Protocol):
    def fit(self, weights: dict[str, Tensor]) -> "FitResult": ...
    def generate(self, layer_id: str, code: Tensor | None) -> Tensor: ...
    def cost_bits(self) -> BitsBreakdown:   # |θ|, Σ|c_ℓ|, b_eff
    def cost_flops(self, layer_id: str) -> int:
```

**Principe d'architecture :** le générateur est *offline* (fit payé une fois) ;
l'inférence ne fait qu'`generate` (résident) + le forward AirLLM. Aucune
modification du chemin nominal d'`air_llm` tant que Stage 3 n'a pas franchi son gate.

---

## 5. Plan de simulation (avant tout GPU)

Objectif : **décider go/no-go par profil matériel sans rien dépenser.**

`roofline_sim.py` prend `(BW_slow, C_gpu, u, bitwidth, FLOPs_gen(r), B batch,
ρ sparsité, K accept spéculatif)` et calcule `tokens/s`, `énergie/token`, et la
**région de crossover** `t_recompute < t_reload`. Sorties :
- carte de chaleur du gain vs `(BW_slow, r)` pour NVMe Gen4, RAM, HBM ;
- point d'équilibre `r*` (rang max du générateur encore rentable) par tier ;
- propagation d'incertitude (Monte-Carlo sur `u`, variance de bande passante).

**Gate de simulation :** si, même au `r*` optimiste, le gain prédit < 2× sur tous
les tiers réalistes, on **ne fait pas** C2 (mais C1 reste poursuivi comme résultat
de compression). Décision tracée, datée, signée.

---

## 6. Protocole expérimental (stages de dé-risquage)

| Stage | Modèle | But | Sortie décisive |
|------|--------|-----|-----------------|
| 0 | GPT-2 124M / Pythia-160M | rôder la méthodo C1, boucle rapide, repro totale | premières courbes `b_eff` vs `Δppl` |
| 1 | Pythia-1.4B / Llama-3.2-1B | **loi d'échelle** de la compressibilité avec la profondeur | `b_eff(N)` croît-il **sous-linéairement** ? |
| 2 | 7–8B | perplexité + downstream réels | tenue sur MMLU subset à iso-`b_eff` |
| 3 | intégration AirLLM | C2 systèmes, microbench roofline | `tokens/s` réel reload vs recompute |
| 4 | 70B | confirmation si signal | preuve à l'échelle cible |

**La loi d'échelle décisive (Stage 1) :** on mesure si `b_eff` *diminue* quand le
modèle grandit (plus de redondance partageable par un `θ` commun). **Si oui**, la
méthode se renforce exactement là où le problème est le plus dur — c'est la
*signature d'une vraie découverte*. **Si `b_eff` est plat ou croît**, l'hypothèse
est gravement affaiblie et on le déclare.

**Datasets :** WikiText-2/103, C4 (hold-out strict), MMLU (sous-ensemble fixé,
seed gelée) via `lm-eval-harness` en mode déterministe.

**Métriques :** `Δppl`, accuracy downstream, `b_eff`, erreur Frobenius relative,
FLOPs générateur, latence recompute vs reload mesurée, tokens/s, énergie/token.

**Baselines obligatoires (sinon le résultat est creux) :** fp16 reload ; 8-bit &
4-bit (l'actuel d'AirLLM) ; 2-bit AQLM/QuIP# ; bas-rang naïf. **Le générateur doit
battre la frontière de quantification à iso-perplexité**, pas seulement fp16.

**Matrice d'ablation :** familles G1–G5 × {avec/sans code par couche} ×
{partage θ inter-couches : aucun / par bloc / global} × {cible : poids bruts /
résidu après quantif}.

---

## 7. Validation — critères de succès (pré-enregistrés)

- **C1 validé** si un générateur atteint `Δppl ≤ 0.2` (WikiText-2) **et**
  `b_eff ≤ 2.0` bits/poids, soit sous la frontière AQLM/QuIP# à iso-`Δppl`.
- **Scaling validé** si `b_eff(8B) < b_eff(1.4B) < b_eff(160M)` de façon monotone
  et statistiquement significative (3 seeds, IC bootstrap 95 % disjoints).
- **C2 validé** si `t_recompute < t_reload` mesuré (pas simulé) sur ≥ 1 profil
  matériel réaliste, à iso-qualité, pour un générateur passant C1.
- **Découverte** déclarée seulement si **C1 ∧ scaling ∧ C2**.

## 8. Réfutation — critères de falsification (pré-enregistrés)

On s'engage à **publier l'échec** si :
- `H0_C1` tient : aucun générateur sous la frontière SOTA n'atteint `Δppl ≤ 0.2`
  jusqu'à 8B → l'hypothèse d'information est fausse à l'échelle utile.
- **Scaling défavorable** : `b_eff` plat ou croissant avec `N` → le partage
  inter-couches ne capture pas de structure exploitable ; thèse gravement affaiblie.
- `H0_C2` tient : `t_recompute ≥ t_reload` sur tous les profils réalistes pour les
  générateurs viables → la découverte « inférence sans I/O » est morte (mais C1
  survit comme schéma de compression de stockage).
- **Découplage qualité/reconstruction** : Frobenius bas mais `Δppl` haut →
  signal d'alarme méthodologique ; on évalue **toujours** la tâche, jamais la seule
  reconstruction.

Ces seuils sont **gelés** avant de regarder les données.

---

## 9. Qualité, tests & reproductibilité

- **Pré-enregistrement** : chaque expérience a un manifeste `registry/*.yaml`
  (modèle, seed, dataset hash, seuils) commité **avant** exécution.
- **Déterminisme** : seeds fixées, `torch.use_deterministic_algorithms(True)`,
  versions épinglées, hash des datasets.
- **Tests** : unitaires hermétiques sur `WeightGenerator` (round-trip `fit→generate`
  sur tenseurs jouets), tests de propriété (un générateur identité reconstruit à
  l'erreur machine ; `cost_bits` cohérent avec la taille réelle sérialisée),
  tests du simulateur contre cas analytiques fermés.
- **CI** : étend `.github/workflows/ci.yml` (déjà en place) — compile + pyflakes +
  la suite unitaire `kolmogorov/tests` (sans réseau/GPU).
- **Tracking** : journal d'expériences local versionné (+ W&B optionnel) ;
  figures regénérables d'un coup (`report/`).
- **Statistiques** : ≥ 3 seeds, intervalles de confiance bootstrap, tests de
  significativité sur les comparaisons de frontière ; pas de cherry-picking.
- **Revue** : tout franchissement de gate exige une *écriture de décision* (qui,
  quand, sur quelles données, verdict).

---

## 10. Registre des risques & slack (marges/contingences)

| Risque | Impact | Probabilité | Mitigation / Plan B (slack) |
|-------|--------|-------------|------------------------------|
| Frobenius bas mais tâche dégradée | élevé | moyen | évaluer la tâche d'abord ; cibler le résidu post-quantif |
| Coût d'entraînement du générateur explose | moyen | moyen | amorti offline une fois ; budgété hors chemin chaud |
| C1 ✓ mais C2 ✗ | moyen | moyen | **pivot** : « stocker les codes, pas recalculer » = gain de compression pur, publiable |
| Scaling défavorable | élevé | moyen | déclarer tôt (Stage 1) ; coût limité car petits modèles |
| Variance matérielle / repro systèmes | moyen | élevé | simulateur multi-profils + microbench répétés, IC |
| Sur-ingénierie prématurée | moyen | élevé | gates stricts ; rien à l'échelle 70B avant signal 1.4B |
| Dérive de pré-enregistrement | critique | faible | seuils gelés en commit horodaté ; revue croisée |

**Slack temporel & budget :** chaque stage porte une marge explicite (buffer 30 %)
et un **gate de sortie** ; un stage qui échoue son gate **arrête la branche**
correspondante sans contaminer les autres (C1 et C2 sont découplés exprès).

---

## 11. Plan d'exécution (jalons & gates)

- **§11.0 — Revue de littérature & vérification des sources** (gate : biblio
  confirmée, frontière baseline chiffrée). *Aucune dépense GPU.*
- **§11.1 — Simulateur roofline** (§5) → gate go/no-go C2 par profil matériel.
- **§11.2 — Stage 0** (GPT-2/Pythia-160M) : méthodo C1, premières courbes.
- **§11.3 — Stage 1** (≤1.4B) : **loi d'échelle** → gate décisif principal.
- **§11.4 — Stage 2** (7–8B) : perplexité + downstream.
- **§11.5 — Stage 3** : intégration AirLLM + microbench C2.
- **§11.6 — Stage 4** (70B) : confirmation (conditionnel aux gates précédents).

Chaque jalon : livrable + écriture de décision + figures regénérables. On **ne
saute pas** un gate.

## 12. Livrables & seuils de publication

- **Papier A (C1)** : « Effective Description Length of LLM Weights » — déclenché si
  C1 + scaling, **indépendamment de C2**.
- **Papier B (C2/systèmes)** : « Recompute-over-Reload Inference » — si C2.
- **Papier C (théorie)** : la borne §2.2 — travail séparé, plus long horizon.
- **Rapport négatif** : si réfutation, publication de l'échec avec les courbes
  (un résultat nul rigoureux sur l'information intrinsèque est précieux).

---

### Annexe — Pourquoi commencer par Stage 0/1 et pas par le rêve à 405B
La valeur d'information par dollar est maximale en bas de l'échelle : la **loi
d'échelle** (Stage 1) prédit le sort de toute la thèse pour un coût négligeable. Un
scientifique sérieux paie d'abord l'expérience qui peut **tuer** son idée le moins
cher possible. C'est tout l'inverse de « tester directement sur 405B ».
