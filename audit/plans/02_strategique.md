# Plan 02 — Stratégique : quel verrou unique on attaque

## Le verrou unique : le MUR MÉMOIRE, réexprimé en MUR DE BANDE PASSANTE

L'objectif (1T params, ≤8 Go RAM, offline, **sans perte**) rend la RAM insuffisante par 2 à 3
ordres de grandeur : 1T params en bf16 = **2 To** de poids. Aucune machine cible ne tient ça en
RAM. AirLLM résout *déjà* le sous-problème « ça ne tient pas en RAM » par le streaming par
couche (`airllm_base.py:450-600`) : le pic RAM suit la plus grosse couche, pas le modèle.

**Donc le verrou n'est plus la capacité RAM — il devient la BANDE PASSANTE de lecture.** En
streaming, le coût incompressible par token est :

```
latence/token ≥ (octets à lire pour ce token) / (débit disque réel)
```

C'est l'unique grandeur qui décide du succès. Tout le reste (overhead dequant, fragmentation,
overhead Python) est secondaire d'un ou deux ordres de grandeur (cf. Plan 06, Plan 02-§physique).

## Pourquoi AirLLM (layer streaming) est le bon point de départ — justifié vs alternatives

| Approche | Tient 1T en ≤8 Go ? | Sans perte ? | Verdict |
|---|---|---|---|
| **AirLLM (streaming par couche)** | Oui (pic = 1 couche) `airllm_base.py:597-600` | Oui par construction (réordonnance I/O, ne touche pas le calcul) | **Base retenue** |
| Quantization agressive (GGUF Q4/Q2, AWQ…) | Réduit mais 1T-Q4 ≈ 250–500 Go, ne tient toujours pas en RAM, et **AVEC PERTE** | Non | Écartée (viole l'invariant) |
| Offload disque générique (`accelerate` device_map=disk) | Oui mais pas de prefetch fin, pas de découpe par couche optimisée | Oui | Inférieur : AirLLM fait déjà mieux (prefetch overlap `airllm_base.py:441-487`) |
| Tensor/pipeline parallelism multi-nœuds | Hors cible (1 Pi) | Oui | Hors périmètre matériel |
| Distillation / petit modèle | — | **Non** (change le modèle) | Écartée par définition |

AirLLM est la **seule** base qui satisfait déjà « tient en RAM » ET « sans perte par
construction ». On part donc d'elle et on attaque son unique faiblesse résiduelle : la
bande passante.

## La thèse stratégique (le pivot)

> Le streaming dense lit **tous** les poids à chaque token. Pour 1T sur Pi, c'est physiquement
> rédhibitoire (cf. Plan « Le Mur », chiffré : des heures par token sur SD). **Le seul gain
> sans perte capable de casser ce mur est de NE PAS LIRE la majorité des poids — sans changer
> la sortie.** Or il existe un cas où des poids non lus ont une contribution
> *mathématiquement nulle* : les **experts non routés d'un modèle MoE**.

C'est le cœur stratégique :
- Un modèle **≥1T MoE** (ex. Kimi K2 ≈ 1,04 T params, ~32 B actifs/token ; DeepSeek-V3 671 B,
  37 B actifs — `[HYPOTHÈSE chiffres publics, à re-vérifier au Sprint 1]`) n'active que `k` de
  `N` experts par token. Les `N−k` autres sont multipliés par un poids de gating **exactement
  nul** → leur lecture est inutile **et leur omission ne change pas un seul logit**.
- Le code actuel **gâche** ce gisement : `AirLLMMixtral` (`airllm_mixtral.py:8-14`) hérite du
  streaming dense → il lit les 8 experts par couche alors que 2 suffisent.

## Verrous secondaires (attaqués seulement après le principal)

1. **Recompute quadratique** : KV-cache désactivé (`airllm_base.py:410-412`) → restaurer un
   KV-cache sans perte (Plan 03/09).
2. **Reboot par forward** : `del/init_model` à chaque token (`airllm_base.py:421-423`) →
   overhead fixe, pas un mur physique mais une dette (Plan 03).
3. **Pas de mmap** : `load_file` charge tout (`safetensor_model_persister.py:36-37`) → mmap +
   page cache = réutilisation inter-tokens sans perte (Plan 09).

---

## Section transversale A — Justification

Le choix « attaquer la bande passante via la sparsité MoE exacte » est justifié par
élimination : c'est la **seule** famille de techniques qui (a) réduit les octets/token d'un
ordre de grandeur, ET (b) est sans perte par construction. La quantization réduit aussi les
octets mais viole l'invariant. Le mmap/cache réduit les lectures *répétées* mais pas le set
unique d'un modèle dense. Donc MoE-exact est le seul levier « ordre de grandeur + sans perte ».

## Section transversale B — Preuve sans perte (de la stratégie)

La stratégie elle-même est un document ; sa preuve sans perte est déléguée aux plans qui
implémentent (Plan 04 invariants, Plan 09 démonstration MoE). Principe : *la sortie d'une
couche MoE est `Σ_{e∈top-k} g_e · Expert_e(x)`* ; les experts hors top-k ont `g_e = 0` par
définition du routage top-k. Ne pas charger un expert de poids `g_e=0` laisse la somme
identique au bit près (cf. Plan 09 pour la démonstration formelle et l'oracle).

## Section transversale C — Tests

- **Test de routage** : sur un modèle MoE de référence, instrumenter le gating et vérifier que
  pour chaque token/chaque couche, l'ensemble des experts à poids non-nul == top-k. Oracle :
  comparer les logits « lecture de tous les experts » vs « lecture des seuls top-k » → doivent
  être **bit-identiques** (même ordre de sommation préservé, cf. Plan 04).

## Section transversale D — Stress tests

- **Pire cas de routage** : prompt adverse maximisant la diversité d'experts routés sur la
  fenêtre → mesurer le set d'experts uniques à lire et donc la borne haute d'octets/token.
- **Dérive thermique** : 72 h de génération continue, vérifier que le throttle SoC n'altère
  pas le *résultat* (il n'altère que la *vitesse* — à prouver).

## Section transversale E — Mitigation

- **Risque fatal** : et si la cible n'est pas MoE mais un modèle 1T **dense** ? Alors le mur de
  bande passante est intrinsèque (Plan 06) et la stratégie échoue. **Parade** : restreindre
  explicitement l'objectif « 1T sans perte sur Pi » aux **architectures MoE** ; le déclarer
  comme hypothèse de cadrage au Sprint 1. **Plan B** : pour un 1T dense, prouver le mur et
  livrer le verdict honnête « impossible en interactif sans perte », avec la borne chiffrée
  (la mission autorise explicitement ce verdict).
