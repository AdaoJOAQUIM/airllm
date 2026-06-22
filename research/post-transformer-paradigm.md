# Vers un paradigme de calcul cognitif post-Transformer

> Programme de recherche exploratoire.
> Document de travail — version 1 (2026-06-22).
> Contexte : ce document est rédigé dans le dépôt **AirLLM**, dont la raison d'être
> est de contourner le coût mémoire de l'inférence des grands Transformers. Le
> programme ci-dessous part précisément de là où AirLLM s'arrête : au lieu
> d'*optimiser* le chargement d'un modèle géant, on interroge l'hypothèse que la
> connaissance *doive* prendre la forme d'un fichier de poids géant.

## Avertissement méthodologique (à lire d'abord)

Ce document n'est pas un manifeste. C'est une tentative honnête de séparer trois
choses qu'on confond souvent dans ce genre d'exercice :

1. **Ce qui est réel et déjà démontré** (état de l'art).
2. **Ce qui est plausible mais non démontré** (hypothèses architecturales testables).
3. **Ce qui est de la spéculation rhétorique** (« mémoire active », « compilation
   cognitive », « compression sémantique extrême »…) — des mots qui *sonnent*
   comme des découvertes mais qui ne sont rien tant qu'on ne leur a pas associé
   une définition opérationnelle et une métrique.

Conformément aux contraintes du programme :

- On ne confond pas complexité et innovation. Une idée n'a de valeur que si elle
  *retire* une hypothèse, pas si elle *ajoute* une couche.
- On n'adopte rien parce que c'est populaire (ni Transformer, ni Mamba, ni « agents »).
- On n'accepte aucune hypothèse parce qu'elle domine. En particulier on questionne :
  token, paramètre, poids statique, modèle-fichier, mémoire passive, calcul séquentiel.

La conclusion anticipée, pour être franc tout de suite : **il n'existe pas
aujourd'hui de preuve qu'une architecture unique batte le Transformer sur tous les
axes.** En revanche, il existe plusieurs endroits *identifiables et quantifiables*
où le paradigme actuel gaspille, et au moins deux directions architecturales qui
attaquent ce gaspillage avec un avantage théorique défendable. Le programme se
termine donc sur **un seul prototype** réellement à construire, choisi parce qu'il
est falsifiable avec les ressources d'un dépôt comme celui-ci.

---

## Phase 1 — Cartographie de l'état de l'art et de ses limites

### 1.1 Ce que le Transformer a réellement résolu

Le Transformer (2017) n'a pas « inventé l'attention » ; il a rendu l'apprentissage
de séquences **massivement parallélisable** sur du matériel SIMD. C'est sa vraie
innovation, souvent mal comprise : son succès est d'abord un succès *de
co-conception matériel/algorithme* (le GPU aime les multiplications de matrices
denses), pas un succès de plausibilité cognitive. Toute proposition « post-Transformer »
qui ignore cette contrainte matérielle est morte-née. C'est la première leçon.

### 1.2 Les hypothèses fondamentales et où elles cassent

| Hypothèse dominante | Ce qu'elle suppose | Où elle gaspille (mesurable) |
|---|---|---|
| **Le token est l'unité d'information** | Le sens se découpe en sous-mots discrets fixés avant l'entraînement. | La tokenisation détruit de l'information (arithmétique, code, langues peu dotées) et impose un coût quadratique sur une granularité arbitraire. Mesure : *bits/octet* atteignables vs effectivement utilisés. |
| **Le paramètre est l'unité de connaissance** | Savoir = poids figé dans une matrice. | Redondance massive : un modèle 70B est compressible 4–8× (quantization, distillation) *sans perte mesurable*. Donc ≥ 75 % des bits ne portent pas de connaissance. Mesure directe. |
| **Les poids sont statiques** | À l'inférence, le modèle ne change pas. | Tout fait nouveau exige un ré-entraînement ou un contexte gonflé. Le savoir et le calcul sont divorcés. |
| **Le modèle est un fichier** | La connaissance est un blob monolithique chargé en bloc. | C'est *exactement* le problème qu'AirLLM contourne : on streame 405B de poids pour produire quelques tokens. Le ratio bits-lus / bits-utiles-pour-cette-requête est catastrophique. |
| **La mémoire est un stockage passif** | KV-cache = tampon qu'on relit linéairement. | Le contexte ne se réorganise pas, ne s'indexe pas, ne s'oublie pas sélectivement. Coût mémoire O(longueur). |
| **Le calcul est une exécution séquentielle de couches** | Profondeur fixe, même effort pour « 2+2 » et pour une preuve. | Pas d'allocation de calcul dirigée par la difficulté. Mesure : FLOPs dépensés vs FLOPs nécessaires (énorme pour les tâches faciles). |

### 1.3 L'état de l'art des alternatives déjà tentées

Honnêteté obligatoire : presque toutes les « primitives » demandées par le
programme existent déjà sous une forme partielle. Les ignorer serait réinventer en
moins bien.

- **Coût quadratique de l'attention** → attaqué par : attention linéaire,
  Performer, **State-Space Models (S4, Mamba/Mamba-2)**, **RWKV**, **RetNet**,
  Hyena. Acquis : récurrence à coût O(1) par token en inférence, état de taille
  constante. Limite : leur état compressé *perd* en rappel associatif exact (les
  têtes d'attention restent meilleures pour « copier » un fait vu plus tôt). C'est
  un compromis information-théorique réel, pas un défaut d'ingénierie.
- **Paramètre comme connaissance** → attaqué par : **MoE** (paramètres
  conditionnels), **RAG / mémoire externe**, kNN-LM, **Retrieval-augmented**
  pré-entraînement. Acquis : on peut sortir le savoir factuel des poids. Limite :
  le *raisonnement* reste dans les poids ; la frontière savoir/calcul reste floue.
- **Calcul séquentiel à profondeur fixe** → attaqué par : **adaptive computation
  time**, **early-exit**, Universal Transformers, **chaîne de pensée**, modèles à
  « budget de raisonnement » (calcul au moment du test). Acquis : preuve empirique
  que dépenser plus de calcul au test bat dépenser plus de paramètres. C'est, à mon
  avis, le résultat le plus important et le plus sous-estimé de la période récente.
- **Mémoire active** → tentatives historiques : **Neural Turing Machines**,
  Differentiable Neural Computers, **réseaux de Hopfield modernes** (l'attention
  *est* un Hopfield à un coup). Limite : instables à entraîner, jamais passés à
  l'échelle. L'échec n'était pas conceptuel mais d'optimisation/matériel.
- **Token comme unité** → attaqué par : modèles **byte-level** (ByT5),
  **MegaByte**, patching dynamique (**BLT, Byte Latent Transformer**). Acquis : on
  peut apprendre la granularité au lieu de la fixer.

**Conclusion de la Phase 1.** Les six « nouvelles primitives » demandées ne sont
pas vierges. Le vrai espace de découverte n'est pas « inventer la mémoire active »
mais **trouver la combinaison qui retire une hypothèse coûteuse sans réintroduire
le coût ailleurs** — car c'est ce qui a tué chaque tentative isolée.

---

## Phase 2 — Génération d'architectures radicales

Quatre candidats. Pour chacun : l'hypothèse retirée, le mécanisme, et l'objection
fatale (toujours énoncer l'objection fatale en premier — un candidat sans objection
fatale identifiée est un candidat mal compris).

### A. **Mémoire associative active à oubli sélectif** (« KV-cache qui se réorganise »)
- *Hypothèse retirée* : la mémoire est un tampon passif O(longueur).
- *Mécanisme* : le contexte n'est plus une liste de paires KV append-only mais une
  **mémoire de taille bornée qui se réécrit** : fusion d'entrées redondantes,
  éviction par utilité prédite, ré-indexation. Inspiré des SSM (état borné) mais
  avec écriture *associative et éditable* plutôt que récurrence linéaire fixe.
- *Objection fatale* : décider quoi oublier est un problème de crédit non
  différentiable de bout en bout ; les NTM ont échoué exactement là. Réfutable
  seulement si on trouve un signal d'apprentissage stable (ex : oubli supervisé par
  la perte de prédiction future, pas par RL).

### B. **Connaissance comme programme, pas comme poids** (compilation cognitive)
- *Hypothèse retirée* : paramètre = connaissance ; modèle = fichier.
- *Mécanisme* : séparer un **noyau de raisonnement** petit et figé d'une **base de
  connaissances** explicite, structurée, éditable (graphe + index vectoriel), où le
  noyau *compile* à la volée un mini-programme de récupération/transformation. La
  « compilation cognitive » devient une opération définie : `intention → plan de
  récupération + plan de calcul`, pas un slogan.
- *Objection fatale* : c'est RAG + planification. La nouveauté réelle se réduit à :
  *le noyau peut-il être assez petit pour que 99 % des bits vivent dans une mémoire
  éditable sans ré-entraînement, tout en gardant le raisonnement ?* Si non, on a
  juste déplacé le monolithe.

### C. **Calcul dirigé par l'intention à budget variable** (profondeur ≠ constante)
- *Hypothèse retirée* : calcul = pile de couches de profondeur fixe.
- *Mécanisme* : un contrôleur alloue un budget de FLOPs par requête *avant* et
  *pendant* le calcul, en fonction d'une estimation de difficulté ; les cas faciles
  sortent tôt, les cas durs bouclent. Unifie early-exit + chaîne de pensée + ACT
  sous un objectif explicite : minimiser FLOPs sous contrainte de qualité.
- *Objection fatale* : l'estimation de difficulté est elle-même un calcul, et les
  modèles sont notoirement mal calibrés sur leur propre incertitude. Réfutable par
  une métrique : *FLOPs économisés à iso-qualité* sur un benchmark à difficulté
  hétérogène.

### D. **Représentation continue sans tokens** (patching latent dynamique)
- *Hypothèse retirée* : le token discret pré-figé est l'unité.
- *Mécanisme* : entrée au niveau octet, regroupée dynamiquement en patchs latents
  de longueur variable selon l'entropie locale (peu de calcul sur le prévisible,
  beaucoup sur le surprenant). C'est la ligne BLT/MegaByte poussée à sa logique.
- *Objection fatale* : déjà partiellement démontré ; le risque est qu'on gagne en
  équité linguistique et en robustesse sans gagner en *capacité cognitive*. Avantage
  réel surtout sur compression et langues/codes mal tokenisés.

---

## Phase 3 — Évaluation (notes 1–5, justifiées, pas décoratives)

| Critère | A. Mémoire active | B. Connaissance-programme | C. Calcul à budget | D. Sans-token |
|---|---|---|---|---|
| **Nouveauté** | 4 | 3 | 3 | 2 |
| **Faisabilité (ressources modestes)** | 2 | 3 | **4** | 3 |
| **Avantage théorique** | 4 | 4 | **4** | 3 |
| **Coût computationnel (faible = mieux)** | 3 | 3 | **5** | 3 |
| **Potentiel de rupture** | 5 | 4 | 3 | 2 |
| **Falsifiabilité (peut-on le tuer vite ?)** | 2 | 3 | **5** | 4 |

Lecture des notes :

- **A** a le plus haut plafond mais la plus faible falsifiabilité à court terme :
  c'est un *programme de thèse*, pas un prototype de dépôt.
- **B** est le plus prometteur stratégiquement (il attaque directement le
  « modèle-fichier » qui est la douleur d'AirLLM) mais demande une infra de
  connaissances et un protocole d'évaluation lourd pour être convaincant.
- **C** est le seul qui soit *à la fois* à fort avantage théorique, peu coûteux, et
  **immédiatement falsifiable** : on peut mesurer, sur du matériel ordinaire, si
  l'allocation de calcul dirigée par la difficulté réduit les FLOPs à qualité
  constante. Un résultat négatif est aussi informatif qu'un positif.
- **D** est le moins risqué mais le moins susceptible de constituer une *rupture*
  cognitive ; c'est une amélioration de représentation, déjà en cours dans la
  littérature.

**Décision (contrainte « ne construire que ce qui démontre un avantage mesurable »)** :
le prototype de la Phase 4 est **C — calcul dirigé par l'intention à budget variable**,
parce que c'est le seul dont on peut établir l'avantage *ou la réfutation* avec une
expérience honnête et reproductible, sans cluster. B est désigné comme **direction
de recherche n°2** à instruire ensuite, car il est le plus aligné avec la mission
de fond du dépôt.

---

## Phase 4 — Prototype falsifiable : « budget de calcul adaptatif »

### 4.1 Hypothèse précise et réfutable

> Sur un mélange de requêtes à difficulté hétérogène, un routeur qui alloue le
> nombre de passes de calcul (profondeur effective ou longueur de raisonnement)
> en fonction d'une estimation de difficulté atteint la même qualité qu'un calcul
> à budget fixe maximal, **en consommant strictement moins de FLOPs en moyenne**.
> Si l'économie de FLOPs à iso-qualité n'est pas significative, l'hypothèse est
> fausse pour ce régime.

### 4.2 Protocole minimal (réalisable dans ce dépôt)

1. Prendre un modèle décodeur déjà supporté par AirLLM (petit, ex. 7B/quantisé) —
   on réutilise l'infra de chargement existante.
2. Construire un jeu d'évaluation **mixte** : tâches triviales (lookup factuel,
   format) + tâches difficiles (multi-étapes arithmétiques/logiques).
3. Trois politiques comparées à iso-modèle :
   - `fixed-low` : budget de raisonnement minimal.
   - `fixed-high` : budget maximal (oracle de qualité).
   - `adaptive` : un estimateur de difficulté **bon marché** (entropie/marge des
     premiers tokens, ou un classifieur léger) choisit le budget par requête.
4. Métriques : qualité (exactitude), **FLOPs/tokens générés moyens**, et la courbe
   **qualité vs coût** (Pareto). Le test décisif = `adaptive` domine-t-il la
   frontière de Pareto entre `fixed-low` et `fixed-high` ?

### 4.3 Ce que ce dépôt fournit déjà / ce qui manque

- *Déjà là* : chargement mémoire-frugal des poids (`air_llm/`), donc le coût
  expérimental est tenable même sur petit GPU.
- *À écrire* : (i) l'estimateur de difficulté, (ii) la boucle de génération à
  budget variable, (iii) le harnais de mesure FLOPs/qualité. Aucun de ces trois
  morceaux ne requiert de ré-entraînement — c'est volontaire : on veut un test
  *cheap* et *honnête*, pas une démonstration coûteuse qui cache ses hypothèses.

Un squelette de harnais d'expérience accompagne ce document :
[`research/experiments/adaptive_compute_probe.py`](experiments/adaptive_compute_probe.py).
Il est **délibérément non concluant** : c'est un cadre de mesure, pas un résultat.
Présenter un chiffre non mesuré comme un acquis violerait la règle « ne jamais
confondre complexité et innovation » — et la règle, plus simple, de ne pas mentir.

---

## Phase 5 — Esquisse d'une architecture complète (honnête sur son statut)

Si — et seulement si — le prototype C valide l'allocation adaptative, l'architecture
cible n'est pas « un nouveau bloc qui remplace l'attention ». C'est une
**recomposition** qui retire trois hypothèses à la fois :

```
                ┌─────────────────────────────────────────────┐
   intention ──▶│  PLANIFICATEUR (noyau figé, petit)           │  ← retire « calcul séquentiel fixe »
                │   estime difficulté → alloue budget          │
                └───────────────┬─────────────────────────────┘
                                │ plan de récupération + plan de calcul
              ┌─────────────────▼───────────────────┐
              │  MÉMOIRE ÉDITABLE (savoir explicite) │  ← retire « paramètre = connaissance », « modèle = fichier »
              │   graphe + index ; lecture ET écriture│
              └─────────────────┬───────────────────┘
                                │ contexte pertinent borné
              ┌─────────────────▼───────────────────┐
              │  EXÉCUTEUR à profondeur variable     │  ← retire « profondeur constante »
              │   boucle jusqu'au budget alloué      │
              └─────────────────────────────────────┘
```

Pourquoi cette forme et pas une autre :

- Elle **localise** chaque hypothèse retirée dans un module distinct, donc chaque
  retrait est testable isolément (falsifiabilité modulaire).
- Elle respecte la contrainte matérielle de la Phase 1 : le noyau reste dense et
  parallélisable ; l'innovation est dans le *contrôle* (combien calculer, quoi lire),
  pas dans un opérateur exotique mal adapté au GPU.
- Elle réduit les quatre gaspillages ciblés par le programme :
  - *mémoire* : savoir hors des poids, contexte borné et réécrit ;
  - *calcul* : effort proportionnel à la difficulté ;
  - *information* : granularité d'entrée apprenable (option D branchable ici) ;
  - *contexte* : mémoire indexée plutôt que relue linéairement.

Ce que cette esquisse **n'est pas** : une preuve. C'est une hypothèse d'architecture
dont chaque arête est un pari réfutable. La discipline du programme veut qu'on ne
construise le module suivant qu'après que le précédent ait démontré un avantage
*mesuré*. L'ordre recommandé : **C** (prouvable maintenant) → **B** (mémoire
éditable) → **A** (réécriture/oubli, le plus dur) → intégration.

---

## Synthèse pour décideur pressé

1. Le Transformer gagne d'abord parce qu'il épouse le GPU. Toute alternative doit
   respecter cette contrainte ou perdre. (Phase 1)
2. Les « nouvelles primitives » demandées existent déjà en pièces détachées (SSM,
   MoE, RAG, calcul-au-test, byte-latent). La découverte possible est dans leur
   *recomposition retirant des hypothèses*, pas dans un opérateur miracle. (Phases 1–2)
3. Le gaspillage le plus net et le plus quantifiable est le **calcul uniforme** :
   même effort pour le facile et le difficile. C'est l'angle d'attaque le plus
   falsifiable à coût réduit. (Phase 3)
4. **À construire maintenant** : un test honnête du calcul adaptatif à budget
   variable (Phase 4) ; squelette fourni, résultat non encore mesuré.
5. **Architecture cible** : planificateur figé + mémoire éditable + exécuteur à
   profondeur variable — proposée comme hypothèse modulaire, pas comme acquis. (Phase 5)

> Règle finale du programme, et de ce document : un chiffre non mesuré n'est pas un
> résultat, et une couche ajoutée n'est pas une innovation. Tant que le prototype C
> n'a pas tourné, la bonne réponse à « avons-nous trouvé le paradigme post-Transformer ? »
> reste : *pas encore — mais voici exactement l'expérience qui le dirait*.
