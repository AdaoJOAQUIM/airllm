# Cognitive Efficiency Engine (CEE)

> Programme de recherche — Phase 2 (Lab Phase 1).
> Document de travail, 2026-06-22. Suite de [`post-transformer-paradigm.md`](post-transformer-paradigm.md).

## Pourquoi ce document existe

La Phase 1 a établi une discipline : *un chiffre non mesuré n'est pas un résultat,
une couche ajoutée n'est pas une innovation.* La Phase 2 transforme cette discipline
en **instrument**.

L'objectif de fond n'a pas changé : avant de parler d'« AirLLM 2050 » ou de
trillions de paramètres, il faut répondre à une question préalable et beaucoup plus
difficile :

> **Comment obtenir plus d'intelligence par unité de calcul ?**

Le piège à éviter est connu et a été nommé en Phase 1 : on « améliore » un système
en lui greffant une base vectorielle, des agents, une mémoire, une couche
d'abstraction — et on déclare la victoire sans jamais vérifier que le gain net est
positif. Or **chacune de ces greffes coûte des tokens.** Une mémoire qu'on relit,
un planificateur qui délibère, un générateur qui produit du code à valider : tout
cela consomme du calcul. La seule question qui compte est :

> Pour chaque capacité ajoutée, le système produit-il **plus de travail cognitif
> utile par token dépensé**, oui ou non ?

Le CEE est le moteur qui répond à cette question par la mesure, pas par la
conviction. C'est le « moteur multiplicateur » : il ne multiplie rien tout seul, il
**mesure si quelque chose multiplie**, et donne donc le droit (ou non) de passer à
l'échelle.

## Le glissement de définition qu'il faut faire

Définition naïve, à rejeter :

> « 1 token = 1 milliard d'actions »

C'est un slogan, pas une métrique. Il confond le *fantasme* d'efficacité avec sa
*mesure*. La chaîne de valeur réelle, que ce document rend opérationnelle, est :

```
   tokens consommés
        │
        ▼
   décisions utiles produites        (combien de la "pensée" était sur la cible)
        │
        ▼
   actions automatisées réussies     (combien d'effets externes ont abouti)
        │
        ▼
   temps humain économisé            (métrique molle — à traiter avec prudence)
        │
        ▼
   complexité gérée                  (taille du problème réellement maîtrisé)
```

Chaque flèche est un **taux de conversion mesurable**. Le CEE mesure chaque flèche
séparément, parce qu'un système peut exceller sur l'une et gaspiller sur une autre
(p. ex. beaucoup de décisions utiles mais des actions qui échouent — typique d'un
bon raisonneur mal outillé).

## Les métriques (définitions opérationnelles)

Toutes les quantités ci-dessous sont mesurables à partir d'un *journal de run*
(voir [`benchmarks/`](benchmarks/)). Aucune ne repose sur une intuition non
chiffrable.

| Symbole | Nom | Définition opérationnelle |
|---|---|---|
| `T` | Coût en calcul | `tokens_in + tokens_out` du run complet (toutes étapes, tous sous-agents inclus). On compte en **kilotokens** `T/1000`. C'est l'unité de coût parce qu'elle est la plus directement liée au compute facturé et qu'elle inclut *toute* la surcharge des greffes. |
| `D_u` | Décisions utiles | Nombre de sous-objectifs/décisions corrects **et conservés** (non annulés, non redondants). Jugé selon une grille (voir threats to validity). |
| `D_t` | Décisions totales | Toutes les décisions/sous-objectifs tentés. |
| `A_s / A_a` | Conversion d'action | Actions externes réussies / tentées (un fichier écrit qui passe les tests, une commande qui avance la tâche). |
| `Q` | Qualité de solution | `[0,1]` : 0 = échec, 1 = tâche résolue selon le critère **pré-enregistré** de la tâche. Partiel autorisé. |
| `K` | Complexité de la tâche | Score `[0,1]` fixé **dans la spec de la tâche, avant le run** (nb de contraintes/fichiers/dépendances à coordonner). Jamais auto-attribué par le système jugé. |
| `H_b` | Référence humaine | Estimation pré-enregistrée du temps humain pour résoudre la tâche. **Métrique la plus molle** — secondaire. |
| `H_o` | Temps de supervision | Temps humain réellement dépensé à superviser/corriger le run. |

### Métrique principale : le Quotient d'Efficacité Cognitive (CEQ)

```
            K · Q            complexité réellement maîtrisée, à qualité prouvée
  CEQ  =  ─────────   =   ───────────────────────────────────────────────────
           T / 1000                       par kilotoken
```

Unité : **points-de-complexité par kilotoken.** Plus c'est haut, plus le système
extrait d'intelligence utile de chaque unité de calcul. C'est *la* quantité que le
programme cherche à augmenter — avant tout ajout de paramètres.

### Métriques de la chaîne (diagnostic, pas score unique)

- **Rendement décisionnel** : `D_u / D_t` — quelle fraction de la pensée a porté.
- **Frugalité** : `D_u / (T/1000)` — décisions utiles par kilotoken.
- **Conversion d'action** : `A_s / A_a` — le raisonnement se transforme-t-il en effet ?
- **Levier humain** (mou) : `(H_b · Q − H_o) / (T/1000)` — secondes humaines nettes
  économisées par kilotoken. Toujours rapporté avec un avertissement.

### La seule chose qui autorise à parler de « multiplicateur »

```
                  CEQ(configuration)
  Multiplicateur = ───────────────────
                   CEQ(baseline)
```

Un multiplicateur > 1 *mesuré sur une suite pré-enregistrée* est la **seule** preuve
recevable qu'une capacité ajoute de l'intelligence par unité de calcul. Un
multiplicateur ≤ 1 signifie : la greffe coûte plus qu'elle ne rapporte → on la
retire, quelle que soit sa popularité.

## Le protocole : une étude d'ablation, pas une démo

C'est le cœur méthodologique. On ne « montre pas que le système marche ». On
**isole la contribution marginale de chaque capacité** par ablation, exactement les
configurations que tu as listées :

| Config | Pile | Question qu'elle teste |
|---|---|---|
| `C0` baseline | Claude Code seul | Référence. CEQ de base. |
| `C1` | + mémoire éditable | La mémoire rapporte-t-elle plus de `D_u` qu'elle ne coûte en `T` ? |
| `C2` | + planificateur | La planification réduit-elle les décisions inutiles (`D_t − D_u`) assez pour payer son coût ? |
| `C3` | + génération automatique | Le code généré convertit-il (`A_s/A_a`) sans exploser `T` en allers-retours de validation ? |
| `C4` | + calcul adaptatif | Le budget variable (cf. Phase 4 du doc 1) baisse-t-il `T` à `Q` constant ? |

Deux lectures :

1. **Marginale** : `ΔCEQ` de chaque capacité ajoutée *seule* sur `C0`. Identifie ce
   qui multiplie isolément.
2. **Cumulée** : empiler `C0 → C0+C1 → … ` et regarder où le CEQ **cesse de monter
   ou redescend**. Les capacités interagissent : une mémoire sans planificateur peut
   nuire (plus de contexte à relire sans usage dirigé). Le point où le cumul décroît
   est une information de premier ordre.

Hypothèse de travail (réfutable) : *le calcul adaptatif (`C4`) et le planificateur
(`C2`) ont un multiplicateur > 1 ; la mémoire brute (`C1`) a un multiplicateur ≤ 1
tant qu'elle n'est pas dirigée par le planificateur.* Si les données disent le
contraire, l'hypothèse tombe — c'est le but.

## Les trois axes de benchmark

Découpés comme demandé, parce qu'une seule moyenne masque les compromis :

- [`benchmarks/token_efficiency/`](benchmarks/token_efficiency/) — `D_u` et `Q` par
  kilotoken. Cible le gaspillage de **tokens**.
- [`benchmarks/task_completion/`](benchmarks/task_completion/) — `A_s/A_a` et `Q` sur
  tâches multi-étapes réelles. Cible le gaspillage de **contexte/coordination**.
- [`benchmarks/reasoning_cost/`](benchmarks/reasoning_cost/) — courbe `Q` vs `T` sur
  difficulté hétérogène. Cible le gaspillage de **calcul** (le cœur du doc 1).

Le moteur de score commun est [`benchmarks/cee.py`](benchmarks/cee.py) : il
consomme des **journaux de run** (JSONL) et produit le rapport CEQ + ablation. Il
ne *génère pas* les runs — il les *mesure*. C'est volontaire : la mesure doit être
indépendante du système mesuré.

## Threats to validity (à ne pas enterrer en annexe)

1. **Subjectivité de `D_u`.** « Décision utile » est jugé. Mitigation : grille
   écrite, jugement à l'aveugle de la configuration, double codage sur un
   sous-échantillon, report d'un accord inter-juges.
2. **`H_b` (temps humain) est une estimation.** C'est la métrique la plus
   manipulable. Elle reste **secondaire** ; aucune conclusion ne repose sur elle
   seule.
3. **Biais de sélection des tâches.** Une suite choisie pour flatter une capacité
   donne un faux multiplicateur. Mitigation : suite **pré-enregistrée** avant de
   voir les résultats, couvrant explicitement le facile *et* le difficile.
4. **Surapprentissage du benchmark.** Mitigation : tâches tenues secrètes / tournées,
   et on rapporte la *forme* de la courbe `Q` vs `T`, pas un seul score.
5. **Coût de mesure compté.** `T` inclut TOUTE la surcharge (mémoire relue,
   délibération du planificateur, validations). Pas de tricherie consistant à ne
   compter que les tokens « finaux ».

## Comment ça nourrit l'objectif initial

L'enchaînement reste celui que tu as posé, mais rendu scientifique :

```
  CEE (ce document)                  ← mesure le multiplicateur, honnêtement
        │  identifie quelle capacité a un multiplicateur > 1
        ▼
  Moteur multiplicateur              ← on ne construit QUE les capacités prouvées > 1
        │  l'efficacité par token, pas le nombre de paramètres
        ▼
  AirLLM 2050                        ← passage à l'échelle d'un système déjà efficace
```

La rupture visée n'est pas « AirLLM avec plus d'optimisations » ni « 1 trillion de
paramètres ». C'est : **un système qui sait combien d'intelligence dépenser pour une
tâche donnée**, et dont on a *mesuré* qu'il en dépense moins pour le même résultat.
Le CEE est la condition d'entrée : on n'a pas le droit de scaler ce dont on n'a pas
prouvé l'efficacité.

> Statut honnête de ce document : c'est l'**instrument de mesure et son protocole**,
> plus un harnais de scoring exécutable. Les multiplicateurs ne sont pas encore
> mesurés sur de vrais runs — le harnais tourne sur un journal **synthétique**
> clairement étiqueté, dont les chiffres ne valent que comme test de l'arithmétique.
> Mesurer les vrais runs est l'étape suivante, et elle exige d'instrumenter de
> véritables sessions (voir `benchmarks/README.md`).
