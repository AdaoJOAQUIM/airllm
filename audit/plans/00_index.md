# Phase 0 — Les 9 plans (livrables AVANT toute exécution)

> Mission : inférence ≥1T params, locale/offline, **sans perte**, sur Raspberry Pi (≤8 Go) ou
> ordinateur faible équivalent. Base : fork AirLLM (layer streaming).
> Statut : **Phase 0 livrée — en attente de validation utilisateur avant Phases 1-4.**

## Invariant maître
Parité de sortie stricte vs modèle de référence pleine précision (logits identiques à une borne
d'arrondi flottant explicite et prouvée près). Toute technique est **sans perte par
construction** ou **vérifiée par oracle** — sinon écartée.

## Les 9 plans
1. [Technique](01_technique.md) — cartographie réelle, flux mémoire, format des poids.
2. [Stratégique](02_strategique.md) — verrou unique = mur de bande passante ; pourquoi AirLLM.
3. [Amélioration](03_amelioration.md) — backlog priorisé sans perte (A1…A8).
4. [Qualité](04_qualite.md) — invariants, bornes d'erreur, gardien dtype.
5. [Validation](05_validation.md) — baseline, métriques, oracle de parité.
6. [Réfutation](06_refutation.md) — les murs (R1…R6), tentative active de se tromper.
7. [Exécution](07_execution.md) — sprints falsifiables, go/no-go chiffrés.
8. [Intégration](08_integration.md) — branchements par flags off-par-défaut.
9. [Rupture](09_rupture.md) — 5 paradigm shifts sans perte ; pari principal désigné.

## Thèse en une phrase
Le streaming dense lit tous les poids par token → mur physique (heures/token sur SD, Plan 06).
Le seul gain sans perte d'un ordre de grandeur est de **ne pas lire les poids dont la
contribution est exactement nulle** : les **experts MoE non routés** (pari principal, Plan 09
Rupture 1, preuve ε=0). Conséquence de cadrage : la cible 1T DOIT être MoE, en régime offline.

## Découvertes notables de l'autopsie (Phase 1 anticipée)
- Reboot complet du modèle à **chaque** token (`airllm_base.py:421-423`).
- KV-cache **désactivé** sur transformers récent → recompute quadratique (`airllm_base.py:410-412`).
- Compression existante = **avec perte** + **CUDA-only** → inutilisable sur Pi (`utils.py:94-176`).
- **Cast dtype potentiellement avec perte** : `running_dtype` défaut fp16 sur poids bf16
  (`airllm_base.py:317-319`) — bug de parité à corriger en prérequis (Plan 06-R6).
- `AirLLMMixtral` ne gâche la sparsité MoE (lit tous les experts) (`airllm_mixtral.py:8-14`).
