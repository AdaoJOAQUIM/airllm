# Plan 06 — Réfutation : ce qui tuerait le projet

Discipline : on essaie activement de prouver que l'objectif est **impossible**. Chaque
réfutation est soit confirmée (mur physique → on l'accepte et on pivote), soit levée par
mesure.

## R1 — Le mur de bande passante (réfutation la plus forte)

**Thèse de mort** : « lire les poids actifs d'un 1T à chaque token est plus lent que toute
patience humaine. »

Chiffrage (ordres de grandeur, à raffiner au Sprint 1) :

| Cas | Octets à lire/token | Débit réel `[HYPOTHÈSE, à mesurer]` | Latence/token |
|---|---|---|---|
| 1T **dense** bf16, streaming naïf | 2 To | SD UHS-I ~40 MB/s | **~14 h/token** |
| 1T dense bf16 | 2 To | SSD USB3 sur Pi ~300 MB/s | **~1,9 h/token** |
| 1T **MoE**, ~32 B actifs bf16 (A1) | ~64 Go | SD ~40 MB/s | **~27 min/token** |
| 1T MoE, ~32 B actifs bf16 (A1) | ~64 Go | SSD USB3 ~300 MB/s | **~3,6 min/token** |
| 1T MoE, actifs + backbone caché RAM (A1+A5) | < 64 Go (experts seuls) | SSD ~300 MB/s | **~minutes/token** |

**Verdict honnête** : pour un 1T **dense**, R1 est **confirmée** — pas d'inférence interactive
sans perte sur Pi. Pour un 1T **MoE**, R1 est **repoussée** au régime « batch offline »
(minutes/token), pas tuée. ⇒ La survie du projet **exige** une cible MoE (cf. Plan 02-E). C'est
le résultat de réfutation central : il borne le périmètre.

## R2 — Latence/token incompressible

Même avec A1+A2+A3+A5, la borne basse reste `octets_experts_actifs/token ÷ débit`. Si un usage
exige < 1 s/token, **réfuté** : impossible sur Pi sans perte. ⇒ On ne promet QUE du débit
offline (génération longue non interactive, type « calcul de nuit »). À cadrer comme contrat.

## R3 — Endurance flash (mort matérielle)

**Thèse de mort** : « le streaming use la carte SD jusqu'à la panne. »
- Nuance cruciale : le streaming **LIT**, il n'écrit pas (le forward ne réécrit pas les poids ;
  les écritures sont au `split_and_save_layers` une fois, `utils.py:321-323`). Les **lectures
  n'usent pas** la flash NAND (seules les écritures/effacements le font).
- **Donc R3 est largement réfutée pour l'inférence** : usure ≈ 0 en régime stable. Les seules
  écritures sont : (a) le split initial (one-shot), (b) un éventuel KV-cache sur disque (A3) →
  **à surveiller** : si on swappe le KV-cache sur SD à chaque token, on réintroduit de l'usure.
  Parade : KV-cache en RAM ou sur SSD (meilleure endurance), pas sur SD.

## R4 — Le routage MoE n'est pas assez creux

**Thèse de mort** : « sur un vrai prompt, l'union des experts routés sur la séquence approche
N → on relit presque tout. »
- Par token, seuls k experts/couche. Mais sur N tokens, l'**union** peut être grande. Toutefois
  A1 raisonne **par token** (on charge top-k pour CE token), pas par séquence. Le mmap (A2) +
  page cache rend gratuit le re-routage d'un expert déjà chaud. **Réfutation levée** tant que la
  RAM cache suffit pour les experts chauds. `[HYPOTHÈSE : distribution de routage à mesurer]`.

## R5 — La parité bit-à-bit MoE est fausse

**Thèse de mort** : « omettre des experts change la sortie à cause d'égalités dans le top-k ou
d'un ordre de sommation différent. »
- Risque réel d'**égalité de scores de gating** → départage non déterministe → expert différent
  routé. Si cela arrive, A1 n'est PAS sans perte sur ce token.
- **Parade/preuve** (Plan 04, Plan 09) : reproduire exactement la règle de top-k du modèle de
  référence (départage par indice), et sommer dans le même ordre (ajouter `0·E_e` est neutre en
  IEEE-754). Si le routage de référence est lui-même non déterministe, A1 charge un sur-ensemble
  → sans perte garanti, gain moindre. **Réfutation levée** sous condition prouvable.

## R6 — Le cast dtype casse déjà la parité (réfutation interne)

`airllm_base.py:317-319` caste en `running_dtype` (fp16 par défaut). Si la cible est bf16, le
**code actuel est déjà avec perte** → « AirLLM sans perte » est faux par défaut.
**Réfutation d'une fausse prémisse** : on doit corriger I3 (Plan 04) AVANT de revendiquer quoi
que ce soit. C'est un prérequis, pas un détail.

## Synthèse des murs

| # | Nature | Statut | Conséquence |
|---|---|---|---|
| R1 | Physique (bande passante) | Confirmé (dense) / Repoussé (MoE) | Cible DOIT être MoE + régime offline |
| R2 | Physique (latence basse) | Confirmé | Pas d'interactif ; contrat = batch |
| R3 | Physique (usure flash) | Réfuté pour lecture | Surveiller seulement le KV sur disque |
| R4 | Statistique (routage) | Conditionnel | Dépend du cache d'experts chauds |
| R5 | Numérique (parité MoE) | Levable | Reproduire top-k + ordre de somme |
| R6 | Ingénierie (dtype) | Bug actuel | Corriger I3 en prérequis |

---

## Section transversale A — Justification

La réfutation précède l'optimisation pour éviter d'investir dans une voie morte. On sépare
explicitement « physique » (R1,R2,R3) de « ingénierie/numérique » (R4,R5,R6) car seules les
premières sont des murs vrais ; les secondes sont des conditions à satisfaire.

## Section transversale B — Preuve sans perte

R5 et R6 sont les réfutations qui visent directement l'invariant. Elles sont traitées par
preuve (Plan 09) et par gardien (Plan 04). Tant qu'elles ne sont pas levées par oracle, A1 est
considéré **non sans perte** et interdit en production.

## Section transversale C — Tests

- `test_bandwidth_wall` : mesure `octets/token ÷ débit` et compare à la latence observée ;
  confirme/infirme R1/R2 sur cible.
- `test_routing_sparsity` : mesure la distribution du nombre d'experts uniques par token et par
  fenêtre (R4).
- `test_moe_tie_break` : injecte des scores de gating à égalité, vérifie le départage (R5).

## Section transversale D — Stress tests

- 72 h de génération : compteur d'écritures flash (R3) ; vérifier usure ≈ 0 hors KV.
- Prompt adverse-MoE : pousse R4 au pire cas.
- Run lent (SD) vs rapide (SSD) : mêmes logits ⇒ confirme que la vitesse n'altère pas le
  résultat (lève une objection R2 sur la qualité).

## Section transversale E — Mitigation

- **Si R1 confirmée même en MoE** (débit trop faible) : **Plan B** = livrer le verdict chiffré
  « impossible interactif » + maximiser le débit offline ; c'est un livrable honnête autorisé.
- **Si R5 non levable** (routage de réf non déterministe) : **Plan B** = sur-ensemble d'experts
  (sans perte, gain réduit). 
- **Si R4 fatale** (routage trop dense) : **Plan B** = compression sans perte des shards (A6)
  pour réduire les octets, gain modeste mais réel.
