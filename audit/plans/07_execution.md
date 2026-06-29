# Plan 07 — Exécution : sprints falsifiables

Règle d'or : **1 sprint = 1 hypothèse falsifiable + 1 critère go/no-go chiffré + vérification
« sans perte » du sprint précédent.** Aucun code optimisant n'est écrit avant que le Sprint 1
ait posé la baseline et l'oracle.

## Sprint 0 — Prérequis de vérité (avant toute optimisation)
- **Hypothèse** : « le code actuel est sans perte ». **Falsifiable** : test dtype I3.
- **Livrables** : oracle de parité (Plan 05), jeu `parity_prompts.jsonl`, env pinné, gardien
  `test_dtype_preserved`.
- **Go/No-Go** : oracle reproductible + R6 traitée (`running_dtype` aligné sur dtype natif,
  `airllm_base.py:317-319`). Si la parité du code actuel est fausse, on la corrige ici.

## Sprint 1 — Mesure du Mur (le sprint qui décide tout)
- **Hypothèse falsifiable** : « le débit disque réel rend le 1T-MoE faisable en régime offline
  (< X min/token) ». 
- **Actions** : mesurer débit séquentiel ET aléatoire sur SD + SSD USB3 cible (`fio` + lecture
  réelle d'un shard `airllm_base.py:269-300`) ; mesurer latence/token baseline + octets/token
  sur un modèle réel ; décomposer latence (Plan 05).
- **Go/No-Go chiffré** : 
  - GO si `débit_réel ≥ seuil` tel que `octets_actifs_MoE/token ÷ débit ≤ budget offline cible`
    (à fixer, ex. ≤10 min/token sur SSD).
  - NO-GO (pivot vers verdict d'impossibilité, Plan 06-R1) sinon.

## Sprint 2 — Sparsité MoE exacte (A1), prouvée sans perte
- **Hypothèse falsifiable** : « ne charger que les experts top-k donne des logits **bit-à-bit
  identiques** (`ε=0`) à la lecture de tous les experts. »
- **Actions** : (1) sous-découper les couches MoE en shards par expert (`utils.py:315`) ;
  (2) exécuter le gating avant le chargement, charger seulement les experts routés
  (`airllm_base.py:469,500-586`) ; (3) oracle bit-à-bit vs lecture dense.
- **Go/No-Go** : GO si `delta = 0` sur 100% du jeu (incl. adverse-MoE) ET octets/token réduits
  d'un facteur mesuré ≥ (N/k)·(part experts). NO-GO si `delta > 0` (R5 non levée).

## Sprint 3 — mmap + cache experts chauds (A2)
- **Hypothèse** : « `safe_open`+page cache réduit les lectures répétées sans changer un bit. »
- **Go/No-Go** : GO si `torch.equal(logits)` (ε=0) ET octets disque lus/token mesurés < Sprint 2
  sur séquence à re-routage. NO-GO sinon.

## Sprint 4 — KV-cache sans perte (A3)
- **Hypothèse** : « réactiver le KV-cache (`airllm_base.py:410-412`) donne `ε ≤ ε_admis` et
  supprime le recompute quadratique. »
- **Go/No-Go** : GO si parité tenue ET latence/token sur séquence longue ≪ baseline (passage de
  O(N²·W) à O(N·W)). NO-GO si parité cassée ou KV ne tient pas en RAM (→ Plan B disque).

## Sprint 5 — Cache backbone RAM (A5) + nettoyage reboot (A4) + masque paresseux (A7)
- **Hypothèse** : « garder le backbone dense et le squelette méta en RAM entre tokens est
  sans perte et réduit l'overhead fixe. »
- **Go/No-Go** : GO si ε=0 ET latence/token réduite du coût `reboot` mesuré au Sprint 1.

## Sprint 6 — Compression sans perte des shards froids (A6) + pré-tri (A8)
- **Hypothèse** : « zstd/LZ4 sur octets safetensors réduit les octets disque, décompression
  exacte. »
- **Go/No-Go** : GO si décompression byte-identique (ε=0 trivial) ET ratio > seuil rentable vs
  surcoût CPU de décompression sur SoC ARM.

## Sprint 7 — Endurance & 72 h
- **Hypothèse** : « run continu 72 h : usure flash ≈ 0, pas de fuite mémoire, parité stable. »
- **Go/No-Go** : GO si écritures flash ≈ 0 hors KV, pic RAM stable, logits identiques début/fin.

## Dépendances entre sprints
```
S0 (oracle/dtype) ──> S1 (mur) ──> S2 (MoE A1) ──> S3 (mmap) ──> S4 (KV) ──> S5 ──> S6 ──> S7
                           │
                           └─ NO-GO ─> Verdict d'impossibilité chiffré (Plan 06-R1)
```
Chaque flèche = « sprint précédent validé sans perte » (gardien Plan 04).

---

## Section transversale A — Justification

L'ordre n'est pas arbitraire : S1 (mur) avant tout car il peut tuer le projet → on ne code pas
A1 si le débit ne suit pas. A1 (S2) avant les multiplicateurs (S3–S6) car c'est le seul gain
« ordre de grandeur ». S7 en dernier car l'endurance ne se valide que sur le système complet.

## Section transversale B — Preuve sans perte

Chaque sprint a son critère GO conditionné à un `ε` (Plan 04) vérifié par l'oracle (Plan 05).
Un sprint qui ne peut prouver sa parité est NO-GO, indépendamment du gain de vitesse.

## Section transversale C — Tests

Chaque sprint livre : test de parité (bloquant) + test de gain chiffré (bloquant) + test de
non-régression mémoire. Versionnés dans `audit/results/sprintN/`.

## Section transversale D — Stress tests

S7 est le sprint stress dédié, mais chaque sprint ≥S2 inclut le prompt adverse-MoE et une
exécution sous RAM bridée (cgroup) avant son GO.

## Section transversale E — Mitigation

- **Risque : un sprint bloque indéfiniment** (ex. S2 ne lève pas R5). **Parade** : timebox +
  Plan B documenté par sprint (S2→sur-ensemble d'experts ; S4→KV disque ; S6→pas de
  compression). **Plan B global** : si S1 est NO-GO, basculer tout l'effort sur le livrable
  « verdict d'impossibilité chiffré + maximisation du débit offline ».
