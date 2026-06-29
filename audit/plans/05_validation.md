# Plan 05 — Validation : prouver qu'un gain est RÉEL ET sans perte

Un gain n'existe que s'il est (1) mesuré sur la cible réelle, ET (2) accompagné d'une preuve de
parité. Les deux conditions sont nécessaires ; aucune seule ne suffit.

## Baseline reproductible

### Matériel cible (à figer au Sprint 1)
- **Primaire** : Raspberry Pi 4/5, RAM ≤8 Go, stockage = carte SD (UHS-I) **et** SSD USB3
  (deux profils de débit très différents — cf. Plan « Le Mur »).
- **Substitut CI** : machine x86 sans GPU, RAM bridée à 8 Go via cgroup, stockage throttlé via
  `--device-read-bps` (cgroup/blkio) pour émuler la SD. Permet l'itération rapide ; la mesure
  finale reste sur Pi physique.

### Logiciel
- Pinner les versions (`requirements.txt` actuel pinne `bitsandbytes==0.39.0` mais
  `transformers`/`accelerate` en `git+` non pinné `requirements.txt:2-4` → **risque de
  non-reproductibilité**, à corriger : pinner des SHA exacts).
- Graine fixe, `torch.use_deterministic_algorithms(True)`.

## Métriques (toutes mesurées, jamais théoriques)

| Métrique | Définition | Outil |
|---|---|---|
| **Débit disque réel** | MB/s en lecture séquentielle ET aléatoire 4K–4M sur le stockage cible | `fio` / `hdparm` + lecture réelle d'un shard via `load_layer_to_cpu` instrumenté (`airllm_base.py:269-300`) |
| **Octets/token lus** | somme des tailles de shards effectivement lus pour 1 token | compteur dans `load_layer` (`utils.py:115`) |
| **Latence/token** | wall-clock par token généré | `profiling_mode` (`airllm_base.py:417-418, 626-632`) |
| **Pic RAM** | RSS max pendant 1 forward | `resource.getrusage` / `/proc` |
| **Parité** | `‖logits_opt−logits_ref‖_∞` | oracle ci-dessous |
| **Endurance flash** | octets écrits/jour (cf. Plan « Le Mur ») | compteur d'écritures |

## L'ORACLE de parité (la pièce maîtresse)

```
def parity_oracle(prompt, ref_runner, opt_runner, eps):
    logits_ref = ref_runner(prompt)      # HF Transformers vanille, dtype natif
    logits_opt = opt_runner(prompt)      # AirLLM optimisé
    delta = (logits_opt - logits_ref).abs().max().item()
    assert delta <= eps, f"VIOLATION sans-perte: {delta} > {eps}"
    return delta
```

- **`ref_runner`** = HF `AutoModelForCausalLM` chargé normalement (quand le modèle tient) OU,
  pour un 1T qui ne tient nulle part, une **référence par couche** : on compare la sortie de
  CHAQUE couche optimisée à la sortie de la même couche calculée naïvement (chargement complet,
  tous experts, sans nos optimisations). La composition de parités-par-couche prouve la parité
  globale (les couches sont fonctionnelles et déterministes).
- **Jeu de référence reproductible** : un fichier `audit/datasets/parity_prompts.jsonl` (à
  créer) de N prompts couvrant : court, long (proche `max_seq_len`), tokens rares, adverse-MoE
  (force une grande diversité d'experts).

## Protocole de validation d'un item (go/no-go)

Un item du backlog (Plan 03) est **validé** ssi, sur le jeu de référence + stress :
1. **Parité** : `delta ≤ ε_admis` (Plan 04) sur 100% des prompts. Sinon → NO-GO immédiat.
2. **Gain réel** : octets/token (ou latence/token) **mesuré** strictement meilleur que la
   baseline, sur le matériel cible, avec intervalle de confiance (≥30 répétitions).
3. **Non-régression mémoire** : pic RAM ≤ baseline.

Aucun item ne passe en `main` du fork sans ces 3 cases cochées.

## Comment on distingue « limite physique » de « limite d'ingénierie »

Pour chaque mesure de latence, décomposer via le profiler (`profiler.py`,
`airllm_base.py:276-298`) :
`latence = lecture_disque + transfert_device + compute + overhead_python`.
- Si `lecture_disque` ≈ `octets/token ÷ débit_réel` → **limite physique** (irréductible sans
  réduire les octets, c.-à-d. sans A1).
- Si `overhead_python`/`reboot` (`airllm_base.py:421-423`) domine → **limite d'ingénierie**
  (réductible). Cette décomposition est livrée à chaque sprint.

---

## Section transversale A — Justification

On exige *mesuré sur cible* parce que la mission l'impose (« Mesure > opinion ») et parce que
le débit SD réel diverge fortement du débit nominal (aléatoire ≪ séquentiel). On exige
*parité + gain ensemble* parce qu'un gain de vitesse obtenu en cassant la parité est un faux
gain (ce serait de la quantization déguisée). L'oracle par-couche est choisi car un 1T ne tient
sur aucune référence monolithique — la décomposition par couche est la seule voie de preuve
exécutable.

## Section transversale B — Preuve sans perte

L'oracle EST le mécanisme de preuve « sans perte vérifiée » de l'invariant. Sa solidité tient
à : (a) référence = implémentation indépendante (HF vanille), (b) couverture du jeu incluant le
pire cas adverse, (c) borne `ε` explicite par item (Plan 04).

## Section transversale C — Tests

Le plan définit lui-même la suite de tests de validation : `test_throughput`, `test_bytes_
per_token`, `test_parity_oracle`, `test_peak_ram`. Chacun produit un artefact chiffré versionné
dans `audit/results/`.

## Section transversale D — Stress tests

La validation n'est acceptée que si la parité tient AUSSI sous stress (Plan 04-D). Un gain
mesuré uniquement en conditions nominales mais qui casse la parité sous RAM saturée est rejeté.
Mesure d'endurance flash sur run 72 h (extrapolée + échantillon réel).

## Section transversale E — Mitigation

- **Risque : baseline non reproductible** (transformers `git+` non pinné, `requirements.txt:2`).
  **Parade** : pinner des SHA + publier `audit/env.lock`. **Plan B** : conteneur figé.
- **Risque : substitut CI ment** (x86 throttlé ≠ Pi). **Parade** : recaler systématiquement le
  go/no-go final sur Pi physique ; le substitut ne sert qu'au tri rapide. **Plan B** : louer un
  Pi distant pour la CI nocturne.
