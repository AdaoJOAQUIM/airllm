# HYPOTHESES INVENTORY — NEURAL RUNTIME ENGINE

## Auditeur: Comité Scientifique Indépendant
## Date: 2026-06-23

---

## RÈGLES DE CLASSIFICATION

Chaque hypothèse est classifiée:

| Classification | Définition | Preuve requise |
|---------------|------------|----------------|
| DÉMONTRÉE | Prouvée mathématiquement ou empiriquement | Publications/Expériences |
| VALIDÉE | Preuves fortes mais pas conclusions | Littérature/Partial data |
| PLAUSIBLE | Raisonnable mais non prouvé | Théorie/Intuition |
| SPÉCULATIVE | Idée intéressante, peu de preuves | Hypothèses |
| RÉFUTÉE | Preuves contre | Contre-exemples |

---

## HYPOTHÈSES CENTRALES

### H1: "Intelligence as a Generative Process"

**Énoncé:** L'intelligence n'est pas stockée dans les poids, mais générée par un processus computationnel.

**Source:** README.md - "Paradigm Shift: Intelligence as a Generative Process"

**Classification:** ⚠️ SPÉCULATIVE

**Justification:**
- Aucune preuve neuroscientifique
- Modèles actuels (GPT, Llama) stockent explicitement la connaissance
- Échec des approches génératives pures (Hypernetworks)

**Littérature associée:**
- Ha, D., Dai, A., & Le, Q. V. (2016). "Hypernetworks" - Montre limitations
- Brock, A., et al. (2018). "SMASH" - Expériences négatives

**Risques:**
- R1: Information-theoretic limits
- R2: No free lunch theorem
- R3: Computational complexity

**Conclusion:** ❌ SPÉCULATIVE - Pas de preuves pour support

---

### H2: "Weights have Fractal/Self-Similar Structure"

**Énoncé:** Les matrices de poids ont une structure auto-similaire qui permet compression massive.

**Source:** representation/fractal.py, RESEARCH.md

**Classification:** ⚠️ PLAUSIBLE

**Justification:**
- Structure hiérarchique dans transformers (Self-attention)
- Low-rank structures observées
- Mais: Corrélation mesurée ~0.45 (besoin 0.70+ pour compression efficace)

**Données du projet:**
```python
# representation/fractal.py - SelfSimilarityAnalyzer
measured_correlations = {
    "attention_layers": 0.65,
    "ffn_layers": 0.30,
    "embeddings": 0.15,
    "avg": 0.45
}
# Pour fractal compression 4x+: besoin correlation > 0.70
```

**Littérature:**
- Olshausen, B. A., & Field, D. J. (1996). "Sparse coding" - Structure emerges
- Evidence for low-rank: Yu et al. 2017

**Conclusion:** 🔶 PLAUSIBLE - Struktur existe mais insuffisante pour claims

---

### H3: "Hypernetworks can Generate High-Quality Weights"

**Énoncé:** Un hypernetwork peut générer des poids équivalents à ceux entraînés.

**Source:** generation/weight_generator.py

**Classification:** ❌ RÉFUTÉE

**Justification:**
1. **Information-theoretic limit:**
   - Pour générer W de dimension N, G doit avoir ≥ N paramètres
   - Sinon: compression sans perte impossible

2. **Mode collapse:** Hypernetworks suffer from mode collapse

3. **Literature:**
   - Ha et al. 2016: "Hypernetworks" - montre limitations
   - Chang et al. 2019: "Hypertransformer" - résultats modestes
   - Krueger et al. 2017: "Bayesian hypernetworks" - instabilité

4. **Code analysis:**
   ```python
   # generation/weight_generator.py - NEVER TRAINED
   class WeightGenerator(nn.Module):
       def __init__(self, config):
           # Architecture defined but never trained
           self.generator = nn.Sequential(...)  # Random weights
   ```

**Conclusion:** ❌ RÉFUTÉE - Théorie et empirie contre

---

### H4: "Dynamic Expert Generation Works"

**Énoncé:** Les experts MoE peuvent être générés à la demande avec qualité comparable aux experts entraînés.

**Source:** experts/synthesizer.py

**Classification:** ⚠️ SPÉCULATIVE

**Justification:**
- Mélange de MoE (validé par Mixtral) et hypernetworks (non validé)
- Pas d'implémentation ni validation dans le code

**Littérature:**
- Mixtral-8x7B: Experts fixes, pas générés
- Switch Transformer: Experts fixes
- Route: https://arxiv.org/abs/2401.04066

**Conclusion:** ⚠️ SPÉCULATIVE - Combinaison non prouvée

---

### H5: "Cognitive Prefetching can Predict Layer Access"

**Énoncé:** Les patterns d'accès aux couches sont prévisibles et exploitables.

**Source:** cache/predictor.py

**Classification:** ✅ VALIDÉE

**Justification:**
- Patterns séquentiels en inference (layer 0 → layer 1 → ...)
- Markov chains fonctionnent bien pour prediction
- Implémentation existe et tests passent

**Preuves:**
- cache/predictor.py - MarkovPredictor implémenté
- Tests: test_memory.py - passent

**Conclusion:** ✅ VALIDÉE - Approach accepted in literature

---

### H6: "Hierarchical Memory Enables Large Models"

**Énoncé:** SSD→RAM→VRAM permet d'exécuter modèles > VRAM disponible.

**Source:** memory/hierarchy.py

**Classification:** ✅ DÉMONTRÉE

**Justification:**
- AirLLM démontre déjà cette approche
- PCIe SSD bandwidth ~3.5 GB/s (NVMe)
- Acceptable pour latency < batch processing

**Preuves:**
- memory/hierarchy.py - implémenté et testé
- AirLLM: fonctionne en production

**Conclusion:** ✅ DÉMONTRÉE - Approach validated by AirLLM

---

### H7: "Quantization Preserves Model Capabilities"

**Énoncé:** INT4/INT8 quantization préserve >95% des capacités.

**Source:** compression/quantizer.py

**Classification:** ✅ VALIDÉE

**Justification:**
- bitsandbytes INT8: <2% degradation
- GPTQ: <1% degradation (MMLU)
- QAT methods: near-lossless

**Littérature:**
- Dettmers et al. 2022: "LLM.int8()"
- Frantar et al. 2022: "GPTQ"
- Lin et al. 2024: "AWQ"

**Conclusion:** ✅ VALIDÉE - Littérature et implémentation

---

### H8: "100-1000x Parameter Reduction Possible"

**Énoncé:** On peut avoir 1T-param capability avec 1B-10B params stockés.

**Source:** README.md - "1T+ paramètres sur Raspberry Pi"

**Classification:** ❌ RÉFUTÉE

**Justification:**
1. **Information bottleneck:** Compression >100x lose information
2. **Scaling laws:** More params = better (no plateau)
3. **No empirical demonstration:** No small model = 1T capability

**Chiffres:**
- Compression actuelle validée: 4-8x (quantization + pruning)
- Gap avec claim: 125x à 1000x

**Conclusion:** ❌ RÉFUTÉE - Claims non supportées

---

### H9: "World Models Can Replace Knowledge Storage"

**Énoncé:** Les world models peuvent dériver la connaissance au lieu de la stocker.

**Source:** cognitive/planner.py - "world_model" concept

**Classification:** ❌ RÉFUTÉE

**Justification:**
- Faits arbitraires: "Paris = capitale France" non dérivable
- World models learn physics, not arbitrary facts
- RAG succeeds because storage > computation for facts

**Littérature:**
- World models: Ha & Schmidhuber 2018
- Limitations: Compositional generalization remains hard

**Conclusion:** ❌ RÉFUTÉE - Factual knowledge non dérivable

---

## HYPOTHÈSES SECONDAIRES

### H10: "Mixture of Experts Scales Linearly"

**Énoncé:** MoE avec N experts donne 1/N memory avec 1/N compute.

**Classification:** ✅ VALIDÉE

**Preuves:**
- Mixtral: 2/8 experts = 75% sparsity
- Switch Transformer: validate approach

---

### H11: "Flash Attention Reduces Memory Quadratically"

**Énononcé:** Flash Attention reduce O(N²) → O(N) memory.

**Classification:** ✅ DÉMONTRÉE

**Preuves:**
- Dao et al. 2022: paper + implementation

---

## MATRICE DE SYNTHÈSE

| Hypothèse | Classification | Preuve Level | Validité |
|-----------|---------------|--------------|----------|
| H1: Generative Intelligence | ❌ SPÉCULATIVE | Théorique | FAIBLE |
| H2: Fractal Structure | 🔶 PLAUSIBLE | Partial | MODÉRÉE |
| H3: Hypernetwork Generation | ❌ RÉFUTÉE | Theory + Empiric | NULLE |
| H4: Dynamic Experts | ⚠️ SPÉCULATIVE | Conceptual | FAIBLE |
| H5: Cognitive Prefetch | ✅ VALIDÉE | Implementation | FORTE |
| H6: Hierarchical Memory | ✅ DÉMONTRÉE | Production | CERTAINE |
| H7: Quantization | ✅ VALIDÉE | Production | CERTAINE |
| H8: 1000x Reduction | ❌ RÉFUTÉE | Theory + No Data | NULLE |
| H9: World Models = Storage | ❌ RÉFUTÉE | Theory | NULLE |
| H10: MoE Scales | ✅ VALIDÉE | Production | CERTAINE |
| H11: Flash Attention | ✅ DÉMONTRÉE | Production | CERTAINE |

---

## CONCLUSIONS

### Ce qui EST validé:
1. ✅ Hierarchical memory (AirLLM already does this)
2. ✅ Quantization (proven in literature)
3. ✅ Cognitive prefetch (reasonable approach)
4. ✅ MoE (proven by Mixtral, Switch)

### Ce qui est SPECULATIF:
1. ⚠️ Fractal compression (possible but unproven ratio)
2. ⚠️ Dynamic expert generation (concept interesting)

### Ce qui EST RÉFUTÉ:
1. ❌ Hypernetworks = Weight storage replacement
2. ❌ 1000x compression (no theory or evidence)
3. ❌ World models = knowledge storage
4. ❌ "Intelligence as generative process"

### Verdict:
> Le projet repose sur **3 hypothèses réfutées** (H3, H8, H9) et **1 hypothèse spéculative** (H1) comme fondations. Les composantes действительно рабочие (memory, quantization) sont des extensions de l'existant (AirLLM).

---

*Fin PHASE 2 — HYPOTHESES*
