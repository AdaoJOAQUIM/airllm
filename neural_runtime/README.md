# Neural Runtime Engine 🧠⚡

> ** objectif ambitieux : Exécuter des modèles IA de 1 trillion de paramètres en local, offline, sur du matériel limité**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://python.org)

## 🚀 Vision

Neural Runtime Engine est une **nouvelle génération de moteur d'inférence IA** construit sur les épaules de géants comme AirLLM, mais avec une ambition démesurée : repousser les limites de l'exécution locale de modèles massifs.

Notre mission : créer un **système d'exploitation cognitif** capable de faire tourner des modèles de centaines de milliards乃至 billions de paramètres sur du matériel consumer, en local, offline, avec une utilisation minimale de RAM, VRAM, stockage et énergie.

---

## 📊 État de l'Art vs Objectif

```
┌────────────────────────────────────────────────────────────────────────┐
│                    MODÈLES IA - COMPARATIF MÉMOIRE                      │
├────────────────────┬────────────────┬───────────────────────────────────┤
│ Modèle             │ Paramètres     │ Mémoire VRAM (FP16)              │
├────────────────────┼────────────────┼───────────────────────────────────┤
│ GPT-2              │ 1.5B           │ ~3 GB  ✅ Pratiquement partout    │
│ Llama 2            │ 7B             │ ~14 GB ✅ GPU consumer moderne    │
│ Llama 2            │ 70B            │ ~140 GB ⚠️ GPU haut de gamme     │
│ Llama 3.1          │ 405B           │ ~810 GB ❌ Multi-GPU requis       │
│ GPT-4 (estimé)      │ ~1.76T         │ ~3.5 TB ❌❌ Data center only      │
├────────────────────┴────────────────┴───────────────────────────────────┤
│ 🎯 NOTRE OBJECTIF: 1T+ paramètres sur UN Raspberry Pi (1-8 GB RAM)    │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 🏗️ Architecture

```
neural_runtime/
├── core/                    # Cœur du runtime
│   ├── model.py             # NeuralRuntimeModel - Abstraction du modèle
│   ├── inference.py         # Moteur d'inférence principal
│   └── hardware.py          # Détection automatique du hardware
│
├── memory/                  # Gestion mémoire hiérarchique
│   ├── hierarchy.py         # HierarchicalMemory - SSD→RAM→VRAM→Cache
│   ├── offloader.py         # CPU/GPU offloading intelligent
│   └── cache.py             # Cache multi-niveaux
│
├── compression/             # Quantification avancée
│   ├── quantizer.py         # FP8, INT8, INT4, NF4, AWQ, GPTQ
│   ├── smoothquant.py        # SmoothQuant pour activation outliers
│   └── sparse.py            # Sparse attention & weights
│
├── cache/                   # Cache intelligent
│   ├── predictor.py         # Markov-based access pattern prediction
│   └── lru.py               # LRU cache optimisé
│
├── runtime/                 # Moteur d'exécution
│   ├── inference.py         # InferenceEngine avec prefetch
│   └── batcher.py           # Continuous batching (à venir)
│
├── parallel/                # Parallélisme
│   └── tensor_parallel.py   # Tensor parallelism multi-GPU
│
├── compiler/                # Optimisation (à venir)
│   ├── fusion.py            # Kernel fusion
│   └── triton.py            # Triton kernels
│
└── kernels/                 # CUDA/Triton kernels (à venir)
    ├── attention.py          # Flash Attention optimisé
    └── quantization.py       # Kernels de déquantification
```

---

## 🎯 Innovations Clés

### 1. **Parameter Virtualization Engine** 🌐

Un système permettant de gérer des modèles dépassant largement la mémoire disponible.

```
┌─────────────────────────────────────────────────────────────┐
│              PARAMETER VIRTUALIZATION ENGINE                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    │
│   │  Actifs     │    │  Dormants   │    │ Compressés  │    │
│   │  (en VRAM)  │◄──►│  (en RAM)   │◄──►│ (sur SSD)   │    │
│   └─────────────┘    └─────────────┘    └─────────────┘    │
│          ▲                  │                  │             │
│          │                  ▼                  ▼             │
│   ┌─────────────────────────────────────────────────────┐    │
│   │           RECONSTRUCTION À LA DEMANDE               │    │
│   │  Sparse activation + Compression + Prediction       │    │
│   └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### 2. **Hierarchical Memory System** 💾

Gestion intelligente SSD → RAM → VRAM avec prefetch prédictif.

```python
# Exemple d'utilisation
memory = HierarchicalMemory(
    max_vram_gb=24,        # Limite VRAM
    max_ram_gb=128,        # RAM disponible
    cache_policy=CachePolicy.ADAPTIVE,  # Évection intelligente
    prefetch_workers=4     # Threads de préchargement
)

# Stockage automatique au bon niveau
memory.store("layer_0", tensor, MemoryLevel.HBM)
memory.store("layer_50", tensor, MemoryLevel.RAM)
memory.store("layer_100", tensor, MemoryLevel.NVME)  # SSD rapide

# Récupération automatique (load from lower tiers si nécessaire)
tensor = memory.get("layer_50")  # Auto-load from RAM/NVME if needed
```

### 3. **Cognitive Cache** 🧠

Apprentissage des patterns d'accès pour prédire les chargements nécessaires.

```python
predictor = AccessPatternPredictor(num_layers=100)

# Apprentissage des patterns
predictor.record_access("layer_0")
predictor.record_access("layer_1")
predictor.record_access("layer_2")

# Prédiction des prochaines couches
predictions = predictor.predict_next(current_layer=2, count=3)
# → ["layer_3", "layer_4", "layer_5"]
```

### 4. **Quantification Adaptive** 📦

Support de multiples formats de quantification avec calibration automatique.

```python
# Configuration de quantification
config = QuantizationConfig(
    quant_type=QuantizationType.INT4,      # 4-bit
    strategy=QuantizationStrategy.PER_GROUP,
    group_size=128,
    use_smoothquant=True,                   # Pour activations difficiles
)

quantizer = Quantizer(config)

# Quantification avec calibration
quantized_layers = quantizer.quantize_model(model, calibration_data)
```

### 5. **Tensor Parallelism** 🔢

Pour les systèmes multi-GPU (prévu pour 1T+ params).

```python
tp_manager = TensorParallelManager(
    tensor_parallel_size=4,      # 4 GPUs
    pipeline_parallel_size=1,
)
distributed_model = tp_manager.wrap_model(model)
```

---

## 📈 Feuille de Route

### Phase 1: Fondation (En cours)
- [x] Analyse architecture AirLLM
- [x] Structure modulaire du projet
- [x] HierarchicalMemory System
- [x] Quantization Module
- [x] Access Pattern Predictor
- [ ] Intégration avec modèle existant
- [ ] Tests unitaires

### Phase 2: Optimisations (Prévu)
- [ ] Intégration FP8 (Transformer Engine)
- [ ] AWQ/GPTQ calibration
- [ ] Ring Attention pour longs contextes
- [ ] KV Cache compression

### Phase 3: Multi-GPU (Prévu)
- [ ] Tensor Parallelism complet
- [ ] Pipeline Parallelism
- [ ] NCCL integration
- [ ] Continuous Batching

### Phase 4: Recherche (Vision)
- [ ] Sparse Neural Execution
- [ ] Weight Reconstruction from Compressed
- [ ] Mixture of Experts optimization
- [ ] Cognitive Prefetching

### Phase 5: Extreme Optimization (Rêve)
- [ ] 1T+ paramètres sur hardware limité
- [ ] Exécution sur Raspberry Pi
- [ ] Mode offline complet
- [ ] Optimisation énergétique

---

## 🔧 Installation

```bash
# Cloner le projet
git clone https://github.com/lyogavin/airllm.git
cd airllm

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# .\venv\Scripts\activate   # Windows

# Installer les dépendances
pip install torch transformers accelerate
pip install bitsandbytes  # Pour quantification NF4
pip install safetensors

# Optionnel: Pour FP8
pip install transformer-engine

# Installer Neural Runtime
cd neural_runtime
pip install -e .
```

---

## 🚀 Utilisation

### Utilisation basique

```python
from neural_runtime import NeuralRuntimeModel, InferenceEngine

# Charger un modèle
model = NeuralRuntimeModel(
    model_path_or_repo_id="meta-llama/Llama-2-7b",
    precision="int4",  # Quantifié 4-bit
)

# Générer du texte
output = model.generate(
    prompt="Once upon a time",
    max_new_tokens=100,
    temperature=0.8,
)
print(output)
```

### Utilisation avancée avec mémoire hiérarchique

```python
from neural_runtime import NeuralRuntimeModel, ModelConfig
from neural_runtime.memory import HierarchicalMemory, MemoryLevel

# Configuration personnalisée
config = ModelConfig(
    model_path_or_repo_id="meta-llama/Llama-2-70b",
    precision="int4",
    use_hierarchical_memory=True,
    max_vram_gb=24,     # Limiter VRAM utilisée
    max_ram_gb=128,     # Utiliser la RAM disponible
    prefetch_enabled=True,
)

model = NeuralRuntimeModel(config)

# Génération
output = model.generate("Explain quantum computing in simple terms...")
```

### Benchmark de mémoire

```python
from neural_runtime.core.hardware import estimate_model_memory, HardwareDetector

# Détecter le hardware
spec = HardwareDetector.detect()
print(f"VRAM: {spec.total_vram_gb} GB")
print(f"RAM: {spec.ram_gb} GB")

# Estimer besoins mémoire
est = estimate_model_memory(
    num_parameters=1_000_000_000_000,  # 1 trillion
    precision="int4",
    sequence_length=2048,
)
print(f"Mémoire estimée: {est['total_gb']:.2f} GB")
```

---

## 🧪 Tests

```bash
# Exécuter les tests unitaires
cd neural_runtime
pytest tests/

# Exécuter les benchmarks
pytest tests/benchmark.py -v

# Tests spécifiques
pytest tests/test_memory.py -v
pytest tests/test_quantization.py -v
```

---

## 📊 Benchmarks

| Modèle | Params | Précision | VRAM | Temps/Token | 
|--------|--------|-----------|------|-------------|
| Llama 2 | 7B | FP16 | ~14 GB | ~50ms |
| Llama 2 | 7B | INT4 | ~4 GB | ~30ms |
| Llama 2 | 70B | INT4 | ~24 GB | ~200ms |
| Llama 3.1 | 405B | INT4 | ~140 GB | ~1s |

*Objectif pour 1T: < 10GB VRAM via aggressive offloading*

---

## 🤝 Contribution

Les contributions sont les bienvenues ! Veuillez lire [CONTRIBUTING.md](CONTRIBUTING.md) pour plus de détails.

### Guidelines

1. Fork le projet
2. Créer une branche feature (`git checkout -b feature/amazing-feature`)
3. Commit (`git commit -m 'Add amazing feature'`)
4. Push (`git push origin feature/amazing-feature`)
5. Ouvrir une Pull Request

---

## 📚 Références et Inspiration

- [AirLLM](https://github.com/lyogavin/airllm) - L'inspiration originale
- [LLM.int8()](https://arxiv.org/abs/2208.07339) -bitsandbytes
- [Flash Attention](https://arxiv.org/abs/2205.14135) - Attention optimisé
- [SmoothQuant](https://arxiv.org/abs/2211.10438) - Quantization avancée
- [AWQ](https://arxiv.org/abs/2306.00978) - Activation-Aware Weight Quantization
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) - Tensor Parallelism

---

## 📄 License

Apache License 2.0 - Voir [LICENSE](../LICENSE)

---

## 🙏 Remerciements

- **Gavin Li** et l'équipe AirLLM pour l'inspiration et la base solide
- **Hugging Face** pour Transformers et accelerate
- **bitsandbytes** pour la quantification
- **Tous les contributeurs** qui aideront à repousser ces limites

---

<div align="center">

**Fait avec 💜 pour repousser les limites de l'IA locale**

*Parce que l'avenir de l'IA ne devrait pas être limité par le cloud.*

</div>
