"""
Neural Runtime Model - Core model abstraction.
Designed for ultra-large models with hierarchical memory management.
"""

import os
import gc
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging

import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoConfig, AutoTokenizer, GenerationConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

from .hardware import HardwareSpec, HardwareDetector, DeviceType
from ..memory.hierarchy import HierarchicalMemory, MemoryLevel
from ..compression.quantizer import Quantizer, QuantizationConfig
from ..cache.predictor import AccessPatternPredictor

logger = logging.getLogger(__name__)


@dataclass
class LayerInfo:
    """Information about a model layer."""
    name: str
    index: int
    size_bytes: int
    is_embedding: bool = False
    is_lm_head: bool = False
    is_norm: bool = False
    is_transformer: bool = False


@dataclass
class ModelConfig:
    """Configuration for Neural Runtime Model."""
    # Model source
    model_path_or_repo_id: str
    
    # Precision and compression
    precision: str = "auto"  # fp16, bf16, fp8, int8, int4, auto
    quantization_config: Optional[QuantizationConfig] = None
    
    # Memory management
    memory_level: MemoryLevel = MemoryLevel.HBM  # Where to keep active layers (HBM = VRAM on GPU)
    max_vram_gb: float = 0.0  # 0 = auto-detect
    max_ram_gb: float = 0.0
    use_hierarchical_memory: bool = True
    
    # Inference settings
    max_sequence_length: int = 512
    batch_size: int = 1
    use_cache: bool = True
    
    # Optimization
    use_flash_attention: bool = True
    use_tensor_parallel: bool = False
    tensor_parallel_size: int = 1
    
    # Prefetching
    prefetch_enabled: bool = True
    predict_access_pattern: bool = True
    
    # HuggingFace
    hf_token: Optional[str] = None


class NeuralRuntimeModel(nn.Module):
    """
    Neural Runtime Model - A new generation inference engine.
    
    Key improvements over AirLLM:
    - Hierarchical memory management (VRAM → RAM → SSD)
    - Adaptive precision selection
    - Intelligent prefetching
    - Multi-GPU support via tensor parallelism
    - Parameter virtualization
    
    Usage:
        model = NeuralRuntimeModel("meta-llama/Llama-2-70b")
        output = model.generate("Hello, world!")
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        
        self.config = config
        self._validate_config()
        
        # Hardware detection
        self.hardware = HardwareDetector.detect(verbose=True)
        
        # Apply precision based on hardware
        if config.precision == "auto":
            config.precision = self.hardware.recommended_precision
        
        # Set device
        self._setup_device()
        
        # Initialize components
        self._init_tokenizer()
        self._init_model()
        self._init_layer_index()
        
        # Memory management
        if config.use_hierarchical_memory:
            self._init_memory_system()
        
        # Cache predictor
        if config.predict_access_pattern:
            self._init_predictor()
        
        # Quantization
        if config.quantization_config:
            self._init_quantizer()
        
        logger.info(f"NeuralRuntimeModel initialized")
        logger.info(f"  Model: {self.model_path}")
        logger.info(f"  Precision: {config.precision}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Layers: {len(self.layer_info)}")
    
    def _validate_config(self) -> None:
        """Validate and normalize configuration."""
        if not self.config.model_path_or_repo_id:
            raise ValueError("model_path_or_repo_id is required")
        
        # Normalize precision
        valid_precisions = ["auto", "fp32", "fp16", "bf16", "fp8", "int8", "int4"]
        if self.config.precision not in valid_precisions:
            raise ValueError(f"Invalid precision: {self.config.precision}. "
                           f"Must be one of {valid_precisions}")
    
    def _setup_device(self) -> None:
        """Setup computation device."""
        if self.hardware.device_type == DeviceType.CUDA_MULTI and self.config.use_tensor_parallel:
            self.device = torch.device("cuda")
            self.tensor_parallel_size = min(
                self.hardware.device_count,
                self.config.tensor_parallel_size
            )
        elif self.hardware.device_type == DeviceType.CUDA:
            self.device = torch.device("cuda:0")
            self.tensor_parallel_size = 1
        elif self.hardware.device_type == DeviceType.MPS:
            self.device = torch.device("mps")
            self.tensor_parallel_size = 1
        else:
            self.device = torch.device("cpu")
            self.tensor_parallel_size = 1
        
        logger.info(f"Using device: {self.device}")
    
    def _init_tokenizer(self) -> None:
        """Initialize tokenizer."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_path_or_repo_id,
                token=self.config.hf_token,
                trust_remote_code=True
            )
            
            # Set pad token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
        except Exception as e:
            logger.warning(f"Failed to load tokenizer: {e}")
            self.tokenizer = None
    
    def _init_model(self) -> None:
        """Initialize the base model."""
        from accelerate import init_empty_weights
        from transformers import AutoModelForCausalLM
        
        self.model_path = self._resolve_model_path()
        
        # Load config
        self.model_config = AutoConfig.from_pretrained(
            self.model_path,
            token=self.config.hf_token,
            trust_remote_code=True
        )
        
        # Generation config
        try:
            self.generation_config = GenerationConfig.from_pretrained(
                self.model_path
            )
        except:
            self.generation_config = GenerationConfig()
        
        # Create empty model
        with init_empty_weights():
            self.model = AutoModelForCausalLM.from_config(
                self.model_config,
                trust_remote_code=True
            )
        
        self.model.eval()
        self.model.tie_weights()
    
    def _resolve_model_path(self) -> Path:
        """Resolve model path from repo ID or local path."""
        path = Path(self.config.model_path_or_repo_id)
        
        if path.exists():
            return path
        
        # Download from HuggingFace
        import huggingface_hub
        cache_path = huggingface_hub.snapshot_download(
            self.config.model_path_or_repo_id,
            token=self.config.hf_token
        )
        return Path(cache_path)
    
    def _init_layer_index(self) -> None:
        """Index all model layers for memory management."""
        self.layer_info: List[LayerInfo] = []
        
        # Standard layer naming conventions
        layer_prefixes = self._detect_layer_naming()
        
        # Index embedding
        embed_layer = self._find_embedding_layer()
        if embed_layer:
            size = sum(p.numel() * p.element_size() for p in embed_layer.parameters())
            self.layer_info.append(LayerInfo(
                name="embedding",
                index=0,
                size_bytes=size,
                is_embedding=True
            ))
        
        # Index transformer layers
        layer_idx = 1
        transformer_layers = self._find_transformer_layers()
        for idx, layer in enumerate(transformer_layers):
            size = sum(p.numel() * p.element_size() for p in layer.parameters())
            self.layer_info.append(LayerInfo(
                name=f"layer_{idx}",
                index=layer_idx,
                size_bytes=size,
                is_transformer=True
            ))
            layer_idx += 1
        
        # Index norm layer
        norm_layer = self._find_norm_layer()
        if norm_layer:
            size = sum(p.numel() * p.element_size() for p in norm_layer.parameters())
            self.layer_info.append(LayerInfo(
                name="final_norm",
                index=layer_idx,
                size_bytes=size,
                is_norm=True
            ))
            layer_idx += 1
        
        # Index lm_head
        lm_head = self._find_lm_head()
        if lm_head:
            size = sum(p.numel() * p.element_size() for p in lm_head.parameters())
            self.layer_info.append(LayerInfo(
                name="lm_head",
                index=layer_idx,
                size_bytes=size,
                is_lm_head=True
            ))
        
        logger.info(f"Indexed {len(self.layer_info)} layers")
    
    def _detect_layer_naming(self) -> Dict[str, str]:
        """Detect layer naming convention from config."""
        arch = self.model_config.architectures[0] if self.model_config.architectures else ""
        
        if "Llama" in arch:
            return {
                "embed": "model.embed_tokens",
                "layer_prefix": "model.layers",
                "norm": "model.norm",
                "lm_head": "lm_head"
            }
        elif "Qwen" in arch:
            return {
                "embed": "model.embed_tokens",
                "layer_prefix": "model.layers",
                "norm": "model.norm",
                "lm_head": "lm_head"
            }
        elif "ChatGLM" in arch:
            return {
                "embed": "model.embedding",
                "layer_prefix": "model.layers",
                "norm": "model.final_layernorm",
                "lm_head": "lm_head"
            }
        else:
            # Try to auto-detect
            return {
                "embed": "model.embed_tokens",
                "layer_prefix": "model.layers",
                "norm": "model.norm",
                "lm_head": "lm_head"
            }
    
    def _find_embedding_layer(self) -> Optional[nn.Module]:
        """Find the embedding layer."""
        prefixes = ["embed_tokens", "embedding", "wte", "word_embeddings"]
        return self._find_layer_by_suffix(prefixes)
    
    def _find_transformer_layers(self) -> List[nn.Module]:
        """Find all transformer layers."""
        layer_modules = []
        
        # Common patterns
        for attr_name in dir(self.model):
            if "layers" in attr_name or "h_" in attr_name:
                obj = getattr(self.model, attr_name)
                if isinstance(obj, nn.ModuleList):
                    layer_modules = list(obj)
                    break
        
        # Fallback: search recursively
        if not layer_modules:
            for name, module in self.model.named_modules():
                if "layer" in name.lower() and isinstance(module, nn.ModuleList):
                    layer_modules = list(module)
                    break
        
        return layer_modules
    
    def _find_norm_layer(self) -> Optional[nn.Module]:
        """Find the final normalization layer."""
        suffixes = ["norm", "final_layernorm", "ln_f"]
        return self._find_layer_by_suffix(suffixes)
    
    def _find_lm_head(self) -> Optional[nn.Module]:
        """Find the LM head."""
        if hasattr(self.model, "lm_head"):
            return self.model.lm_head
        if hasattr(self.model, "output"):
            return self.model.output
        return self._find_layer_by_suffix(["lm_head", "output"])
    
    def _find_layer_by_suffix(self, suffixes: List[str]) -> Optional[nn.Module]:
        """Find a layer by suffix matching."""
        for name, module in self.model.named_modules():
            for suffix in suffixes:
                if name.endswith(suffix) or suffix in name:
                    return module
        return None
    
    def _init_memory_system(self) -> None:
        """Initialize hierarchical memory management."""
        max_vram = self.config.max_vram_gb or self.hardware.vram_per_device_gb
        max_ram = self.config.max_ram_gb or self.hardware.ram_gb * 0.8
        
        self.memory = HierarchicalMemory(
            max_vram_gb=max_vram,
            max_ram_gb=max_ram,
            storage_path=self.model_path / "neural_cache"
        )
        
        logger.info(f"Hierarchical memory initialized: VRAM={max_vram:.1f}GB, RAM={max_ram:.1f}GB")
    
    def _init_predictor(self) -> None:
        """Initialize access pattern predictor."""
        self.predictor = AccessPatternPredictor(
            num_layers=len(self.layer_info)
        )
    
    def _init_quantizer(self) -> None:
        """Initialize quantization system."""
        self.quantizer = Quantizer(self.config.quantization_config)
    
    # ==================== Forward Methods ====================
    
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_values: Optional[List] = None,
        use_cache: bool = True,
        **kwargs
    ) -> CausalLMOutputWithPast:
        """
        Forward pass with hierarchical memory management.
        """
        raise NotImplementedError("Use InferenceEngine for forward pass")
    
    def generate(
        self,
        prompt: Union[str, List[str]],
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9,
        **kwargs
    ) -> Union[str, List[str]]:
        """
        Generate text from prompt(s).
        
        Args:
            prompt: Input text or list of texts
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            
        Returns:
            Generated text(s)
        """
        from ..runtime.inference import InferenceEngine
        
        engine = InferenceEngine(self)
        return engine.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs
        )
    
    # ==================== Utility Methods ====================
    
    def get_memory_footprint(self) -> Dict[str, float]:
        """Get current memory footprint in GB."""
        footprint = {}
        
        # VRAM
        if torch.cuda.is_available():
            footprint["vram_allocated_gb"] = torch.cuda.memory_allocated() / (1024**3)
            footprint["vram_reserved_gb"] = torch.cuda.memory_reserved() / (1024**3)
        
        # RAM (estimate)
        import psutil
        footprint["ram_used_gb"] = psutil.Process().memory_info().rss / (1024**3)
        
        return footprint
    
    def clean_memory(self) -> None:
        """Clean up memory."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def __repr__(self) -> str:
        return (
            f"NeuralRuntimeModel(\n"
            f"  model={self.config.model_path_or_repo_id},\n"
            f"  precision={self.config.precision},\n"
            f"  device={self.device},\n"
            f"  layers={len(self.layer_info)}\n"
            f")"
        )
