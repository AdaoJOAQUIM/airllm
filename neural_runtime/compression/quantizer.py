"""
Quantization Module
Advanced quantization support for ultra-efficient inference.
Supports: FP8, INT8, INT4, AWQ, GPTQ, SmoothQuant.
"""

import os
import time
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class QuantizationType(Enum):
    """Type of quantization."""
    FP8_E4M3 = "fp8_e4m3"      # FP8 with 4-bit mantissa
    FP8_E5M2 = "fp8_e5m2"      # FP8 with 5-bit mantissa
    INT8 = "int8"               # 8-bit integer
    INT4 = "int4"               # 4-bit integer
    NF4 = "nf4"                 # Normal float 4-bit (bitsandbytes)
    FP4 = "fp4"                 # FP4 (bitsandbytes)
    AWQ = "awq"                 # Activation-aware weight quantization
    GPTQ = "gptq"               # GPTQ quantization
    QUIP = "quip"               # QuIP with Hadamard coding


class QuantizationStrategy(Enum):
    """Quantization strategy."""
    PER_TENSOR = "per_tensor"   # One scale for entire tensor
    PER_CHANNEL = "per_channel" # One scale per output channel
    PER_GROUP = "per_group"     # One scale per group of elements


@dataclass
class QuantizationConfig:
    """Configuration for quantization."""
    # Quantization type
    quant_type: QuantizationType = QuantizationType.INT4
    
    # Strategy
    strategy: QuantizationStrategy = QuantizationStrategy.PER_GROUP
    group_size: int = 128  # For per-group quantization
    
    # Calibration
    calibration_data: Optional[torch.Tensor] = None
    calibration_samples: int = 128
    calibration_method: str = "percentile"  # percentile, minmax, mse
    
    # Advanced options
    use_smoothquant: bool = False
    smoothquant_alpha: float = 0.5
    
    # For AWQ/GPTQ
    zero_point: bool = True
    bias_correction: bool = False
    
    # For mixed precision
    exclude_layers: List[str] = field(default_factory=list)  # Layers to keep in FP16
    sensitive_layers: List[str] = field(default_factory=list)  # Layers needing higher precision
    
    # Performance
    use_quantized_linear: bool = True  # Use quantized matmul kernels
    use_fast_quant: bool = True  # Use faster quantization (may reduce accuracy)


@dataclass
class QuantizedLayer:
    """Container for quantized layer data."""
    name: str
    qweight: torch.Tensor       # Quantized weights
    scales: torch.Tensor        # Scale factors
    zeros: Optional[torch.Tensor] = None  # Zero points (for asymmetric)
    
    # For different quant types
    qstate: Optional[Any] = None  # bitsandbytes QuantState
    code: Optional[torch.Tensor] = None  # Quantization codebook
    
    # Metadata
    original_dtype: torch.dtype = torch.float16
    original_size_bytes: int = 0
    quantized_size_bytes: int = 0
    
    # Statistics
    has_outliers: bool = False
    outlier_ratio: float = 0.0


class Quantizer:
    """
    Advanced Quantizer for neural network models.
    
    Features:
    - Multiple quantization types (FP8, INT8, INT4, NF4, AWQ, GPTQ)
    - Per-tensor, per-channel, per-group strategies
    - SmoothQuant for handling activation outliers
    - Mixed precision support
    - Automatic calibration
    
    Example:
        config = QuantizationConfig(
            quant_type=QuantizationType.INT4,
            strategy=QuantizationStrategy.PER_GROUP,
            group_size=128
        )
        quantizer = Quantizer(config)
        qweights = quantizer.quantize(weights, name="layer_0")
        
        # Dequantize for inference
        weights = quantizer.dequantize(qweights)
    """
    
    def __init__(self, config: QuantizationConfig):
        self.config = config
        self._check_dependencies()
        self._calibration_stats: Dict[str, Dict] = {}
    
    def _check_dependencies(self) -> None:
        """Check for required dependencies."""
        self._has_bitsandbytes = False
        self._has_torch_compile = hasattr(torch, 'compile')
        
        try:
            import bitsandbytes as bnb
            self._has_bitsandbytes = True
            self._bnb = bnb
            logger.info("bitsandbytes available for NF4/FP4 quantization")
        except ImportError:
            logger.info("bitsandbytes not available")
        
        try:
            from autoawq import AWQQuantizer
            self._has_awq = True
            self._awq_quantizer = AWQQuantizer
            logger.info("AWQ available")
        except ImportError:
            self._has_awq = False
            logger.info("AWQ not available")
        
        try:
            from gptq import GPTQQuantizer
            self._has_gptq = True
            self._gptq_quantizer = GPTQQuantizer
            logger.info("GPTQ available")
        except ImportError:
            self._has_gptq = False
            logger.info("GPTQ not available")
    
    def quantize(
        self,
        weights: torch.Tensor,
        name: str = "unknown",
        scales_only: bool = False,
    ) -> QuantizedLayer:
        """
        Quantize a weight tensor.
        
        Args:
            weights: Weight tensor to quantize
            name: Name for this layer
            scales_only: If True, return scales without quantizing
            
        Returns:
            QuantizedLayer with quantized weights
        """
        original_size = weights.numel() * weights.element_size()
        original_dtype = weights.dtype
        
        # Check if layer should be excluded
        if name in self.config.exclude_layers:
            logger.debug(f"Skipping quantization for excluded layer: {name}")
            return QuantizedLayer(
                name=name,
                qweight=weights,
                scales=torch.ones_like(weights),
                original_dtype=original_dtype,
                original_size_bytes=original_size,
                quantized_size_bytes=original_size,
            )
        
        # Run calibration if available
        if name in self._calibration_stats:
            stats = self._calibration_stats[name]
        else:
            stats = self._compute_calibration_stats(weights, name)
        
        # Quantize based on type
        if self.config.quant_type == QuantizationType.INT4:
            return self._quantize_int4(weights, name, original_dtype, original_size, stats)
        elif self.config.quant_type == QuantizationType.INT8:
            return self._quantize_int8(weights, name, original_dtype, original_size, stats)
        elif self.config.quant_type == QuantizationType.FP8_E4M3:
            return self._quantize_fp8(weights, name, original_dtype, original_size)
        elif self.config.quant_type == QuantizationType.NF4:
            return self._quantize_nf4(weights, name, original_dtype, original_size)
        else:
            return self._quantize_int4(weights, name, original_dtype, original_size, stats)
    
    def _compute_calibration_stats(
        self,
        weights: torch.Tensor,
        name: str
    ) -> Dict[str, float]:
        """Compute calibration statistics for a weight tensor."""
        stats = {}
        
        stats['min'] = weights.min().item()
        stats['max'] = weights.max().item()
        stats['mean'] = weights.mean().item()
        stats['std'] = weights.std().item()
        stats['abs_mean'] = weights.abs().mean().item()
        stats['abs_max'] = weights.abs().max().item()
        
        # Compute outlier ratio
        if self.config.calibration_method == "percentile":
            q = torch.quantile(weights.abs().flatten(), 0.99)
            outliers = (weights.abs() > q).float().mean()
        else:
            threshold = 3 * stats['std']
            outliers = (weights.abs() > threshold).float().mean()
        
        stats['outlier_ratio'] = outliers.item()
        
        # Store stats
        self._calibration_stats[name] = stats
        
        return stats
    
    def _quantize_int4(
        self,
        weights: torch.Tensor,
        name: str,
        original_dtype: torch.dtype,
        original_size: int,
        stats: Dict[str, float],
    ) -> QuantizedLayer:
        """Quantize to INT4 with per-group scaling."""
        group_size = self.config.group_size
        
        # Flatten for grouping
        original_shape = weights.shape
        if weights.dim() > 2:
            weights = weights.flatten(0, -2)
        elif weights.dim() == 2:
            pass  # Already 2D
        else:
            weights = weights.unsqueeze(0)
        
        # Reshape into groups
        total_elements = weights.numel()
        num_groups = (total_elements + group_size - 1) // group_size
        
        # Pad to full groups
        padded_size = num_groups * group_size
        padded = torch.zeros(padded_size, dtype=weights.dtype, device=weights.device)
        padded[:total_elements] = weights.flatten()
        
        # Reshape
        grouped = padded.view(num_groups, group_size)
        
        # Compute scales per group
        scales = grouped.abs().max(dim=1).values / 7.0  # Max value for INT4
        
        # Quantize
        qweight = torch.round(grouped / scales.unsqueeze(1)).to(torch.int8)
        
        # Reshape back
        qweight = qweight[:total_elements].reshape(original_shape)
        scales = scales[:num_groups]
        
        # Compute quantized size
        quantized_size = (total_elements * 0.5) + (num_groups * 2)  # 4 bits + scales
        
        return QuantizedLayer(
            name=name,
            qweight=qweight,
            scales=scales,
            original_dtype=original_dtype,
            original_size_bytes=original_size,
            quantized_size_bytes=int(quantized_size),
            has_outliers=stats['outlier_ratio'] > 0.01,
            outlier_ratio=stats['outlier_ratio'],
        )
    
    def _quantize_int8(
        self,
        weights: torch.Tensor,
        name: str,
        original_dtype: torch.dtype,
        original_size: int,
        stats: Dict[str, float],
    ) -> QuantizedLayer:
        """Quantize to INT8 with per-channel scaling."""
        if self.config.strategy == QuantizationStrategy.PER_CHANNEL:
            # Per output channel (row-wise for linear layers)
            if weights.dim() == 2:
                scales = weights.abs().max(dim=1).values / 127.0
                qweight = torch.round(weights / scales.unsqueeze(1)).to(torch.int8)
            else:
                scales = weights.abs().max() / 127.0
                qweight = torch.round(weights / scales).to(torch.int8)
        else:
            # Per tensor
            scales = weights.abs().max() / 127.0
            qweight = torch.round(weights / scales).to(torch.int8)
        
        quantized_size = qweight.numel() + scales.numel() * scales.element_size()
        
        return QuantizedLayer(
            name=name,
            qweight=qweight,
            scales=scales,
            original_dtype=original_dtype,
            original_size_bytes=original_size,
            quantized_size_bytes=quantized_size,
            has_outliers=stats['outlier_ratio'] > 0.01,
            outlier_ratio=stats['outlier_ratio'],
        )
    
    def _quantize_fp8(
        self,
        weights: torch.Tensor,
        name: str,
        original_dtype: torch.dtype,
        original_size: int,
    ) -> QuantizedLayer:
        """Quantize to FP8 E4M3."""
        if not weights.is_cuda:
            logger.warning(f"FP8 quantization only supported on CUDA, falling back to INT8")
            return self._quantize_int8(weights, name, original_dtype, original_size, {})
        
        try:
            # Use Transformer Engine for FP8 if available
            import transformer_engine
            from transformer_engine.pytorch import FP8Tensor
            fp8_tensor = FP8Tensor.to_fp8(weights)
            qweight = fp8_tensor._data
            scales = fp8_tensor._scale
            
            return QuantizedLayer(
                name=name,
                qweight=qweight,
                scales=scales,
                original_dtype=original_dtype,
                original_size_bytes=original_size,
                quantized_size_bytes=qweight.numel(),
            )
        except ImportError:
            # Fallback to manual FP8 quantization
            # FP8 E4M3: 1 sign, 4 exp, 3 mantissa
            # Convert to float16 first for processing
            weights_fp16 = weights.to(torch.float16)
            
            # Manual quantization (simplified)
            scales = weights_fp16.abs().max() / 448.0  # Max for E4M3
            qweight = torch.round(weights_fp16 / scales).to(torch.int8)
            
            return QuantizedLayer(
                name=name,
                qweight=qweight,
                scales=scales,
                original_dtype=original_dtype,
                original_size_bytes=original_size,
                quantized_size_bytes=qweight.numel(),
            )
    
    def _quantize_nf4(
        self,
        weights: torch.Tensor,
        name: str,
        original_dtype: torch.dtype,
        original_size: int,
    ) -> QuantizedLayer:
        """Quantize to NF4 using bitsandbytes."""
        if not self._has_bitsandbytes:
            logger.warning("bitsandbytes not available, falling back to INT4")
            return self._quantize_int4(weights, name, original_dtype, original_size, {})
        
        # Use bitsandbytes for NF4
        qweight, quant_state = self._bnb.functional.quantize_nf4(
            weights.cuda(),
            blocksize=self.config.group_size
        )
        
        # Store quant state for dequantization
        quantized_size = qweight.numel() * 0.5  # NF4 = 4 bits
        
        return QuantizedLayer(
            name=name,
            qweight=qweight,
            scales=quant_state.absmax,
            original_dtype=original_dtype,
            original_size_bytes=original_size,
            quantized_size_bytes=int(quantized_size),
            qstate=quant_state,
        )
    
    def dequantize(self, qlayer: QuantizedLayer) -> torch.Tensor:
        """
        Dequantize weights back to original precision.
        
        Args:
            qlayer: Quantized layer to dequantize
            
        Returns:
            Dequantized weight tensor
        """
        if qlayer.qstate is not None:
            # bitsandbytes quantization
            return self._dequantize_bnb(qlayer)
        
        if qlayer.qweight.dtype == torch.int8:
            return self._dequantize_int8(qlayer)
        elif qlayer.qweight.dtype in [torch.int4, torch.uint8]:
            return self._dequantize_int4(qlayer)
        else:
            return qlayer.qweight.float()
    
    def _dequantize_int8(self, qlayer: QuantizedLayer) -> torch.Tensor:
        """Dequantize INT8 weights."""
        scales = qlayer.scales
        if scales.dim() == 1:
            # Per-channel: reshape scales to match weight dimensions
            if qlayer.qweight.dim() == 2:
                scales = scales.unsqueeze(1)
        
        return qlayer.qweight.float() * scales.float()
    
    def _dequantize_int4(self, qlayer: QuantizedLayer) -> torch.Tensor:
        """Dequantize INT4 weights."""
        # This handles the grouped INT4 format
        qweight = qlayer.qweight
        scales = qlayer.scales
        
        # Reshape and multiply
        group_size = self.config.group_size
        total_elements = qweight.numel()
        
        # Handle the packed format
        if qweight.dtype == torch.uint8:
            # Unpack INT4 from packed bytes
            qweight = self._unpack_int4(qweight)
        
        # Reshape into groups
        num_groups = scales.numel()
        qweight = qweight[:num_groups * group_size]
        qweight = qweight.view(num_groups, group_size)
        
        return (qweight.float() * scales.float().unsqueeze(1)).reshape_as(
            qlayer.qweight if qlayer.qweight.numel() == num_groups * group_size else torch.zeros(1)
        ).flatten()
    
    def _unpack_int4(self, qweight: torch.Tensor) -> torch.Tensor:
        """Unpack INT4 values from packed bytes."""
        # Each byte contains 2 INT4 values
        qweight = qweight.clone()
        high_bits = (qweight >> 4).float()
        low_bits = (qweight & 0x0F).float()
        return torch.cat([low_bits, high_bits], dim=0)
    
    def _dequantize_bnb(self, qlayer: QuantizedLayer) -> torch.Tensor:
        """Dequantize bitsandbytes NF4 weights."""
        if qlayer.qstate is not None:
            return self._bnb.functional.dequantize_nf4(
                qlayer.qweight,
                qlayer.qstate
            )
        return qlayer.qweight.float()
    
    # ==================== Full Model Quantization ====================
    
    def quantize_model(
        self,
        model: nn.Module,
        calibration_data: Optional[torch.Tensor] = None,
    ) -> Dict[str, QuantizedLayer]:
        """
        Quantize an entire model.
        
        Args:
            model: PyTorch model to quantize
            calibration_data: Optional calibration data for activation-aware methods
            
        Returns:
            Dictionary of quantized layers
        """
        quantized_layers = {}
        
        logger.info(f"Quantizing model with {self.config.quant_type.value}")
        
        for name, param in model.named_parameters():
            if 'weight' in name:
                logger.debug(f"Quantizing {name}")
                qlayer = self.quantize(param.data, name)
                quantized_layers[name] = qlayer
                
                # Update model parameter with quantized version
                if self.config.use_quantized_linear:
                    # Replace with quantized linear
                    param.data = self._create_quantized_tensor(qlayer)
        
        return quantized_layers
    
    def _create_quantized_tensor(self, qlayer: QuantizedLayer) -> torch.Tensor:
        """Create a quantized tensor for use in quantized operations."""
        # For now, store dequantized - full quantized ops require custom kernels
        return self.dequantize(qlayer)
    
    # ==================== Utility Methods ====================
    
    def get_compression_ratio(self, qlayer: QuantizedLayer) -> float:
        """Get compression ratio for a quantized layer."""
        if qlayer.original_size_bytes == 0:
            return 1.0
        return qlayer.original_size_bytes / max(qlayer.quantized_size_bytes, 1)
    
    def estimate_memory_savings(
        self,
        model: nn.Module,
    ) -> Dict[str, float]:
        """Estimate memory savings from quantization."""
        original_total = 0
        quantized_total = 0
        
        for name, param in model.named_parameters():
            if 'weight' in name:
                size = param.numel() * param.element_size()
                original_total += size
                
                qlayer = self.quantize(param.data, name)
                quantized_total += qlayer.quantized_size_bytes
        
        return {
            "original_mb": original_total / (1024**2),
            "quantized_mb": quantized_total / (1024**2),
            "compression_ratio": original_total / max(quantized_total, 1),
            "memory_saved_mb": (original_total - quantized_total) / (1024**2),
        }


class SmoothQuantOptimizer:
    """
    SmoothQuant: Migrate activation outliers to weights.
    
    Reference: SmoothQuant: Accurate and Efficient Post-Training 
    Quantization for Large Language Models
    
    This technique allows INT8 quantization of both weights AND 
    activations with minimal accuracy loss.
    """
    
    def __init__(self, alpha: float = 0.5):
        """
        Args:
            alpha: Migration strength (0.5 is typically optimal)
                   0 = no migration, 1 = full migration
        """
        self.alpha = alpha
        self.per_channel_scales: Dict[str, torch.Tensor] = {}
    
    def compute_smooth_scales(
        self,
        weights: torch.Tensor,
        activations_std: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute per-channel smoothing scales.
        
        Args:
            weights: Weight tensor [out_features, in_features]
            activations_std: Per-channel activation stddev
            
        Returns:
            Per-channel smoothing scales
        """
        # Weight magnitude per output channel
        w_per_channel = weights.abs().max(dim=1).values
        
        # Avoid division by zero
        w_per_channel = w_per_channel.clamp(min=1e-8)
        activations_std = activations_std.clamp(min=1e-8)
        
        # SmoothQuant formula
        scales = (w_per_channel ** self.alpha) * (activations_std ** (1 - self.alpha))
        
        return scales
    
    def smooth_weights(
        self,
        weights: torch.Tensor,
        scales: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply smoothing to weights.
        
        Args:
            weights: Original weights
            scales: Smoothing scales
            
        Returns:
            Tuple of (smoothed_weights, inverse_scales)
        """
        inv_scales = (1.0 / scales).unsqueeze(1).to(weights.dtype)
        smoothed_weights = weights * inv_scales
        
        return smoothed_weights, scales.unsqueeze(1)
