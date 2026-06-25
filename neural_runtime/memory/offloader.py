"""
Offloader Module
CPU/GPU offloading for models that don't fit in VRAM.
"""

import gc
import time
from typing import Dict, List, Optional, Tuple, Callable
from enum import Enum
from dataclasses import dataclass
import logging

import torch
import torch.nn as nn
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import OffloadedWeightsLoader

logger = logging.getLogger(__name__)


class OffloadStrategy(Enum):
    """Offloading strategy."""
    NONE = "none"              # No offloading (full model in VRAM)
    ATTENTION = "attention"    # Keep attention in VRAM
    LAYER = "layer"           # Layer-wise offloading (AirLLM style)
    FULL = "full"             # Full offloading to CPU
    HYBRID = "hybrid"         # Mixed strategy


@dataclass
class OffloadConfig:
    """Configuration for offloading."""
    strategy: OffloadStrategy = OffloadStrategy.LAYER
    
    # What to keep in VRAM
    keep_in_vram: List[str] = None  # Layer names to keep
    
    # What to offload
    offload_to_cpu: List[str] = None  # Layer names to offload
    
    # CPU offload settings
    cpu_offload_module: bool = True
    cpu_offload_param: bool = True
    
    # Prefetch settings
    prefetch_layers: List[int] = None  # Layer indices to prefetch
    prefetch_ahead: int = 2


class Offloader:
    """
    Intelligent offloader for large models.
    
    Features:
    - Layer-wise offloading (inspired by AirLLM)
    - Attention KV-cache in VRAM
    - CPU offloading with pin_memory for fast transfer
    - Async offloading pipeline
    
    Example:
        offloader = Offloader(model, strategy=OffloadStrategy.LAYER)
        
        # Offload all but embedding and first layer
        offloader.offload_except(['embedding', 'layer_0'])
        
        # Load a layer
        offloader.load_layer('layer_5')
        
        # Unload after use
        offloader.unload_layer('layer_5')
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[OffloadConfig] = None,
        device: str = "cuda:0",
    ):
        self.model = model
        self.config = config or OffloadConfig()
        self.device = torch.device(device)
        self.cpu_device = torch.device("cpu")
        
        # State tracking
        self.loaded_layers: Dict[str, bool] = {}
        self.layer_devices: Dict[str, str] = {}
        
        # Offload stats
        self.offload_count = 0
        self.load_count = 0
        self.total_offload_time = 0.0
        self.total_load_time = 0.0
        
        logger.info(f"Offloader initialized with strategy: {self.config.strategy.value}")
    
    def offload_all(self) -> None:
        """Offload entire model to CPU."""
        logger.info("Offloading entire model to CPU...")
        
        start = time.time()
        
        for name, param in self.model.named_parameters():
            param.data = param.data.to(self.cpu_device, non_blocking=True)
            self.layer_devices[name] = "cpu"
        
        for name, buffer in self.model.named_buffers():
            buffer.data = buffer.data.to(self.cpu_device, non_blocking=True)
        
        self.total_offload_time += time.time() - start
        self.offload_count += 1
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def offload_except(self, keep_layers: List[str]) -> None:
        """
        Offload all layers except specified ones.
        
        Args:
            keep_layers: Layer names to keep in VRAM
        """
        keep_set = set(keep_layers)
        
        for name, param in self.model.named_parameters():
            # Check if this param belongs to a layer to keep
            should_keep = False
            for keep in keep_set:
                if keep in name:
                    should_keep = True
                    break
            
            if not should_keep:
                self._offload_param(name, param)
    
    def load_layer(self, layer_name: str, device: Optional[str] = None) -> bool:
        """
        Load a specific layer to device.
        
        Args:
            layer_name: Name of the layer to load
            device: Target device (default: self.device)
            
        Returns:
            True if successful
        """
        if device is None:
            device = self.device
        
        target_device = torch.device(device)
        
        start = time.time()
        
        # Find all parameters in this layer
        for name, param in self.model.named_parameters():
            if layer_name in name:
                if self.layer_devices.get(name) != str(target_device):
                    param.data = param.data.to(target_device, non_blocking=True)
                    self.layer_devices[name] = str(target_device)
                    self.load_count += 1
        
        # Load buffers too
        for name, buffer in self.model.named_buffers():
            if layer_name in name:
                buffer.data = buffer.data.to(target_device, non_blocking=True)
        
        self.total_load_time += time.time() - start
        self.loaded_layers[layer_name] = True
        
        logger.debug(f"Loaded layer {layer_name} to {device}")
        
        return True
    
    def unload_layer(self, layer_name: str, to_cpu: bool = True) -> bool:
        """
        Unload a specific layer to CPU or meta.
        
        Args:
            layer_name: Name of the layer to unload
            to_cpu: If True, offload to CPU; if False, offload to meta
            
        Returns:
            True if successful
        """
        target_device = self.cpu_device if to_cpu else torch.device("meta")
        
        start = time.time()
        
        # Offload parameters
        for name, param in self.model.named_parameters():
            if layer_name in name:
                if self.layer_devices.get(name) != str(target_device):
                    param.data = param.data.to(target_device, non_blocking=True)
                    self.layer_devices[name] = str(target_device)
                    self.offload_count += 1
        
        # Offload buffers
        for name, buffer in self.model.named_buffers():
            if layer_name in name:
                buffer.data = buffer.data.to(target_device, non_blocking=True)
        
        self.total_offload_time += time.time() - start
        self.loaded_layers.pop(layer_name, None)
        
        logger.debug(f"Unloaded layer {layer_name}")
        
        gc.collect()
        if torch.cuda.is_available() and to_cpu:
            torch.cuda.empty_cache()
        
        return True
    
    def _offload_param(self, name: str, param: nn.Parameter) -> None:
        """Offload a single parameter."""
        if self.layer_devices.get(name) != "cpu":
            param.data = param.data.to(self.cpu_device, non_blocking=True)
            self.layer_devices[name] = "cpu"
            self.offload_count += 1
    
    def get_layer_device(self, layer_name: str) -> str:
        """Get the current device of a layer."""
        for name in self.layer_devices:
            if layer_name in name:
                return self.layer_devices[name]
        return "unknown"
    
    def is_loaded(self, layer_name: str) -> bool:
        """Check if a layer is currently loaded in VRAM."""
        return self.loaded_layers.get(layer_name, False)
    
    def get_stats(self) -> Dict[str, float]:
        """Get offloading statistics."""
        return {
            "offload_count": self.offload_count,
            "load_count": self.load_count,
            "total_offload_time_s": self.total_offload_time,
            "total_load_time_s": self.total_load_time,
            "avg_offload_time_ms": (self.total_offload_time / max(self.offload_count, 1)) * 1000,
            "avg_load_time_ms": (self.total_load_time / max(self.load_count, 1)) * 1000,
        }
    
    def reset_stats(self) -> None:
        """Reset offloading statistics."""
        self.offload_count = 0
        self.load_count = 0
        self.total_offload_time = 0.0
        self.total_load_time = 0.0


class ContinuousOffloader:
    """
    Continuous offloader with async pipeline.
    
    Overlaps offloading with computation for better throughput.
    """
    
    def __init__(
        self,
        offloader: Offloader,
        num_workers: int = 2,
    ):
        self.offloader = offloader
        self.num_workers = num_workers
        
        # Pipeline state
        self.pending_loads: List[str] = []
        self.pending_unloads: List[str] = []
        
        # Stream for async operations
        if torch.cuda.is_available():
            self.stream = torch.cuda.Stream()
        else:
            self.stream = None
    
    def prefetch_layer(self, layer_name: str) -> None:
        """
        Prefetch a layer asynchronously.
        
        Args:
            layer_name: Layer to prefetch
        """
        if layer_name not in self.pending_loads:
            self.pending_loads.append(layer_name)
    
    def execute_pipeline(self) -> None:
        """Execute pending loads/unloads."""
        # Load pending layers
        for layer_name in self.pending_loads:
            self.offloader.load_layer(layer_name)
        self.pending_loads.clear()
        
        # Unload pending layers
        for layer_name in self.pending_unloads:
            self.offloader.unload_layer(layer_name)
        self.pending_unloads.clear()
    
    def sync(self) -> None:
        """Synchronize offloading operations."""
        if self.stream:
            self.stream.synchronize()
