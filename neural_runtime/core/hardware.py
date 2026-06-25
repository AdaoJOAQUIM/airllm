"""
Hardware Detection and Configuration Module
Automatically detects available hardware and configures optimal settings.
"""

import os
import gc
import platform
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

import torch

logger = logging.getLogger(__name__)


class DeviceType(Enum):
    """Type of computation device."""
    CPU = "cpu"
    CUDA = "cuda"
    CUDA_MULTI = "cuda_multi"
    MPS = "mps"  # Apple Metal
    CPU_OFFLOAD = "cpu_offload"


class MemoryTier(Enum):
    """Memory tier hierarchy from fastest to slowest."""
    HBM = "hbm"        # GPU VRAM
    RAM = "ram"        # System RAM
    NVME = "nvme"      # NVMe SSD
    SATA = "sata"      # SATA SSD/HDD


@dataclass
class HardwareSpec:
    """Complete hardware specification."""
    device_type: DeviceType
    device_count: int = 1
    device_names: List[str] = field(default_factory=list)
    
    # Memory specs per device
    vram_per_device_gb: float = 0.0
    total_vram_gb: float = 0.0
    ram_gb: float = 0.0
    swap_gb: float = 0.0
    
    # Storage specs
    nvme_speed_gbs: float = 0.0  # Read speed GB/s
    sata_speed_gbs: float = 0.0
    
    # Compute specs
    cuda_cores: int = 0
    tensor_cores: int = 0
    compute_capability: Tuple[int, int] = (0, 0)
    
    # CPU specs
    cpu_cores: int = 0
    cpu_threads: int = 0
    cpu_memory_bandwidth_gbs: float = 0.0
    
    # Capabilities
    fp16_support: bool = False
    bf16_support: bool = False
    fp8_support: bool = False
    tensor_parallel_support: bool = False
    nvlink_support: bool = False
    
    # Optimal settings
    recommended_batch_size: int = 1
    recommended_sequence_length: int = 512
    recommended_precision: str = "fp16"
    recommended_offload_strategy: str = "layer"


class HardwareDetector:
    """
    Automatically detects and profiles hardware capabilities.
    
    Usage:
        spec = HardwareDetector.detect()
        print(f"Detected: {spec.device_type}, {spec.total_vram_gb}GB VRAM")
    """
    
    _cached_spec: Optional[HardwareSpec] = None
    
    @classmethod
    def detect(cls, verbose: bool = True) -> HardwareSpec:
        """Detect hardware and return specification."""
        if cls._cached_spec is not None:
            return cls._cached_spec
        
        spec = HardwareSpec(device_type=DeviceType.CPU)
        
        # Detect platform
        is_linux = platform.system() == "Linux"
        is_macos = platform.system() == "Darwin"
        is_windows = platform.system() == "Windows"
        
        # Detect CPU
        cls._detect_cpu(spec)
        
        # Detect GPU
        gpu_detected = cls._detect_gpu(spec)
        
        if gpu_detected:
            if spec.device_count > 1:
                spec.device_type = DeviceType.CUDA_MULTI
            else:
                spec.device_type = DeviceType.CUDA
        elif is_macos and torch.backends.mps.is_available():
            spec.device_type = DeviceType.MPS
        else:
            spec.device_type = DeviceType.CPU
        
        # Detect storage
        cls._detect_storage(spec)
        
        # Detect memory
        cls._detect_memory(spec)
        
        # Determine capabilities
        cls._detect_capabilities(spec)
        
        # Calculate optimal settings
        cls._calculate_optimal_settings(spec)
        
        cls._cached_spec = spec
        
        if verbose:
            cls._log_spec(spec)
        
        return spec
    
    @staticmethod
    def _detect_cpu(spec: HardwareSpec) -> None:
        """Detect CPU capabilities."""
        spec.cpu_cores = os.cpu_count() or 4
        spec.cpu_threads = spec.cpu_cores  # Usually
        
        # Try to get memory bandwidth (rough estimate)
        try:
            import psutil
            mem = psutil.virtual_memory()
            spec.ram_gb = mem.total / (1024**3)
            spec.swap_gb = mem.swap_total / (1024**3) if hasattr(mem, 'swap_total') else 0
            # Estimate bandwidth: ~50-100 GB/s for modern CPUs
            spec.cpu_memory_bandwidth_gbs = min(100, spec.cpu_cores * 10)
        except ImportError:
            import subprocess
            try:
                result = subprocess.run(
                    ["free", "-b"], capture_output=True, text=True
                )
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:
                    parts = lines[1].split()
                    if len(parts) > 1:
                        spec.ram_gb = int(parts[1]) / (1024**3)
            except:
                pass
    
    @staticmethod
    def _detect_gpu(spec: HardwareSpec) -> bool:
        """Detect GPU capabilities."""
        if not torch.cuda.is_available():
            return False
        
        spec.device_count = torch.cuda.device_count()
        spec.device_names = [torch.cuda.get_device_name(i) for i in range(spec.device_count)]
        
        # Get VRAM per device
        for i in range(spec.device_count):
            props = torch.cuda.get_device_properties(i)
            vram_gb = props.total_memory / (1024**3)
            spec.vram_per_device_gb = max(spec.vram_per_device_gb, vram_gb)
            spec.total_vram_gb += vram_gb
            spec.cuda_cores = props.multi_processor_count * 128  # Approximate
            spec.compute_capability = (props.major, props.minor)
            
            # Check for tensor cores (Ampere+)
            if props.major >= 8:
                spec.tensor_cores = props.multi_processor_count * 4
                spec.tensor_parallel_support = True
            
            # NVLink support (compute capability 8.0+)
            if props.major >= 8:
                spec.nvlink_support = True
        
        return True
    
    @staticmethod
    def _detect_storage(spec: HardwareSpec) -> None:
        """Detect storage speed."""
        import shutil
        
        # Try to measure read speed
        try:
            # Create temp file for benchmark
            import tempfile
            import time
            
            temp_dir = tempfile.gettempdir()
            test_file = os.path.join(temp_dir, 'speed_test.bin')
            
            # Write test
            size_mb = 100
            data = b'0' * (1024 * 1024)
            start = time.time()
            with open(test_file, 'wb') as f:
                for _ in range(size_mb):
                    f.write(data)
            write_time = time.time() - start
            
            # Read test
            start = time.time()
            with open(test_file, 'rb') as f:
                _ = f.read(size_mb * 1024 * 1024)
            read_time = time.time() - start
            
            # Clean up
            os.remove(test_file)
            
            read_speed = (size_mb / 1024) / read_time
            write_speed = (size_mb / 1024) / write_time
            
            # Classify
            if read_speed > 2:  # GB/s
                spec.nvme_speed_gbs = read_speed
            else:
                spec.sata_speed_gbs = read_speed
                
        except Exception:
            # Assume SSD if we can't measure
            spec.nvme_speed_gbs = 0.5  # Conservative estimate
    
    @staticmethod
    def _detect_memory(spec: HardwareSpec) -> None:
        """Detect system memory."""
        try:
            import psutil
            mem = psutil.virtual_memory()
            spec.ram_gb = mem.total / (1024**3)
        except:
            pass
    
    @staticmethod
    def _detect_capabilities(spec: HardwareSpec) -> None:
        """Detect compute capabilities."""
        if spec.device_type in [DeviceType.CUDA, DeviceType.CUDA_MULTI]:
            spec.fp16_support = True
            spec.bf16_support = spec.compute_capability >= (8, 0)
            spec.fp8_support = spec.compute_capability >= (8, 9)
        elif spec.device_type == DeviceType.MPS:
            spec.fp16_support = True
            spec.bf16_support = True
        else:
            spec.fp16_support = True
    
    @staticmethod
    def _calculate_optimal_settings(spec: HardwareSpec) -> None:
        """Calculate optimal runtime settings based on hardware."""
        
        # Recommended precision
        if spec.fp8_support:
            spec.recommended_precision = "fp8"
        elif spec.bf16_support:
            spec.recommended_precision = "bf16"
        else:
            spec.recommended_precision = "fp16"
        
        # Batch size based on VRAM
        if spec.total_vram_gb >= 80:
            spec.recommended_batch_size = 8
        elif spec.total_vram_gb >= 40:
            spec.recommended_batch_size = 4
        elif spec.total_vram_gb >= 24:
            spec.recommended_batch_size = 2
        elif spec.total_vram_gb >= 16:
            spec.recommended_batch_size = 1
        else:
            spec.recommended_batch_size = 1
        
        # Sequence length based on VRAM
        if spec.total_vram_gb >= 40:
            spec.recommended_sequence_length = 4096
        elif spec.total_vram_gb >= 24:
            spec.recommended_sequence_length = 2048
        elif spec.total_vram_gb >= 16:
            spec.recommended_sequence_length = 1024
        else:
            spec.recommended_sequence_length = 512
        
        # Offload strategy
        if spec.total_vram_gb >= 80:
            spec.recommended_offload_strategy = "none"  # Full model fits
        elif spec.total_vram_gb >= 40:
            spec.recommended_offload_strategy = "attention"  # Keep attention in VRAM
        elif spec.total_vram_gb >= 16:
            spec.recommended_offload_strategy = "layer"  # AirLLM strategy
        else:
            spec.recommended_offload_strategy = "aggressive"  # Minimal VRAM
    
    @staticmethod
    def _log_spec(spec: HardwareSpec) -> None:
        """Log hardware specification."""
        logger.info("=" * 50)
        logger.info("HARDWARE DETECTION RESULTS")
        logger.info("=" * 50)
        logger.info(f"Device Type: {spec.device_type.value}")
        logger.info(f"Device Count: {spec.device_count}")
        if spec.device_names:
            for i, name in enumerate(spec.device_names):
                logger.info(f"  GPU {i}: {name}")
        logger.info(f"Total VRAM: {spec.total_vram_gb:.1f} GB")
        logger.info(f"System RAM: {spec.ram_gb:.1f} GB")
        logger.info(f"Compute Capability: {spec.compute_capability[0]}.{spec.compute_capability[1]}")
        logger.info(f"Recommended Precision: {spec.recommended_precision}")
        logger.info(f"Recommended Batch Size: {spec.recommended_batch_size}")
        logger.info(f"Recommended Seq Length: {spec.recommended_sequence_length}")
        logger.info("=" * 50)
    
    @classmethod
    def get_memory_info(cls) -> Dict[str, float]:
        """Get current memory usage information."""
        info = {}
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                mem_allocated = torch.cuda.memory_allocated(i) / (1024**3)
                mem_reserved = torch.cuda.memory_reserved(i) / (1024**3)
                mem_total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                info[f"cuda:{i}"] = {
                    "allocated_gb": mem_allocated,
                    "reserved_gb": mem_reserved,
                    "total_gb": mem_total,
                    "free_gb": mem_total - mem_reserved,
                }
        
        try:
            import psutil
            mem = psutil.virtual_memory()
            info["ram"] = {
                "total_gb": mem.total / (1024**3),
                "available_gb": mem.available / (1024**3),
                "used_gb": mem.used / (1024**3),
                "percent": mem.percent,
            }
        except ImportError:
            pass
        
        return info


def get_optimal_dtype(precision: str = "auto") -> torch.dtype:
    """Get optimal torch dtype based on precision setting."""
    if precision == "auto":
        spec = HardwareDetector.detect(verbose=False)
        precision = spec.recommended_precision
    
    dtype_map = {
        "fp8": torch.float8_e4m3fn,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    
    return dtype_map.get(precision, torch.float16)


def estimate_model_memory(
    num_parameters: int,
    precision: str = "fp16",
    include_kv_cache: bool = True,
    batch_size: int = 1,
    sequence_length: int = 512,
    num_layers: int = 1,
    hidden_size: int = 1,
    num_heads: int = 1,
) -> Dict[str, float]:
    """
    Estimate memory requirements for a model.
    
    Args:
        num_parameters: Total number of parameters
        precision: Data type (fp16, fp8, int8, int4)
        include_kv_cache: Include KV cache memory
        batch_size: Batch size
        sequence_length: Sequence length
        num_layers: Number of layers
        hidden_size: Hidden size
        num_heads: Number of attention heads
        
    Returns:
        Dictionary with memory breakdown in GB
    """
    # Parameter memory
    precision_bytes = {
        "fp32": 4,
        "fp16": 2,
        "bf16": 2,
        "fp8": 1,
        "int8": 1,
        "int4": 0.5,
    }
    
    bytes_per_param = precision_bytes.get(precision, 2)
    model_memory_gb = (num_parameters * bytes_per_param) / (1024**3)
    
    # KV cache memory (per token per layer)
    kv_bytes = bytes_per_param * 2 * hidden_size  # k and v
    kv_memory_per_layer = (batch_size * sequence_length * kv_bytes) / (1024**3)
    total_kv_memory = kv_memory_per_layer * num_layers if include_kv_cache else 0
    
    # Activation memory (rough estimate)
    activation_memory = 0.1 * num_layers * batch_size * sequence_length / (1024**3)
    
    return {
        "model_weights_gb": model_memory_gb,
        "kv_cache_gb": total_kv_memory,
        "activations_gb": activation_memory,
        "total_gb": model_memory_gb + total_kv_memory + activation_memory,
        "precision": precision,
        "bytes_per_param": bytes_per_param,
    }
