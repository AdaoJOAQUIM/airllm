"""
System Breaker
============

Practical tests to find actual failure modes and breaking points.

This module runs real experiments to break the system.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any, Callable
from dataclasses import dataclass
import gc
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class FailureCase:
    """A case where the system failed."""
    test_name: str
    failure_mode: str
    conditions: Dict[str, Any]
    error_message: str
    stack_trace: str = ""
    recoverable: bool = False
    
    def to_dict(self) -> Dict:
        return {
            "test_name": self.test_name,
            "failure_mode": self.failure_mode,
            "conditions": self.conditions,
            "error_message": self.error_message,
            "recoverable": self.recoverable,
        }


class SystemBreaker:
    """
    The breaker - finds how the system actually fails.
    
    Unlike the theoretical critic, this runs REAL experiments.
    """
    
    def __init__(self):
        self.failures: List[FailureCase] = []
    
    def break_fractal_compression(self) -> List[FailureCase]:
        """Try to break fractal compression."""
        from ..representation.fractal import FractalCompressor
        
        failures = []
        
        # Test 1: Random weights (no structure)
        try:
            torch.manual_seed(42)
            random_weights = torch.randn(2048, 2048)
            
            compressor = FractalCompressor(similarity_threshold=0.7)
            compressed = compressor.compress(random_weights)
            
            # Check compression ratio
            if compressed.get("compression_ratio", 0) < 1.5:
                failures.append(FailureCase(
                    test_name="fractal_random_weights",
                    failure_mode="no_compression",
                    conditions={"similarity_threshold": 0.7, "weight_type": "random"},
                    error_message=f"Compression ratio only {compressed.get('compression_ratio', 0):.2f}x",
                    recoverable=True,
                ))
        except Exception as e:
            failures.append(FailureCase(
                test_name="fractal_random_weights",
                failure_mode="exception",
                conditions={"similarity_threshold": 0.7},
                error_message=str(e),
                recoverable=False,
            ))
        
        # Test 2: Extreme sizes
        for size in [128, 8192]:
            try:
                matrix = torch.randn(size, size)
                compressor = FractalCompressor()
                compressed = compressor.compress(matrix)
                reconstructed = compressor.decompress(compressed)
                
                # Check reconstruction
                error = (matrix - reconstructed).abs().mean().item()
                if error > 0.5:
                    failures.append(FailureCase(
                        test_name=f"fractal_size_{size}",
                        failure_mode="high_error",
                        conditions={"size": size},
                        error_message=f"Reconstruction error {error:.4f}",
                        recoverable=True,
                    ))
            except Exception as e:
                failures.append(FailureCase(
                    test_name=f"fractal_size_{size}",
                    failure_mode="exception",
                    conditions={"size": size},
                    error_message=str(e),
                    recoverable=True,  # Can continue
                ))
        
        # Test 3: Non-square matrices
        try:
            matrix = torch.randn(1024, 4096)
            compressor = FractalCompressor()
            compressed = compressor.compress(matrix)
            reconstructed = compressor.decompress(compressed)
            
            if reconstructed.shape != matrix.shape:
                failures.append(FailureCase(
                    test_name="fractal_nonsquare",
                    failure_mode="shape_mismatch",
                    conditions={"shape": (1024, 4096)},
                    error_message=f"Expected {matrix.shape}, got {reconstructed.shape}",
                    recoverable=False,
                ))
        except Exception as e:
            failures.append(FailureCase(
                test_name="fractal_nonsquare",
                failure_mode="exception",
                conditions={"shape": (1024, 4096)},
                error_message=str(e),
                recoverable=True,
            ))
        
        self.failures.extend(failures)
        return failures
    
    def break_neural_codec(self) -> List[FailureCase]:
        """Try to break neural codec."""
        from ..representation.codec import NeuralCodec
        
        failures = []
        
        # Test 1: Training instability
        try:
            codec = NeuralCodec((1024, 1024))
            original = torch.randn(1024, 1024)
            
            # Try to compress and decompress
            compressed = codec.encode(original)
            reconstructed = codec.decode(compressed)
            
            # Check error
            error = (original - reconstructed).abs().mean().item()
            if error > 0.3:
                failures.append(FailureCase(
                    test_name="neural_codec_high_error",
                    failure_mode="reconstruction_error",
                    conditions={"shape": (1024, 1024)},
                    error_message=f"Reconstruction error {error:.4f}",
                    recoverable=True,
                ))
        except Exception as e:
            failures.append(FailureCase(
                test_name="neural_codec",
                failure_mode="exception",
                conditions={},
                error_message=str(e),
                recoverable=False,
            ))
        
        # Test 2: Large matrices
        for size in [4096, 8192]:
            try:
                matrix = torch.randn(size, size)
                codec = NeuralCodec((size, size))
                compressed = codec.encode(matrix)
                reconstructed = codec.decode(compressed)
                
                # This will likely OOM on limited systems
                error = (matrix - reconstructed).abs().mean().item()
                logger.info(f"Large matrix {size}x{size}: error = {error:.4f}")
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    failures.append(FailureCase(
                        test_name=f"neural_codec_oom_{size}",
                        failure_mode="out_of_memory",
                        conditions={"size": size},
                        error_message=str(e),
                        recoverable=False,
                    ))
                else:
                    failures.append(FailureCase(
                        test_name=f"neural_codec_{size}",
                        failure_mode="runtime_error",
                        conditions={"size": size},
                        error_message=str(e),
                        recoverable=True,
                    ))
        
        self.failures.extend(failures)
        return failures
    
    def break_expert_generation(self) -> List[FailureCase]:
        """Try to break expert generation."""
        from ..experts.synthesizer import DynamicExpertSynthesizer
        
        failures = []
        
        # Test 1: Same seed reproducibility
        try:
            synthesizer = DynamicExpertSynthesizer(
                expert_dim=512,
                hidden_dim=128,
                max_experts=8,
            )
            
            # Generate with same seed twice
            weights1 = synthesizer.generator.generate(expert_id=0, seed=42)
            weights2 = synthesizer.generator.generate(expert_id=0, seed=42)
            
            # Check reproducibility
            for name in weights1.keys():
                diff = (weights1[name] - weights2[name]).abs().mean().item()
                if diff > 1e-5:
                    failures.append(FailureCase(
                        test_name="expert_reproducibility",
                        failure_mode="non_deterministic",
                        conditions={"seed": 42},
                        error_message=f"Weight {name} differs: {diff:.6f}",
                        recoverable=False,
                    ))
                    break
            
            # Check different seeds produce different weights
            weights3 = synthesizer.generator.generate(expert_id=0, seed=123)
            diff = (weights1["gate_proj.weight"] - weights3["gate_proj.weight"]).abs().mean().item()
            if diff < 1e-4:
                failures.append(FailureCase(
                    test_name="expert_diversity",
                    failure_mode="same_output_different_seed",
                    conditions={"seed1": 42, "seed2": 123},
                    error_message=f"Weights identical: diff={diff:.8f}",
                    recoverable=False,
                ))
                
        except Exception as e:
            failures.append(FailureCase(
                test_name="expert_generation",
                failure_mode="exception",
                conditions={},
                error_message=str(e),
                recoverable=False,
            ))
        
        # Test 2: Out-of-range expert ID
        try:
            weights = synthesizer.generator.generate(expert_id=999)
            # This should still work due to modulo, but let's see
        except Exception as e:
            failures.append(FailureCase(
                test_name="expert_oob_id",
                failure_mode="exception",
                conditions={"expert_id": 999, "max": 8},
                error_message=str(e),
                recoverable=True,
            ))
        
        self.failures.extend(failures)
        return failures
    
    def break_hierarchical_memory(self) -> List[FailureCase]:
        """Try to break hierarchical memory."""
        from ..memory.hierarchy import HierarchicalMemory
        
        failures = []
        
        # Test 1: Memory exhaustion
        for limit_gb in [0.5, 0.1, 0.01]:
            try:
                memory = HierarchicalMemory(
                    max_vram_gb=limit_gb,
                    max_ram_gb=limit_gb * 2,
                )
                
                # Try to store more than limit
                for i in range(1000):
                    tensor = torch.randn(4096, 4096)  # 64MB each
                    try:
                        memory.store(f"layer_{i}", tensor)
                    except Exception:
                        failures.append(FailureCase(
                            test_name=f"memory_exhaustion_{limit_gb}g",
                            failure_mode="storage_failed",
                            conditions={"limit_gb": limit_gb, "layers_stored": i},
                            error_message=f"Failed to store layer {i}",
                            recoverable=True,
                        ))
                        break
                
            except Exception as e:
                failures.append(FailureCase(
                    test_name=f"memory_limit_{limit_gb}g",
                    failure_mode="exception",
                    conditions={"limit_gb": limit_gb},
                    error_message=str(e),
                    recoverable=False,
                ))
        
        self.failures.extend(failures)
        return failures
    
    def break_parameter_virtualizer(self) -> List[FailureCase]:
        """Try to break parameter virtualizer."""
        from ..parameter_virt.virtualizer import ParameterVirtualizer
        
        failures = []
        
        # Create a dummy model
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(512, 512)
        
        try:
            model = DummyModel()
            virtualizer = ParameterVirtualizer(model, target_active_ratio=0.1)
            
            # Test 1: Access all parameters
            for name in list(model.state_dict().keys()):
                param = virtualizer.get_parameter(name)
                if param is None:
                    failures.append(FailureCase(
                        test_name="virtualizer_get_none",
                        failure_mode="parameter_not_found",
                        conditions={"param": name},
                        error_message="get_parameter returned None",
                        recoverable=True,
                    ))
            
            # Test 2: Virtualize aggressively
            virtualizer.virtualize(target_active_ratio=0.01)
            
            # Access all - some should be approximated
            errors = []
            for name, param in model.named_parameters():
                retrieved = virtualizer.get_parameter(name)
                if retrieved is not None:
                    diff = (param.data - retrieved).abs().mean().item()
                    errors.append(diff)
            
            if errors:
                avg_error = sum(errors) / len(errors)
                if avg_error > 0.5:
                    failures.append(FailureCase(
                        test_name="virtualizer_high_error",
                        failure_mode="approximation_error",
                        conditions={"active_ratio": 0.01, "avg_error": avg_error},
                        error_message=f"Average approximation error {avg_error:.4f}",
                        recoverable=True,
                    ))
                
        except Exception as e:
            failures.append(FailureCase(
                test_name="virtualizer",
                failure_mode="exception",
                conditions={},
                error_message=str(e),
                recoverable=False,
            ))
        
        self.failures.extend(failures)
        return failures
    
    def run_all_breaker_tests(self) -> Dict[str, List[FailureCase]]:
        """Run all breaker tests."""
        results = {}
        
        tests = [
            ("fractal", self.break_fractal_compression),
            ("neural_codec", self.break_neural_codec),
            ("expert_generation", self.break_expert_generation),
            ("hierarchical_memory", self.break_hierarchical_memory),
            ("parameter_virtualizer", self.break_parameter_virtualizer),
        ]
        
        for name, test_fn in tests:
            logger.info(f"\n{'='*50}")
            logger.info(f"BREAKING: {name}")
            logger.info(f"{'='*50}")
            
            try:
                failures = test_fn()
                results[name] = failures
                logger.info(f"  Found {len(failures)} failures")
            except Exception as e:
                logger.error(f"  Breaker test crashed: {e}")
                results[name] = [FailureCase(
                    test_name=name,
                    failure_mode="breaker_crash",
                    conditions={},
                    error_message=str(e),
                    recoverable=False,
                )]
        
        return results
    
    def generate_breaker_report(self) -> str:
        """Generate breaker report."""
        report = []
        report.append("=" * 70)
        report.append("SYSTEM BREAKER REPORT")
        report.append("=" * 70)
        
        # Group by failure type
        by_mode = {}
        for failure in self.failures:
            if failure.failure_mode not in by_mode:
                by_mode[failure.failure_mode] = []
            by_mode[failure.failure_mode].append(failure)
        
        report.append(f"\nTotal failures found: {len(self.failures)}")
        
        report.append("\n--- FAILURES BY MODE ---")
        for mode, failures in sorted(by_mode.items(), key=lambda x: -len(x[1])):
            report.append(f"\n{mode.upper()} ({len(failures)}):")
            for f in failures[:3]:  # Show first 3
                report.append(f"  [{f.test_name}] {f.error_message[:50]}...")
        
        report.append("\n--- RECOVERY ANALYSIS ---")
        recoverable = sum(1 for f in self.failures if f.recoverable)
        unrecoverable = len(self.failures) - recoverable
        report.append(f"  Recoverable: {recoverable}")
        report.append(f"  Unrecoverable: {unrecoverable}")
        
        return "\n".join(report)


def run_breaker_tests():
    """Run all breaker tests."""
    logging.basicConfig(level=logging.INFO)
    
    breaker = SystemBreaker()
    results = breaker.run_all_breaker_tests()
    
    report = breaker.generate_breaker_report()
    print("\n" + report)
    
    return breaker
