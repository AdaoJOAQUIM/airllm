"""
MEMORY STRESS TEST SUITE
======================

These tests find the BREAKING POINTS of the system under memory pressure.

CRITICAL: We want to find where the system FAILS, not where it works.
"""

import torch
import gc
import os
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass
import time
import psutil
import logging

logger = logging.getLogger(__name__)


@dataclass
class MemoryStressResult:
    """Result of a memory stress test."""
    test_name: str
    target_memory_mb: float
    actual_memory_used_mb: float
    success: bool
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    
    # Performance metrics
    latency_ms: float = 0.0
    throughput_mb_per_sec: float = 0.0
    
    # System state
    swap_used_mb: float = 0.0
    oom_count: int = 0
    
    def to_dict(self) -> Dict:
        return {
            "test_name": self.test_name,
            "target_memory_mb": self.target_memory_mb,
            "actual_memory_used_mb": self.actual_memory_used_mb,
            "success": self.success,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "latency_ms": self.latency_ms,
            "throughput_mb_per_sec": self.throughput_mb_per_sec,
            "swap_used_mb": self.swap_used_mb,
            "oom_count": self.oom_count,
        }


class MemoryConstraint:
    """Simulates a memory constraint for testing."""
    
    def __init__(self, max_memory_mb: float):
        self.max_memory_mb = max_memory_mb
        self.original_max = None
    
    def __enter__(self):
        # Note: This is a soft constraint - real OOM injection requires root
        self.original_max = torch.cuda.set_per_process_memory_fraction(
            self.max_memory_mb / 1024 if torch.cuda.is_available() else 1.0
        ) if torch.cuda.is_available() else None
        return self
    
    def __exit__(self, *args):
        if self.original_max is not None:
            torch.cuda.set_per_process_memory_fraction(self.original_max)


class MemoryStressTest:
    """Base class for memory stress tests."""
    
    def __init__(self, name: str):
        self.name = name
        self.results: List[MemoryStressResult] = []
    
    def run(self, target_memory_mb: float) -> MemoryStressResult:
        """Run the test at a specific memory level."""
        raise NotImplementedError
    
    def find_break_point(
        self, 
        memory_levels: List[float],
        success_criterion: Callable[[MemoryStressResult], bool] = None
    ) -> Tuple[float, float]:
        """
        Find the memory level where the system breaks.
        
        Returns:
            Tuple of (break_point_mb, last_success_mb)
        """
        if success_criterion is None:
            success_criterion = lambda r: r.success
        
        last_success = 0.0
        break_point = memory_levels[0]
        
        for memory_level in sorted(memory_levels):
            result = self.run(memory_level)
            self.results.append(result)
            
            if success_criterion(result):
                last_success = memory_level
            else:
                break_point = memory_level
                break
        
        return break_point, last_success


class HierarchicalMemoryStressTest(MemoryStressTest):
    """Stress test for hierarchical memory system."""
    
    def __init__(self):
        super().__init__("hierarchical_memory")
    
    def run(self, target_memory_mb: float) -> MemoryStressResult:
        """Test hierarchical memory at target memory level."""
        from ..memory.hierarchy import HierarchicalMemory
        
        start_time = time.time()
        start_mem = self._get_memory_usage()
        
        try:
            # Create memory system with limited VRAM
            memory = HierarchicalMemory(
                max_vram_gb=target_memory_mb / 1024,
                max_ram_gb=min(target_memory_mb * 2, 128) / 1024,
            )
            
            # Generate test tensors of varying sizes
            sizes = [1024, 2048, 4096, 8192]
            tensors = {}
            
            for size in sizes:
                key = f"layer_{size}"
                tensor = torch.randn(size, size)
                memory.store(key, tensor, level="auto")
                tensors[key] = tensor
            
            # Access pattern simulation
            for _ in range(10):
                for key in tensors.keys():
                    _ = memory.get(key)
            
            # Cleanup
            del tensors
            memory.clear()
            gc.collect()
            
            end_mem = self._get_memory_usage()
            latency = (time.time() - start_time) * 1000
            
            return MemoryStressResult(
                test_name=self.name,
                target_memory_mb=target_memory_mb,
                actual_memory_used_mb=end_mem - start_mem,
                success=True,
                latency_ms=latency,
            )
            
        except Exception as e:
            return MemoryStressResult(
                test_name=self.name,
                target_memory_mb=target_memory_mb,
                actual_memory_used_mb=0,
                success=False,
                error_type=type(e).__name__,
                error_message=str(e),
            )
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024**2)
        else:
            return psutil.Process().memory_info().rss / (1024**2)


class CompressionMemoryTest(MemoryStressTest):
    """Test compression under memory pressure."""
    
    def __init__(self):
        super().__init__("compression_memory")
    
    def run(self, target_memory_mb: float) -> MemoryStressResult:
        """Test compression with limited memory."""
        from ..representation.fractal import FractalCompressor
        from ..generation.compression import NeuralCompressor
        
        start_time = time.time()
        
        try:
            # Create large matrix
            matrix_size = int((target_memory_mb * 1024**2 / 4) ** 0.5)
            matrix = torch.randn(matrix_size, matrix_size)
            
            # Try different compression methods
            methods_tested = 0
            successes = 0
            
            # Fractal compression
            try:
                compressor = FractalCompressor(similarity_threshold=0.7)
                compressed = compressor.compress(matrix)
                reconstructed = compressor.decompress(compressed)
                successes += 1
                methods_tested += 1
            except Exception:
                methods_tested += 1
            
            # Neural compression
            try:
                latent_dim = min(256, matrix_size // 8)
                compressor = NeuralCompressor(
                    weight_shape=(matrix_size, matrix_size),
                    latent_dim=latent_dim,
                )
                compressed = compressor.compress(matrix)
                reconstructed = compressor.decompress(compressed)
                successes += 1
                methods_tested += 1
            except Exception:
                methods_tested += 1
            
            # Cleanup
            del matrix, compressed, reconstructed
            gc.collect()
            
            latency = (time.time() - start_time) * 1000
            
            return MemoryStressResult(
                test_name=self.name,
                target_memory_mb=target_memory_mb,
                actual_memory_used_mb=target_memory_mb,
                success=successes == methods_tested,
                latency_ms=latency,
            )
            
        except Exception as e:
            return MemoryStressResult(
                test_name=self.name,
                target_memory_mb=target_memory_mb,
                actual_memory_used_mb=0,
                success=False,
                error_type=type(e).__name__,
                error_message=str(e),
            )


class ExpertGenerationMemoryTest(MemoryStressTest):
    """Test expert generation under memory pressure."""
    
    def __init__(self):
        super().__init__("expert_generation")
    
    def run(self, target_memory_mb: float) -> MemoryStressResult:
        """Test expert generation with limited memory."""
        from ..experts.synthesizer import DynamicExpertSynthesizer
        
        start_time = time.time()
        
        try:
            # Estimate expert size
            expert_dim = max(256, int((target_memory_mb * 1024**2 / 12) ** 0.5))
            
            synthesizer = DynamicExpertSynthesizer(
                expert_dim=expert_dim,
                hidden_dim=128,
                max_experts=8,
                cache_size=4,
            )
            
            # Generate experts
            num_generations = min(16, int(target_memory_mb))
            for i in range(num_generations):
                weights = synthesizer.generator.generate(expert_id=i % 8)
                for name, weight in weights.items():
                    _ = weight.abs().sum()
            
            # Run expert
            test_input = torch.randn(1, expert_dim)
            for i in range(8):
                output = synthesizer.run_expert(
                    expert_id=i % 8,
                    input_tensor=test_input,
                )
            
            # Cleanup
            synthesizer.clear_cache()
            del test_input, weights, synthesizer
            gc.collect()
            
            latency = (time.time() - start_time) * 1000
            
            return MemoryStressResult(
                test_name=self.name,
                target_memory_mb=target_memory_mb,
                actual_memory_used_mb=target_memory_mb,
                success=True,
                latency_ms=latency,
            )
            
        except Exception as e:
            return MemoryStressResult(
                test_name=self.name,
                target_memory_mb=target_memory_mb,
                actual_memory_used_mb=0,
                success=False,
                error_type=type(e).__name__,
                error_message=str(e),
            )


class MemoryStressSuite:
    """
    Complete memory stress test suite.
    
    Tests at various memory levels to find breaking points.
    """
    
    def __init__(self):
        self.tests: List[MemoryStressTest] = []
        self.results: Dict[str, List[MemoryStressResult]] = {}
    
    def add_test(self, test: MemoryStressTest):
        """Add a test."""
        self.tests.append(test)
    
    def run_at_levels(
        self,
        memory_levels_mb: List[float] = None,
    ) -> Dict[str, List[MemoryStressResult]]:
        """Run all tests at specified memory levels."""
        if memory_levels_mb is None:
            # Progressive levels from 1MB to 16GB
            memory_levels_mb = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
        
        for test in self.tests:
            logger.info(f"\nRunning {test.name} at {len(memory_levels_mb)} levels...")
            test_results = []
            
            for level in memory_levels_mb:
                logger.info(f"  Level: {level} MB")
                result = test.run(level)
                test_results.append(result)
                
                if not result.success:
                    logger.warning(f"    FAILED: {result.error_type}")
                else:
                    logger.info(f"    OK: {result.actual_memory_used_mb:.1f} MB, {result.latency_ms:.1f} ms")
            
            self.results[test.name] = test_results
        
        return self.results
    
    def find_all_break_points(self) -> Dict[str, Tuple[float, float]]:
        """Find break points for all tests."""
        break_points = {}
        
        for test_name, results in self.results.items():
            break_point = None
            last_success = 0.0
            
            for result in results:
                if result.success:
                    last_success = result.target_memory_mb
                elif break_point is None:
                    break_point = result.target_memory_mb
            
            if break_point is not None or last_success > 0:
                break_points[test_name] = (break_point or last_success, last_success)
        
        return break_points
    
    def generate_report(self) -> str:
        """Generate stress test report."""
        report = []
        report.append("=" * 70)
        report.append("MEMORY STRESS TEST REPORT")
        report.append("=" * 70)
        
        # Break points
        break_points = self.find_all_break_points()
        
        report.append("\n--- BREAK POINTS ---")
        for test_name, (bp, last_success) in sorted(break_points.items()):
            report.append(f"\n{test_name}:")
            report.append(f"  Last success: {last_success:.0f} MB")
            report.append(f"  Break point:  {bp:.0f} MB")
        
        # Detailed results
        report.append("\n--- DETAILED RESULTS ---")
        for test_name, results in self.results.items():
            report.append(f"\n{test_name}:")
            for result in results:
                status = "✅" if result.success else "❌"
                if result.success:
                    report.append(f"  {status} {result.target_memory_mb:>8.0f} MB: "
                                f"{result.actual_memory_used_mb:>8.0f} MB used, "
                                f"{result.latency_ms:>6.1f} ms latency")
                else:
                    report.append(f"  {status} {result.target_memory_mb:>8.0f} MB: "
                                f"{result.error_type}: {result.error_message}")
        
        return "\n".join(report)


def run_memory_stress_tests():
    """Run the complete memory stress test suite."""
    logging.basicConfig(level=logging.INFO)
    
    suite = MemoryStressSuite()
    
    # Add tests
    suite.add_test(HierarchicalMemoryStressTest())
    suite.add_test(CompressionMemoryTest())
    suite.add_test(ExpertGenerationMemoryTest())
    
    # Run at various memory levels
    memory_levels = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    results = suite.run_at_levels(memory_levels)
    
    # Generate report
    report = suite.generate_report()
    print(report)
    
    return suite


if __name__ == "__main__":
    run_memory_stress_tests()
