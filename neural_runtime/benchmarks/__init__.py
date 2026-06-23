"""Benchmarks module - Scientific validation of compression methods."""

from .benchmark_runner import BenchmarkSuite, BenchmarkResult, run_benchmarks
from .compression_comparison import WeightMatrixBenchmark, run_compression_benchmark

__all__ = [
    "BenchmarkSuite",
    "BenchmarkResult",
    "run_benchmarks",
    "WeightMatrixBenchmark",
    "run_compression_benchmark",
]
