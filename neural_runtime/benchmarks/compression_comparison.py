"""
Compression Method Comparison Benchmark
=====================================

Compare different compression methods on the same weight matrix.
This provides direct comparison of:
- INT4 quantization
- Learned compression
- Hypernetwork generation
- Fractal compression

Results are saved for scientific analysis.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any
import time
import json
from dataclasses import dataclass


@dataclass
class CompressionResult:
    """Result of a single compression method."""
    method: str
    original_size_bytes: int
    compressed_size_bytes: int
    
    # Quality metrics
    mse: float
    cosine_similarity: float
    spectral_error: float
    
    # Performance
    compression_time_ms: float
    decompression_time_ms: float
    
    # Derived
    compression_ratio: float
    quality_score: float  # 0-1
    
    def to_dict(self) -> Dict:
        return {
            "method": self.method,
            "original_size_bytes": self.original_size_bytes,
            "compressed_size_bytes": self.compressed_size_bytes,
            "compression_ratio": self.compression_ratio,
            "mse": self.mse,
            "cosine_similarity": self.cosine_similarity,
            "spectral_error": self.spectral_error,
            "compression_time_ms": self.compression_time_ms,
            "decompression_time_ms": self.decompression_time_ms,
            "quality_score": self.quality_score,
        }


class WeightMatrixBenchmark:
    """
    Benchmark a single weight matrix with multiple compression methods.
    """
    
    def __init__(self, matrix: torch.Tensor, name: str = "unknown"):
        self.matrix = matrix
        self.name = name
        self.results: List[CompressionResult] = []
    
    def run_all(self) -> List[CompressionResult]:
        """Run all compression methods."""
        methods = [
            ("INT4_Quantization", self.compress_int4),
            ("INT8_Quantization", self.compress_int8),
            ("Learned_Compression", self.compress_learned),
            ("LowRank_SVD", self.compress_svd),
            ("Pruning", self.compress_pruning),
            ("Mixed_Precision", self.compress_mixed),
        ]
        
        for method_name, compress_fn in methods:
            try:
                result = compress_fn()
                self.results.append(result)
                print(f"  ✓ {method_name}: {result.compression_ratio:.1f}x compression, "
                      f"quality: {result.quality_score:.2%}")
            except Exception as e:
                print(f"  ✗ {method_name}: Failed - {e}")
        
        return self.results
    
    def compress_int4(self) -> CompressionResult:
        """INT4 quantization."""
        start = time.time()
        
        # Quantize to INT4 (4-bit)
        # Each value in 0-15 range
        max_val = self.matrix.abs().max()
        scale = 15.0 / max_val
        
        quantized = (self.matrix * scale).round().clamp(0, 15)
        
        # Store quantized + scale
        compressed = quantized.to(torch.uint8)  # 4 bits per value
        
        compress_time = (time.time() - start) * 1000
        
        # Decompress for comparison
        start = time.time()
        decompressed = (compressed.float() / scale)
        decomp_time = (time.time() - start) * 1000
        
        original_bytes = self.matrix.numel() * self.matrix.element_size()
        compressed_bytes = compressed.numel() * compressed.element_size() + 8  # +scale
        
        # Metrics
        mse = F.mse_loss(decompressed, self.matrix).item()
        cos_sim = F.cosine_similarity(
            self.matrix.flatten().unsqueeze(0),
            decompressed.flatten().unsqueeze(0)
        ).item()
        
        return CompressionResult(
            method="INT4_Quantization",
            original_size_bytes=original_bytes,
            compressed_size_bytes=compressed_bytes,
            mse=mse,
            cosine_similarity=cos_sim,
            spectral_error=self._spectral_error(self.matrix, decompressed),
            compression_time_ms=compress_time,
            decompression_time_ms=decomp_time,
            compression_ratio=original_bytes / compressed_bytes,
            quality_score=max(0, 1 - mse * 100),
        )
    
    def compress_int8(self) -> CompressionResult:
        """INT8 quantization."""
        start = time.time()
        
        max_val = self.matrix.abs().max()
        scale = 127.0 / max_val
        
        quantized = (self.matrix * scale).round().clamp(-128, 127).to(torch.int8)
        
        compress_time = (time.time() - start) * 1000
        
        # Decompress
        start = time.time()
        decompressed = quantized.float() / scale
        decomp_time = (time.time() - start) * 1000
        
        original_bytes = self.matrix.numel() * self.matrix.element_size()
        compressed_bytes = quantized.numel() * quantized.element_size() + 8
        
        mse = F.mse_loss(decompressed, self.matrix).item()
        cos_sim = F.cosine_similarity(
            self.matrix.flatten().unsqueeze(0),
            decompressed.flatten().unsqueeze(0)
        ).item()
        
        return CompressionResult(
            method="INT8_Quantization",
            original_size_bytes=original_bytes,
            compressed_size_bytes=compressed_bytes,
            mse=mse,
            cosine_similarity=cos_sim,
            spectral_error=self._spectral_error(self.matrix, decompressed),
            compression_time_ms=compress_time,
            decompression_time_ms=decomp_time,
            compression_ratio=original_bytes / compressed_bytes,
            quality_score=max(0, 1 - mse * 10),
        )
    
    def compress_learned(self) -> CompressionResult:
        """
        Learned compression using autoencoder.
        
        Note: This is a simplified version. Real implementation would train.
        """
        from ..generation.compression import NeuralCompressor
        
        start = time.time()
        
        h, w = self.matrix.shape
        latent_dim = min(256, (h * w) // 100)  # 100x compression target
        
        compressor = NeuralCompressor(
            weight_shape=(h, w),
            latent_dim=latent_dim,
        )
        
        # Compress
        latent = compressor.compress(self.matrix)
        compressed_size = latent.numel() * latent.element_size()
        
        compress_time = (time.time() - start) * 1000
        
        # Decompress
        start = time.time()
        decompressed = compressor.decompress(latent)
        decomp_time = (time.time() - start) * 1000
        
        original_bytes = h * w * self.matrix.element_size()
        
        mse = F.mse_loss(decompressed, self.matrix).item()
        cos_sim = F.cosine_similarity(
            self.matrix.flatten().unsqueeze(0),
            decompressed.flatten().unsqueeze(0)
        ).item()
        
        return CompressionResult(
            method="Learned_Compression",
            original_size_bytes=original_bytes,
            compressed_size_bytes=compressed_size,
            mse=mse,
            cosine_similarity=cos_sim,
            spectral_error=self._spectral_error(self.matrix, decompressed),
            compression_time_ms=compress_time,
            decompression_time_ms=decomp_time,
            compression_ratio=original_bytes / max(compressed_size, 1),
            quality_score=max(0, 1 - mse * 50),
        )
    
    def compress_svd(self) -> CompressionResult:
        """Low-rank approximation using SVD."""
        start = time.time()
        
        # SVD decomposition
        U, S, V = torch.svd(self.matrix)
        
        # Keep top-k singular values (rank reduction)
        h, w = self.matrix.shape
        rank = min(h, w)
        
        # Target: 10x compression
        k = rank // 10
        
        # Store compressed: U[:,k], S[:k], V[:,k]
        compressed_size = (U[:, :k].numel() + S[:k].numel() + V[:, :k].numel()) * 4
        
        compress_time = (time.time() - start) * 1000
        
        # Decompress
        start = time.time()
        decompressed = torch.mm(U[:, :k], torch.diag(S[:k]))
        decompressed = torch.mm(decompressed, V[:, :k].t())
        decomp_time = (time.time() - start) * 1000
        
        original_bytes = h * w * self.matrix.element_size()
        
        mse = F.mse_loss(decompressed, self.matrix).item()
        cos_sim = F.cosine_similarity(
            self.matrix.flatten().unsqueeze(0),
            decompressed.flatten().unsqueeze(0)
        ).item()
        
        return CompressionResult(
            method="LowRank_SVD",
            original_size_bytes=original_bytes,
            compressed_size_bytes=compressed_size,
            mse=mse,
            cosine_similarity=cos_sim,
            spectral_error=self._spectral_error(self.matrix, decompressed),
            compression_time_ms=compress_time,
            decompression_time_ms=decomp_time,
            compression_ratio=original_bytes / max(compressed_size, 1),
            quality_score=max(0, 1 - mse * 50),
        )
    
    def compress_pruning(self) -> CompressionResult:
        """Magnitude-based pruning."""
        start = time.time()
        
        # Prune 50% of smallest weights
        threshold = torch.quantile(self.matrix.abs().flatten(), 0.5)
        mask = (self.matrix.abs() > threshold).float()
        
        pruned = self.matrix * mask
        
        # Store as sparse representation
        indices = mask.nonzero(as_tuple=True)
        values = pruned[indices]
        
        compressed_size = values.numel() * values.element_size() + indices[0].numel() * 4 * 2
        
        compress_time = (time.time() - start) * 1000
        
        # Decompress
        start = time.time()
        decompressed = torch.zeros_like(self.matrix)
        decompressed[indices] = values
        decomp_time = (time.time() - start) * 1000
        
        original_bytes = self.matrix.numel() * self.matrix.element_size()
        
        mse = F.mse_loss(decompressed, self.matrix).item()
        cos_sim = F.cosine_similarity(
            self.matrix.flatten().unsqueeze(0),
            decompressed.flatten().unsqueeze(0)
        ).item()
        
        return CompressionResult(
            method="Pruning_50pct",
            original_size_bytes=original_bytes,
            compressed_size_bytes=compressed_size,
            mse=mse,
            cosine_similarity=cos_sim,
            spectral_error=self._spectral_error(self.matrix, decompressed),
            compression_time_ms=compress_time,
            decompression_time_ms=decomp_time,
            compression_ratio=original_bytes / max(compressed_size, 1),
            quality_score=max(0, 1 - mse * 50),
        )
    
    def compress_mixed(self) -> CompressionResult:
        """
        Mixed precision: FP16 for important, INT4 for others.
        """
        start = time.time()
        
        # Importance based on gradient magnitude (simulated)
        importance = self.matrix.abs()
        
        # Top 20% in FP16, rest in INT4
        threshold = torch.quantile(importance.flatten(), 0.8)
        
        fp16_mask = importance > threshold
        int4_mask = ~fp16_mask
        
        # FP16 part
        fp16_part = self.matrix * fp16_mask.float()
        
        # INT4 part
        int4_part = self.matrix * int4_mask.float()
        int4_scale = 15.0 / int4_part.abs().max()
        int4_quantized = (int4_part * int4_scale).round().clamp(0, 15).to(torch.uint8)
        
        # Compressed size
        fp16_size = fp16_part.numel() * 2
        int4_size = int4_quantized.numel() * 1 + 8  # +scale
        compressed_size = fp16_size + int4_size
        
        compress_time = (time.time() - start) * 1000
        
        # Decompress
        start = time.time()
        decompressed = fp16_part + int4_quantized.float() / int4_scale
        decomp_time = (time.time() - start) * 1000
        
        original_bytes = self.matrix.numel() * self.matrix.element_size()
        
        mse = F.mse_loss(decompressed, self.matrix).item()
        cos_sim = F.cosine_similarity(
            self.matrix.flatten().unsqueeze(0),
            decompressed.flatten().unsqueeze(0)
        ).item()
        
        return CompressionResult(
            method="Mixed_Precision",
            original_size_bytes=original_bytes,
            compressed_size_bytes=compressed_size,
            mse=mse,
            cosine_similarity=cos_sim,
            spectral_error=self._spectral_error(self.matrix, decompressed),
            compression_time_ms=compress_time,
            decompression_time_ms=decomp_time,
            compression_ratio=original_bytes / max(compressed_size, 1),
            quality_score=max(0, 1 - mse * 30),
        )
    
    def _spectral_error(self, original: torch.Tensor, reconstructed: torch.Tensor) -> float:
        """Compute spectral norm of difference."""
        diff = original - reconstructed
        # Approximate spectral norm via power iteration
        x = torch.randn(diff.shape[0], device=diff.device)
        for _ in range(10):
            x = torch.matmul(diff, x)
            x = x / x.norm()
        spectral = torch.matmul(diff, x).norm().item()
        return spectral


def run_compression_benchmark():
    """Run the complete compression benchmark."""
    print("=" * 70)
    print("COMPRESSION METHOD COMPARISON BENCHMARK")
    print("=" * 70)
    
    # Test with different matrix sizes
    sizes = [
        (512, 512),    # Small
        (2048, 2048),  # Medium (like attention layer)
        (4096, 4096),  # Large (like FFN layer)
    ]
    
    all_results = {}
    
    for h, w in sizes:
        print(f"\n--- Matrix Size: {h}x{w} ({h*w:,} elements) ---")
        
        # Create synthetic weight matrix (with some structure)
        torch.manual_seed(42)
        matrix = torch.randn(h, w)
        
        # Add low-rank structure (simulating real weights)
        u = torch.randn(h, min(64, h))
        v = torch.randn(min(64, w), w)
        matrix += torch.matmul(u, v) * 0.5
        
        benchmark = WeightMatrixBenchmark(matrix, f"{h}x{w}")
        results = benchmark.run_all()
        
        all_results[f"{h}x{w}"] = [r.to_dict() for r in results]
        
        # Print summary
        print("\nResults:")
        print(f"{'Method':<25} {'Ratio':>10} {'MSE':>12} {'CosSim':>10} {'Quality':>10}")
        print("-" * 70)
        for r in results:
            print(f"{r.method:<25} {r.compression_ratio:>9.1f}x {r.mse:>11.6f} "
                  f"{r.cosine_similarity:>9.4f} {r.quality_score:>9.1%}")
    
    # Save results
    import os
    os.makedirs("benchmark_results", exist_ok=True)
    import time
    filename = f"benchmark_results/compression_{int(time.time())}.json"
    
    with open(filename, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n\nResults saved to: {filename}")
    
    return all_results


if __name__ == "__main__":
    run_compression_benchmark()
