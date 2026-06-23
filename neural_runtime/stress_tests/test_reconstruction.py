"""
RECONSTRUCTION VALIDATION TESTS
==============================

These tests verify that compression/generation methods actually preserve
the information needed for inference.

CRITICAL: These tests will FAIL if our hypotheses are wrong.

The goal is to BREAK the system, not confirm it works.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Callable
from dataclasses import dataclass
import json
import time
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class ReconstructionMetrics:
    """Metrics for reconstruction quality."""
    mse: float           # Mean Squared Error
    mae: float           # Mean Absolute Error
    cosine_sim: float    # Cosine similarity
    spectral_norm: float # Spectral norm of difference
    frobenius_norm: float # Frobenius norm of difference
    
    # Information theory metrics
    kl_divergence: float = 0.0
    information_retention: float = 0.0
    
    # Per-layer metrics
    layer_errors: List[float] = None
    
    def __post_init__(self):
        if self.layer_errors is None:
            self.layer_errors = []
    
    def to_dict(self) -> Dict:
        return {
            "mse": self.mse,
            "mae": self.mae,
            "cosine_sim": self.cosine_sim,
            "spectral_norm": self.spectral_norm,
            "frobenius_norm": self.frobenius_norm,
            "kl_divergence": self.kl_divergence,
            "information_retention": self.information_retention,
            "layer_errors": self.layer_errors,
        }
    
    def is_acceptable(self, threshold: float = 0.95) -> bool:
        """Check if reconstruction is acceptable."""
        return self.cosine_sim >= threshold


class ReconstructionTest(ABC):
    """Base class for reconstruction tests."""
    
    @abstractmethod
    def compress(self, weights: torch.Tensor) -> Any:
        """Compress weights."""
        pass
    
    @abstractmethod
    def decompress(self, compressed: Any) -> torch.Tensor:
        """Decompress weights."""
        pass
    
    @abstractmethod
    def name(self) -> str:
        """Test name."""
        pass
    
    def run(self, original: torch.Tensor) -> ReconstructionMetrics:
        """Run the test."""
        # Compress
        compressed = self.compress(original)
        
        # Decompress
        reconstructed = self.decompress(compress)
        
        # Compute metrics
        return self.compute_metrics(original, reconstructed)
    
    def compute_metrics(
        self, 
        original: torch.Tensor, 
        reconstructed: torch.Tensor
    ) -> ReconstructionMetrics:
        """Compute all metrics."""
        # Basic errors
        diff = original - reconstructed
        mse = F.mse_loss(reconstructed, original).item()
        mae = diff.abs().mean().item()
        
        # Cosine similarity
        cos_sim = F.cosine_similarity(
            original.flatten().unsqueeze(0),
            reconstructed.flatten().unsqueeze(0)
        ).item()
        
        # Spectral norm (approximate)
        spectral_norm = self._compute_spectral_norm(diff)
        
        # Frobenius norm
        fro_norm = diff.norm(p='fro').item()
        
        # Information retention (based on cosine similarity)
        info_retention = max(0, cos_sim)
        
        # KL divergence (simplified)
        kl = self._compute_kl(original, reconstructed)
        
        return ReconstructionMetrics(
            mse=mse,
            mae=mae,
            cosine_sim=cos_sim,
            spectral_norm=spectral_norm,
            frobenius_norm=fro_norm,
            kl_divergence=kl,
            information_retention=info_retention,
        )
    
    def _compute_spectral_norm(self, matrix: torch.Tensor) -> float:
        """Approximate spectral norm via power iteration."""
        if matrix.numel() == 0:
            return 0.0
        
        x = torch.randn(matrix.shape[0], device=matrix.device)
        x = x / x.norm()
        
        for _ in range(10):
            y = torch.matmul(matrix, x)
            if matrix.shape[0] != matrix.shape[1]:
                y = torch.matmul(matrix.t(), y)
            else:
                y = torch.matmul(matrix, y)
            x_new = y / (y.norm() + 1e-8)
            
            if torch.abs(x_new - x).max().item() < 1e-6:
                break
            x = x_new
        
        return x.norm().item()
    
    def _compute_kl(
        self, 
        original: torch.Tensor, 
        reconstructed: torch.Tensor
    ) -> float:
        """Compute KL divergence (simplified)."""
        # Normalize to probability distributions
        orig_prob = F.softmax(original.flatten(), dim=0)
        recon_prob = F.softmax(reconstructed.flatten(), dim=0)
        
        # KL divergence
        kl = F.kl_div(
            recon_prob.unsqueeze(0).log(),
            orig_prob.unsqueeze(0),
            reduction='batchmean'
        ).item()
        
        return kl


class FractalReconstructionTest(ReconstructionTest):
    """Test fractal compression reconstruction."""
    
    def name(self) -> str:
        return "fractal_compression"
    
    def compress(self, weights: torch.Tensor) -> Dict:
        from ..representation.fractal import FractalCompressor
        compressor = FractalCompressor(similarity_threshold=0.7)
        return compressor.compress(weights)
    
    def decompress(self, compressed: Dict) -> torch.Tensor:
        from ..representation.fractal import FractalCompressor
        compressor = FractalCompressor()
        return compressor.decompress(compressed)


class LearnedCompressionTest(ReconstructionTest):
    """Test learned autoencoder compression."""
    
    def name(self) -> str:
        return "learned_compression"
    
    def compress(self, weights: torch.Tensor) -> torch.Tensor:
        from ..generation.compression import NeuralCompressor
        compressor = NeuralCompressor(weights.shape, latent_dim=256)
        return compressor.compress(weights)
    
    def decompress(self, compressed: torch.Tensor) -> torch.Tensor:
        from ..generation.compression import NeuralCompressor
        # Need to recreate compressor - this is a limitation
        return compressed  # Will need proper reconstruction


class SVDCompressionTest(ReconstructionTest):
    """Test SVD low-rank compression."""
    
    def __init__(self, rank_ratio: float = 0.1):
        self.rank_ratio = rank_ratio
    
    def name(self) -> str:
        return f"svd_rank_{int(self.rank_ratio * 100)}pct"
    
    def compress(self, weights: torch.Tensor) -> Tuple:
        U, S, V = torch.svd(weights)
        rank = max(1, int(min(weights.shape) * self.rank_ratio))
        return U[:, :rank], S[:rank], V[:, :rank], rank
    
    def decompress(self, compressed: Tuple) -> torch.Tensor:
        U, S, V, rank = compressed
        return torch.matmul(U, torch.diag(S))
        return torch.matmul(torch.matmul(U, torch.diag(S)), V.t())


class QuantizationTest(ReconstructionTest):
    """Test quantization compression."""
    
    def __init__(self, bits: int = 8):
        self.bits = bits
    
    def name(self) -> str:
        return f"quant_{self.bits}bit"
    
    def compress(self, weights: torch.Tensor) -> Tuple:
        if self.bits == 4:
            max_val = weights.abs().max()
            scale = 15.0 / max_val
            quantized = (weights * scale).round().clamp(0, 15).to(torch.uint8)
            return quantized, scale
        else:  # 8-bit
            max_val = weights.abs().max()
            scale = 127.0 / max_val
            quantized = (weights * scale).round().clamp(-128, 127).to(torch.int8)
            return quantized, scale
    
    def decompress(self, compressed: Tuple) -> torch.Tensor:
        quantized, scale = compressed
        if self.bits == 4:
            return quantized.float() / scale
        else:
            return quantized.float() / scale


class PruningTest(ReconstructionTest):
    """Test pruning compression."""
    
    def __init__(self, sparsity: float = 0.5):
        self.sparsity = sparsity
    
    def name(self) -> str:
        return f"prune_{int(self.sparsity * 100)}pct"
    
    def compress(self, weights: torch.Tensor) -> Tuple:
        threshold = torch.quantile(weights.abs().flatten(), self.sparsity)
        mask = (weights.abs() > threshold).float()
        pruned = weights * mask
        
        # Store sparse representation
        indices = mask.nonzero(as_tuple=True)
        values = pruned[indices]
        
        return indices, values, weights.shape
    
    def decompress(self, compressed: Tuple) -> torch.Tensor:
        indices, values, shape = compressed
        reconstructed = torch.zeros(shape, dtype=values.dtype, device=values.device)
        reconstructed[indices] = values
        return reconstructed


class ReconstructionValidationSuite:
    """
    Complete suite for validating reconstruction quality.
    
    This will find where our compression methods FAIL.
    """
    
    def __init__(self):
        self.tests: List[ReconstructionTest] = []
        self.results: Dict[str, ReconstructionMetrics] = {}
    
    def add_test(self, test: ReconstructionTest):
        """Add a test to the suite."""
        self.tests.append(test)
    
    def run_all(
        self,
        matrix_sizes: List[Tuple[int, int]] = None,
        num_trials: int = 5,
    ) -> Dict[str, Dict]:
        """
        Run all tests across multiple matrix sizes and trials.
        
        Returns results that will show where compression fails.
        """
        if matrix_sizes is None:
            matrix_sizes = [
                (512, 512),    # Small
                (1024, 1024),  # Medium
                (2048, 2048),  # Large (attention)
                (4096, 4096),  # Very large (FFN)
            ]
        
        results = {}
        
        for size in matrix_sizes:
            size_key = f"{size[0]}x{size[1]}"
            results[size_key] = {}
            
            for trial in range(num_trials):
                torch.manual_seed(trial)
                matrix = torch.randn(size)
                
                # Add structure (simulating real weights)
                u = torch.randn(size[0], min(64, size[0]))
                v = torch.randn(min(64, size[1]), size[1])
                matrix = matrix + torch.matmul(u, v) * 0.3
                
                for test in self.tests:
                    test_key = f"{test.name()}_trial{trial}"
                    
                    try:
                        metrics = test.run(matrix)
                        results[size_key][test_key] = metrics.to_dict()
                        
                        logger.info(f"{size_key}/{test.name()}: "
                                   f"cos={metrics.cosine_sim:.4f}, "
                                   f"mse={metrics.mse:.6f}")
                    except Exception as e:
                        results[size_key][test_key] = {"error": str(e)}
                        logger.error(f"{size_key}/{test.name()}: FAILED - {e}")
        
        self.results = results
        return results
    
    def find_failures(self, threshold: float = 0.9) -> List[str]:
        """Find tests that failed to meet quality threshold."""
        failures = []
        
        for size_key, tests in self.results.items():
            for test_key, metrics in tests.items():
                if isinstance(metrics, dict) and "cosine_sim" in metrics:
                    if metrics["cosine_sim"] < threshold:
                        failures.append(f"{size_key}/{test_key}")
        
        return failures
    
    def find_acceptable_methods(self, threshold: float = 0.95) -> Dict:
        """Find methods that consistently meet quality threshold."""
        acceptable = {}
        
        for size_key, tests in self.results.items():
            for test_key, metrics in tests.items():
                if isinstance(metrics, dict) and "cosine_sim" in metrics:
                    method = test_key.split("_trial")[0]
                    if method not in acceptable:
                        acceptable[method] = []
                    acceptable[method].append(metrics["cosine_sim"] >= threshold)
        
        # Average
        for method, results_list in acceptable.items():
            acceptable[method] = sum(results_list) / len(results_list)
        
        return acceptable
    
    def generate_report(self) -> str:
        """Generate a validation report."""
        report = []
        report.append("=" * 70)
        report.append("RECONSTRUCTION VALIDATION REPORT")
        report.append("=" * 70)
        
        # Find failures
        failures = self.find_failures()
        
        report.append(f"\nFailed Tests (cosine_sim < 0.9): {len(failures)}")
        for failure in failures:
            report.append(f"  - {failure}")
        
        # Acceptable methods
        acceptable = self.find_acceptable_methods()
        
        report.append("\nAcceptable Methods (>=95% pass rate):")
        for method, rate in sorted(acceptable.items(), key=lambda x: -x[1]):
            status = "✅" if rate >= 0.95 else "⚠️" if rate >= 0.8 else "❌"
            report.append(f"  {status} {method}: {rate:.1%}")
        
        # Summary statistics
        all_cosine = []
        for size_key, tests in self.results.items():
            for metrics in tests.values():
                if isinstance(metrics, dict) and "cosine_sim" in metrics:
                    all_cosine.append(metrics["cosine_sim"])
        
        if all_cosine:
            report.append(f"\nOverall Statistics:")
            report.append(f"  Mean cosine similarity: {sum(all_cosine)/len(all_cosine):.4f}")
            report.append(f"  Min: {min(all_cosine):.4f}")
            report.append(f"  Max: {max(all_cosine):.4f}")
        
        return "\n".join(report)


def run_reconstruction_validation():
    """Run the complete reconstruction validation suite."""
    logging.basicConfig(level=logging.INFO)
    
    suite = ReconstructionValidationSuite()
    
    # Add tests
    suite.add_test(FractalReconstructionTest())
    suite.add_test(LearnedCompressionTest())
    suite.add_test(SVDCompressionTest(0.1))
    suite.add_test(SVDCompressionTest(0.05))
    suite.add_test(QuantizationTest(8))
    suite.add_test(QuantizationTest(4))
    suite.add_test(PruningTest(0.5))
    suite.add_test(PruningTest(0.7))
    
    # Run
    results = suite.run_all(num_trials=3)
    
    # Generate report
    report = suite.generate_report()
    print(report)
    
    # Save results
    with open("reconstruction_validation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return suite


if __name__ == "__main__":
    run_reconstruction_validation()
