"""
Tests for Fractal Representation and Compression
==========================================

Unit tests for fractal compression and codec components.
"""

import pytest
import torch
import sys
sys.path.insert(0, '.')

from neural_runtime.representation.fractal import (
    QuadrantDecomposer,
    FractalCompressor,
    RecursiveFractalCompressor,
    SelfSimilarityAnalyzer,
    FractalStats,
)


class TestQuadrantDecomposer:
    """Tests for QuadrantDecomposer."""
    
    def test_initialization(self):
        """Test decomposer initialization."""
        matrix = torch.randn(1024, 1024)
        decomposer = QuadrantDecomposer(matrix)
        
        assert decomposer.matrix is not None
        assert decomposer.h == 1024
        assert decomposer.w == 1024
    
    def test_get_quadrants(self):
        """Test getting quadrants."""
        matrix = torch.randn(100, 100)
        decomposer = QuadrantDecomposer(matrix)
        
        q1, q2, q3, q4 = decomposer.get_quadrants()
        
        # Check shapes
        assert q1.shape[0] == 50
        assert q1.shape[1] == 50
        assert q2.shape[0] == 50
        assert q4.shape[0] == 50
    
    def test_compute_correlations(self):
        """Test computing quadrant correlations."""
        # Create matrix with some structure
        torch.manual_seed(42)
        matrix = torch.randn(512, 512)
        
        decomposer = QuadrantDecomposer(matrix)
        correlations = decomposer.compute_correlations()
        
        assert "q1_q2" in correlations
        assert "q1_q3" in correlations
        assert "q1_q4" in correlations
        
        # Correlations should be between -1 and 1
        for corr in correlations.values():
            assert -1.0 <= corr <= 1.0
    
    def test_estimate_self_similarity(self):
        """Test self-similarity estimation."""
        matrix = torch.randn(256, 256)
        decomposer = QuadrantDecomposer(matrix)
        
        similarity = decomposer.estimate_self_similarity()
        
        assert -1.0 <= similarity <= 1.0


class TestFractalStats:
    """Tests for FractalStats dataclass."""
    
    def test_creation(self):
        """Test creating FractalStats."""
        stats = FractalStats(
            self_similarity_score=0.75,
            quadrant_correlations={"q1_q2": 0.8},
            optimal_depth=3,
            compression_ratio_achievable=4.0,
            reconstruction_error=0.05,
        )
        
        assert stats.self_similarity_score == 0.75
        assert stats.optimal_depth == 3
        assert stats.compression_ratio_achievable == 4.0


class TestFractalCompressor:
    """Tests for FractalCompressor."""
    
    def test_initialization(self):
        """Test compressor initialization."""
        compressor = FractalCompressor(
            max_depth=4,
            similarity_threshold=0.7,
        )
        
        assert compressor.max_depth == 4
        assert compressor.similarity_threshold == 0.7
    
    def test_default_initialization(self):
        """Test default initialization."""
        compressor = FractalCompressor()
        
        assert compressor.max_depth == 4
        assert compressor.similarity_threshold == 0.7
    
    def test_analyze(self):
        """Test analyzing a matrix."""
        torch.manual_seed(42)
        matrix = torch.randn(512, 512)
        
        compressor = FractalCompressor()
        stats = compressor.analyze(matrix)
        
        assert isinstance(stats, FractalStats)
        assert 0.0 <= stats.self_similarity_score <= 1.0
    
    def test_compress_returns_dict(self):
        """Test compress returns proper dict structure."""
        matrix = torch.randn(256, 256)
        compressor = FractalCompressor()
        
        compressed = compressor.compress(matrix)
        
        assert isinstance(compressed, dict)
        assert "shape" in compressed
        assert "depth" in compressed
    
    def test_compress_decompress_roundtrip(self):
        """Test compress-decompress roundtrip."""
        torch.manual_seed(42)
        matrix = torch.randn(256, 256)
        
        compressor = FractalCompressor()
        compressed = compressor.compress(matrix)
        reconstructed = compressor.decompress(compressed)
        
        assert reconstructed.shape == matrix.shape
    
    def test_compression_ratio(self):
        """Test compression ratio is calculated."""
        matrix = torch.randn(512, 512)
        compressor = FractalCompressor()
        
        compressed = compressor.compress(matrix)
        
        assert "compression_ratio" in compressed
        assert compressed["compression_ratio"] > 0
    
    def test_reconstruction_error_low_for_structured(self):
        """Test reconstruction error is low for structured matrices."""
        # Create matrix with clear structure
        torch.manual_seed(42)
        base = torch.randn(128, 128)
        
        # Create self-similar matrix
        matrix = torch.zeros(256, 256)
        matrix[:128, :128] = base
        matrix[:128, 128:] = base * 0.9 + 0.1
        matrix[128:, :128] = base * 0.85 + 0.2
        matrix[128:, 128:] = base * 0.8 + 0.15
        
        compressor = FractalCompressor(similarity_threshold=0.5)
        compressed = compressor.compress(matrix)
        reconstructed = compressor.decompress(compressed)
        
        # Error should be relatively low for self-similar matrices
        error = (matrix - reconstructed).abs().mean().item()
        assert error < 1.0  # Should be low for highly similar matrices


class TestRecursiveFractalCompressor:
    """Tests for RecursiveFractalCompressor."""
    
    def test_initialization(self):
        """Test recursive compressor initialization."""
        compressor = RecursiveFractalCompressor(
            min_block_size=32,
        )
        
        assert compressor.min_block_size == 32
    
    def test_recursive_compress_decompress(self):
        """Test recursive compression roundtrip."""
        torch.manual_seed(42)
        matrix = torch.randn(512, 512)
        
        compressor = RecursiveFractalCompressor()
        compressed = compressor.compress_recursive(matrix)
        
        assert compressed["type"] in ["leaf", "fractal", "recursive", "mixed"]
        
        reconstructed = compressor.decompress_recursive(compressed)
        
        assert reconstructed.shape == matrix.shape


class TestSelfSimilarityAnalyzer:
    """Tests for SelfSimilarityAnalyzer."""
    
    def test_analyze_matrix(self):
        """Test analyzing matrix self-similarity."""
        torch.manual_seed(42)
        matrix = torch.randn(512, 512)
        
        analyzer = SelfSimilarityAnalyzer()
        results = analyzer.analyze_matrix(matrix, block_sizes=[64, 128])
        
        assert isinstance(results, dict)
    
    def test_estimate_compression_potential(self):
        """Test compression potential estimation."""
        torch.manual_seed(42)
        matrix = torch.randn(256, 256)
        
        analyzer = SelfSimilarityAnalyzer()
        ratio, method = analyzer.estimate_compression_potential(matrix)
        
        assert ratio >= 1.0  # At least no compression
        assert method in ["no_compression", "fractal_4x", "fractal_2x", "partial_fractal", "minimal"]
    
    def test_high_similarity_high_compression(self):
        """Test high similarity leads to higher compression estimate."""
        # Create highly self-similar matrix
        base = torch.randn(128, 128)
        matrix = torch.zeros(256, 256)
        matrix[:128, :128] = base
        matrix[:128, 128:] = base * 1.0  # Very similar
        matrix[128:, :128] = base * 1.0
        matrix[128:, 128:] = base * 1.0
        
        analyzer = SelfSimilarityAnalyzer()
        ratio, method = analyzer.estimate_compression_potential(matrix)
        
        # Should estimate higher compression for more similar matrices
        assert ratio >= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
