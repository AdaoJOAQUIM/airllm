"""
Validator
=========

Scientific validation framework for UCCE.

Measures:
- Compression ratio
- Quality preservation
- Performance
- Resource usage
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum
import time


class MetricType(Enum):
    """Type of metric."""
    COMPRESSION = "compression"  # Size reduction
    QUALITY = "quality"  # Output quality
    PERFORMANCE = "performance"  # Speed/memory
    ACCURACY = "accuracy"  # Task accuracy


@dataclass
class Metric:
    """A single metric measurement."""
    name: str
    value: float
    unit: str
    metric_type: MetricType
    baseline: Optional[float] = None
    
    def improvement_ratio(self) -> float:
        """Calculate improvement ratio vs baseline."""
        if self.baseline is None or self.baseline == 0:
            return 1.0
        return self.value / self.baseline
    
    def improvement_percent(self) -> float:
        """Calculate improvement percentage vs baseline."""
        if self.baseline is None or self.baseline == 0:
            return 0.0
        return (self.value - self.baseline) / self.baseline * 100


@dataclass
class ValidationResult:
    """
    Result of validating a hypothesis.
    
    Contains all measurements and comparisons.
    """
    hypothesis_id: str
    hypothesis_name: str
    
    # Metrics
    metrics: List[Metric]
    
    # Overall scores
    compression_score: float  # 0-1, higher = better compression
    quality_score: float  # 0-1, higher = better quality
    performance_score: float  # 0-1, higher = better performance
    overall_score: float  # Weighted average
    
    # Comparison vs baselines
    vs_fp16: Dict[str, Metric]
    vs_quantized: Dict[str, Metric]
    vs_pruned: Dict[str, Metric]
    
    # Validation criteria
    passed: bool
    failure_reasons: List[str]
    
    # Details
    experiment_time_seconds: float
    notes: str = ""
    
    def summary(self) -> str:
        """Generate a text summary."""
        lines = [
            f"Validation Result: {self.hypothesis_name}",
            "=" * 50,
            f"Overall Score: {self.overall_score:.2%}",
            f"Compression Score: {self.compression_score:.2%}",
            f"Quality Score: {self.quality_score:.2%}",
            f"Performance Score: {self.performance_score:.2%}",
            "",
            "Metrics:",
        ]
        
        for m in self.metrics:
            if m.baseline:
                lines.append(
                    f"  {m.name}: {m.value:.4f} ({m.improvement_percent():+.1f}% vs baseline)"
                )
            else:
                lines.append(f"  {m.name}: {m.value:.4f} {m.unit}")
        
        if self.vs_fp16:
            lines.append("\nvs FP16 Baseline:")
            for name, m in self.vs_fp16.items():
                lines.append(f"  {name}: {m.value:.4f}")
        
        if self.failure_reasons:
            lines.append("\nFailures:")
            for reason in self.failure_reasons:
                lines.append(f"  ❌ {reason}")
        
        return "\n".join(lines)


class Validator:
    """
    Scientific validator for compression methods.
    
    Validates by measuring:
    1. Compression ratio achieved
    2. Quality preservation
    3. Performance metrics
    4. Comparison with baselines
    """
    
    def __init__(self):
        self.results: List[ValidationResult] = []
    
    def validate(
        self,
        hypothesis_id: str,
        hypothesis_name: str,
        original_model_size_bytes: int,
        compressed_model_size_bytes: int,
        quality_metrics: Dict[str, float],
        performance_metrics: Dict[str, float],
        baseline_metrics: Optional[Dict[str, float]] = None,
    ) -> ValidationResult:
        """
        Validate a hypothesis with measurements.
        
        Args:
            hypothesis_id: ID of hypothesis
            hypothesis_name: Name of hypothesis
            original_model_size_bytes: Original model size
            compressed_model_size_bytes: Compressed model size
            quality_metrics: Dict of quality metrics (perp, accuracy, etc.)
            performance_metrics: Dict of performance metrics (latency, memory, etc.)
            baseline_metrics: Optional baseline for comparison
        """
        metrics = []
        
        # Compression metrics
        compression_ratio = original_model_size_bytes / compressed_model_size_bytes
        metrics.append(Metric(
            name="compression_ratio",
            value=compression_ratio,
            unit="x",
            metric_type=MetricType.COMPRESSION,
        ))
        metrics.append(Metric(
            name="size_reduction_percent",
            value=100 * (1 - compressed_model_size_bytes / original_model_size_bytes),
            unit="%",
            metric_type=MetricType.COMPRESSION,
        ))
        
        # Quality metrics
        for name, value in quality_metrics.items():
            metrics.append(Metric(
                name=name,
                value=value,
                unit="score",
                metric_type=MetricType.QUALITY,
                baseline=baseline_metrics.get(name) if baseline_metrics else None,
            ))
        
        # Performance metrics
        for name, value in performance_metrics.items():
            metrics.append(Metric(
                name=name,
                value=value,
                unit="ms",
                metric_type=MetricType.PERFORMANCE,
            ))
        
        # Calculate scores
        compression_score = min(1.0, compression_ratio / 10.0)  # 10x = 100%
        quality_score = self._calculate_quality_score(quality_metrics, baseline_metrics)
        performance_score = self._calculate_performance_score(performance_metrics)
        overall_score = (
            0.3 * compression_score +
            0.5 * quality_score +
            0.2 * performance_score
        )
        
        # Determine if passed
        failure_reasons = []
        if compression_ratio < 1.5:
            failure_reasons.append(
                f"Compression ratio {compression_ratio:.2f}x is too low (need >1.5x)"
            )
        if quality_score < 0.8:
            failure_reasons.append(
                f"Quality score {quality_score:.0%} is too low (need >80%)"
            )
        
        passed = len(failure_reasons) == 0
        
        # Comparison dictionaries
        vs_fp16 = {}
        if baseline_metrics:
            for name in quality_metrics:
                if name in baseline_metrics:
                    vs_fp16[name] = Metric(
                        name=name,
                        value=quality_metrics[name],
                        unit="score",
                        metric_type=MetricType.QUALITY,
                        baseline=baseline_metrics[name],
                    )
        
        result = ValidationResult(
            hypothesis_id=hypothesis_id,
            hypothesis_name=hypothesis_name,
            metrics=metrics,
            compression_score=compression_score,
            quality_score=quality_score,
            performance_score=performance_score,
            overall_score=overall_score,
            vs_fp16=vs_fp16,
            vs_quantized={},  # Would fill with actual comparison
            vs_pruned={},  # Would fill with actual comparison
            passed=passed,
            failure_reasons=failure_reasons,
            experiment_time_seconds=0.0,  # Would measure
        )
        
        self.results.append(result)
        return result
    
    def _calculate_quality_score(
        self,
        metrics: Dict[str, float],
        baseline: Optional[Dict[str, float]],
    ) -> float:
        """Calculate overall quality score."""
        if not metrics:
            return 1.0
        
        if baseline:
            # Calculate preservation ratio
            scores = []
            for name, value in metrics.items():
                if name in baseline and baseline[name] > 0:
                    score = value / baseline[name]
                    scores.append(min(1.0, score))  # Cap at 1.0
            return sum(scores) / len(scores) if scores else 1.0
        
        # If no baseline, assume normalized metrics
        return sum(m.values()) / len(metrics) if metrics else 1.0
    
    def _calculate_performance_score(self, metrics: Dict[str, float]) -> float:
        """Calculate performance score (lower is better)."""
        if not metrics:
            return 1.0
        
        # Assume latency/memory metrics
        # Higher is worse, so invert
        # This is simplified
        return 0.5  # Placeholder
    
    def get_results(self) -> List[ValidationResult]:
        """Get all validation results."""
        return self.results
    
    def summary(self) -> str:
        """Generate a summary of all results."""
        lines = [
            "=" * 60,
            "VALIDATION SUMMARY",
            "=" * 60,
            "",
            f"Total Validations: {len(self.results)}",
            "",
        ]
        
        passed = [r for r in self.results if r.passed]
        failed = [r for r in self.results if not r.passed]
        
        lines.append(f"Passed: {len(passed)}")
        lines.append(f"Failed: {len(failed)}")
        lines.append("")
        
        lines.append("RESULTS:")
        for r in self.results:
            status = "✅" if r.passed else "❌"
            lines.append(
                f"{status} [{r.hypothesis_id}] {r.hypothesis_name}: "
                f"{r.overall_score:.0%} "
                f"(compression: {r.compression_score:.1f}x, "
                f"quality: {r.quality_score:.0%})"
            )
        
        return "\n".join(lines)
