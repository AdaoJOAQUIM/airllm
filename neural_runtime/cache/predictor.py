"""
Access Pattern Predictor
Learns access patterns to prefetch layers efficiently.
"""

import time
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
from dataclasses import dataclass, field
import logging

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AccessPattern:
    """Record of an access pattern."""
    layer_name: str
    timestamp: float
    duration_ms: float = 0.0
    success: bool = True


class MarkovPredictor:
    """
    Markov Chain-based layer access predictor.
    
    Learns transitions between layer accesses to predict next layers.
    """
    
    def __init__(self, order: int = 2):
        self.order = order
        self.transitions: Dict[Tuple, Dict[str, int]] = {}
        self.total_transitions: Dict[Tuple, int] = {}
    
    def update(self, sequence: List[str]) -> None:
        """Update transition probabilities from a sequence."""
        for i in range(len(sequence) - self.order):
            state = tuple(sequence[i:i + self.order])
            next_layer = sequence[i + self.order]
            
            if state not in self.transitions:
                self.transitions[state] = {}
                self.total_transitions[state] = 0
            
            self.transitions[state][next_layer] = \
                self.transitions[state].get(next_layer, 0) + 1
            self.total_transitions[state] += 1
    
    def predict(self, history: List[str]) -> List[str]:
        """Predict next layers based on history."""
        if len(history) < self.order:
            # Not enough history, use sequential prediction
            if history:
                last = history[-1]
                if last.startswith("layer_"):
                    try:
                        idx = int(last.split("_")[1])
                        return [f"layer_{idx + 1}"]
                    except:
                        pass
            return []
        
        state = tuple(history[-self.order:])
        
        if state not in self.transitions:
            return []
        
        # Get most likely next layer
        transitions = self.transitions[state]
        total = self.total_transitions[state]
        
        predictions = []
        for layer, count in sorted(transitions.items(), key=lambda x: -x[1]):
            prob = count / total
            predictions.append((layer, prob))
        
        return [p[0] for p in predictions[:3]]


class AccessPatternPredictor:
    """
    Intelligent predictor for layer access patterns.
    
    Features:
    - Markov chain for sequential patterns
    - Frequency-based prediction
    - Task-specific patterns
    - Real-time adaptation
    
    Example:
        predictor = AccessPatternPredictor(num_layers=100)
        
        # Record some accesses
        predictor.record_access("layer_0")
        predictor.record_access("layer_1")
        predictor.record_access("layer_2")
        
        # Predict next layers
        predictions = predictor.predict_next(current_layer=2, count=3)
        # -> ["layer_3", "layer_4", "layer_5"]
    """
    
    def __init__(
        self,
        num_layers: int,
        history_size: int = 10000,
        learning_rate: float = 0.1,
    ):
        self.num_layers = num_layers
        self.history_size = history_size
        self.learning_rate = learning_rate
        
        # Access history
        self.access_history: deque = deque(maxlen=history_size)
        
        # Statistics
        self.layer_access_count: Dict[int, int] = {}
        self.layer_access_time: Dict[int, List[float]] = {}
        self.total_accesses = 0
        
        # Predictors
        self.markov = MarkovPredictor(order=2)
        
        # Task patterns
        self.task_patterns: Dict[str, List[str]] = {}
        self.current_task: Optional[str] = None
        
        # Sequential bias (most common pattern)
        self.sequential_bias = 0.9  # Start with high sequential assumption
    
    def record_access(
        self,
        layer_name: str,
        duration_ms: float = 0.0,
        success: bool = True,
    ) -> None:
        """
        Record a layer access.
        
        Args:
            layer_name: Name of accessed layer
            duration_ms: Load duration in milliseconds
            success: Whether access was successful
        """
        # Parse layer index
        layer_idx = self._parse_layer_index(layer_name)
        if layer_idx is None:
            return
        
        # Record in history
        access = AccessPattern(
            layer_name=layer_name,
            timestamp=time.time(),
            duration_ms=duration_ms,
            success=success,
        )
        self.access_history.append(access)
        
        # Update statistics
        self.layer_access_count[layer_idx] = \
            self.layer_access_count.get(layer_idx, 0) + 1
        self.total_accesses += 1
        
        if layer_idx not in self.layer_access_time:
            self.layer_access_time[layer_idx] = []
        self.layer_access_time[layer_idx].append(duration_ms)
        
        # Update Markov model
        if len(self.access_history) >= 2:
            sequence = [a.layer_name for a in list(self.access_history)[-10:]]
            self.markov.update(sequence)
    
    def _parse_layer_index(self, layer_name: str) -> Optional[int]:
        """Parse layer index from layer name."""
        if layer_name.startswith("layer_"):
            try:
                return int(layer_name.split("_")[1])
            except ValueError:
                pass
        elif layer_name == "embedding":
            return 0
        elif layer_name == "lm_head":
            return self.num_layers - 1
        elif layer_name == "final_norm":
            return self.num_layers - 2
        
        return None
    
    def predict_next(
        self,
        current_layer: int,
        count: int = 3,
        use_markov: bool = True,
    ) -> List[str]:
        """
        Predict next layers to access.
        
        Args:
            current_layer: Current layer index
            count: Number of predictions
            use_markov: Whether to use Markov prediction
            
        Returns:
            List of predicted layer names
        """
        predictions = []
        
        # Method 1: Sequential prediction (highest confidence)
        for i in range(1, count + 1):
            next_idx = current_layer + i
            if 0 <= next_idx < self.num_layers:
                predictions.append(f"layer_{next_idx}")
        
        # Method 2: Markov chain
        if use_markov and self.access_history:
            history = [a.layer_name for a in list(self.access_history)[-5:]]
            markov_preds = self.markov.predict(history)
            
            for pred in markov_preds:
                if pred not in predictions:
                    predictions.append(pred)
                    if len(predictions) >= count * 2:
                        break
        
        # Method 3: Frequency-based
        if len(predictions) < count and self.layer_access_count:
            sorted_layers = sorted(
                self.layer_access_count.items(),
                key=lambda x: -x[1]
            )
            for idx, _ in sorted_layers:
                name = f"layer_{idx}"
                if name not in predictions:
                    predictions.append(name)
                    if len(predictions) >= count * 2:
                        break
        
        return predictions[:count]
    
    def predict_batch_access(
        self,
        start_layer: int,
        end_layer: int,
    ) -> List[str]:
        """
        Predict access sequence for a range of layers.
        
        Args:
            start_layer: Starting layer
            end_layer: Ending layer (exclusive)
            
        Returns:
            Predicted access order
        """
        return [f"layer_{i}" for i in range(start_layer, end_layer)]
    
    def learn_task_pattern(
        self,
        task_name: str,
        access_sequence: List[str],
    ) -> None:
        """
        Learn a task-specific access pattern.
        
        Args:
            task_name: Name of the task
            access_sequence: Sequence of layer accesses
        """
        self.task_patterns[task_name] = access_sequence
        
        # Update Markov model with this pattern
        self.markov.update(access_sequence)
        
        logger.info(f"Learned pattern for task: {task_name}")
    
    def match_task_pattern(
        self,
        recent_accesses: List[str],
    ) -> Optional[str]:
        """
        Match recent accesses to a known task pattern.
        
        Args:
            recent_accesses: Recent layer accesses
            
        Returns:
            Matched task name or None
        """
        if not recent_accesses or not self.task_patterns:
            return None
        
        # Find best matching pattern
        best_match = None
        best_score = 0.0
        
        for task_name, pattern in self.task_patterns.items():
            # Calculate match score
            match_count = 0
            for i, layer in enumerate(recent_accesses):
                if i < len(pattern) and layer == pattern[i]:
                    match_count += 1
            
            if match_count > 0:
                score = match_count / len(recent_accesses)
                if score > best_score:
                    best_score = score
                    best_match = task_name
        
        if best_score > 0.7:  # Threshold
            return best_match
        
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get predictor statistics."""
        stats = {
            "total_accesses": self.total_accesses,
            "unique_layers_accessed": len(self.layer_access_count),
            "history_size": len(self.access_history),
            "learned_patterns": len(self.task_patterns),
        }
        
        if self.layer_access_count:
            # Most accessed layers
            top_layers = sorted(
                self.layer_access_count.items(),
                key=lambda x: -x[1]
            )[:5]
            stats["top_layers"] = [
                (f"layer_{idx}", count)
                for idx, count in top_layers
            ]
        
        # Average load time per layer
        if self.layer_access_time:
            avg_times = {}
            for idx, times in self.layer_access_time.items():
                if times:
                    avg_times[f"layer_{idx}"] = sum(times) / len(times)
            stats["avg_load_time_ms"] = avg_times
        
        return stats
    
    def reset(self) -> None:
        """Reset predictor state."""
        self.access_history.clear()
        self.layer_access_count.clear()
        self.layer_access_time.clear()
        self.total_accesses = 0
        self.markov = MarkovPredictor(order=2)


class AdaptivePrefetcher:
    """
    Adaptive prefetcher that adjusts based on performance.
    
    Features:
    - Dynamic prefetch distance
    - Performance-based adjustment
    - Bandwidth-aware prefetching
    """
    
    def __init__(
        self,
        predictor: AccessPatternPredictor,
        min_prefetch_distance: int = 1,
        max_prefetch_distance: int = 10,
    ):
        self.predictor = predictor
        self.min_distance = min_prefetch_distance
        self.max_distance = max_prefetch_distance
        
        # Performance tracking
        self.recent_load_times: deque = deque(maxlen=100)
        self.prefetch_hit_rate = 0.0
        self.prefetch_miss_rate = 0.0
        
        # Dynamic distance
        self.current_distance = 3
    
    def compute_optimal_distance(
        self,
        available_bandwidth_gbs: float = 50.0,
    ) -> int:
        """
        Compute optimal prefetch distance based on conditions.
        
        Args:
            available_bandwidth_gbs: Available bandwidth in GB/s
            
        Returns:
            Optimal prefetch distance
        """
        # Average load time
        if self.recent_load_times:
            avg_time_ms = sum(self.recent_load_times) / len(self.recent_load_times)
            
            # If loads are fast, we can prefetch more aggressively
            if avg_time_ms < 50:  # Fast
                distance = min(self.max_distance, 5)
            elif avg_time_ms < 200:  # Medium
                distance = 3
            else:  # Slow
                distance = max(self.min_distance, 1)
        else:
            distance = self.current_distance
        
        # Adjust based on bandwidth
        if available_bandwidth_gbs < 10:
            distance = max(1, distance - 1)
        elif available_bandwidth_gbs > 100:
            distance = min(self.max_distance, distance + 1)
        
        self.current_distance = distance
        return distance
    
    def record_prefetch_success(self, layer_idx: int) -> None:
        """Record successful prefetch."""
        self.prefetch_hit_rate += 1
    
    def record_prefetch_miss(self, layer_idx: int) -> None:
        """Record missed prefetch."""
        self.prefetch_miss_rate += 1
        # Reduce distance on misses
        self.current_distance = max(self.min_distance, self.current_distance - 1)
    
    def record_load_time(self, time_ms: float) -> None:
        """Record actual load time."""
        self.recent_load_times.append(time_ms)
