"""
Expert Router
============

Intelligent routing to experts based on task context.
"""

from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class TaskAwareRouter(nn.Module):
    """
    Router that uses task context for expert selection.
    
    Instead of just routing based on input, uses:
    - Task embedding
    - Historical patterns
    - Expert load balancing
    """
    
    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        top_k: int = 2,
        use_task_embedding: bool = True,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.use_task_embedding = use_task_embedding
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, input_dim)
        
        # Task embedding projection
        if use_task_embedding:
            self.task_proj = nn.Linear(input_dim, input_dim)
        
        # Router
        self.router = nn.Linear(input_dim, num_experts)
        
        # Load balancing
        self.expert_counts = torch.zeros(num_experts)
        self.aux_loss_weight = 0.01
    
    def forward(
        self,
        x: torch.Tensor,
        task_embedding: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Route inputs to experts.
        
        Returns:
            top_k_weights, top_k_indices, load_balancing_loss
        """
        # Project input
        h = self.input_proj(x)
        
        # Combine with task embedding
        if self.use_task_embedding and task_embedding is not None:
            task_h = self.task_proj(task_embedding)
            h = h + task_h
        
        # Router logits
        logits = self.router(h)
        
        # Get top-k
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_logits, dim=-1)
        
        # Load balancing loss
        load_loss = self._compute_load_balance_loss(top_k_indices, x.shape[0])
        
        return top_k_weights, top_k_indices, load_loss
    
    def _compute_load_balance_loss(
        self,
        top_k_indices: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Compute load balancing auxiliary loss."""
        # Count expert assignments
        expert_counts = torch.zeros(self.num_experts, device=top_k_indices.device)
        
        for k in range(self.top_k):
            counts = torch.bincount(top_k_indices[:, k], minlength=self.num_experts)
            expert_counts += counts.float()
        
        # Update running counts
        self.expert_counts = self.expert_counts.to(expert_counts.device)
        self.expert_counts = 0.9 * self.expert_counts + 0.1 * expert_counts.detach()
        
        # Compute loss
        total_tokens = batch_size * self.top_k
        expert_fraction = expert_counts / total_tokens
        router_fraction = self.expert_counts / total_tokens
        
        # Minimize variance between expert fractions
        loss = self.num_experts * (expert_fraction * router_fraction).sum()
        
        return loss * self.aux_loss_weight


class LearnedRouter(nn.Module):
    """
    Router with learned routing strategy.
    
    Can learn optimal routing patterns from data.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        
        # Embed input for routing
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Policy network
        self.policy = nn.Linear(hidden_dim, num_experts)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Get routing probabilities."""
        h = self.embedding(x)
        return F.softmax(self.policy(h), dim=-1)


class AdaptiveRouter:
    """
    Router that adapts based on query complexity.
    
    Simple queries → fewer experts
    Complex queries → more experts
    """
    
    def __init__(self, router: nn.Module):
        self.router = router
        self.complexity_estimator = None
    
    def estimate_complexity(self, x: torch.Tensor) -> float:
        """Estimate query complexity."""
        # Simple heuristic: variance in input
        variance = x.var().item()
        
        # Also consider input magnitude
        magnitude = x.abs().mean().item()
        
        # Combine
        complexity = min(1.0, (variance * magnitude) ** 0.25)
        
        return complexity
    
    def route(
        self,
        x: torch.Tensor,
        max_experts: int = 4,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Route with adaptive expert count.
        
        Returns routing decision and suggested expert count.
        """
        complexity = self.estimate_complexity(x)
        
        # More experts for complex queries
        top_k = max(1, int(complexity * max_experts))
        top_k = min(top_k, max_experts)
        
        # Get routing
        weights, indices, loss = self.router(x, top_k=top_k)
        
        return weights, indices, top_k


class HashRouter:
    """
    Deterministic router using hash of input.
    
    Always routes the same input to the same experts.
    Useful for caching and consistency.
    """
    
    def __init__(self, num_experts: int, top_k: int = 2):
        self.num_experts = num_experts
        self.top_k = top_k
    
    def route(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Route based on input hash."""
        # Create deterministic hash
        hash_input = x.sum(dim=-1).hash()
        
        # Select experts based on hash
        expert_ids = (hash_input % self.num_experts).unsqueeze(-1)
        
        # Create one-hot weights
        batch_size = x.shape[0]
        weights = torch.zeros(batch_size, self.num_experts, device=x.device)
        indices = torch.zeros(batch_size, self.top_k, dtype=torch.long, device=x.device)
        
        for k in range(self.top_k):
            expert_id = ((hash_input + k) % self.num_experts).unsqueeze(-1)
            weights.scatter_(1, expert_id.expand(-1, 1), 1.0 / self.top_k)
            indices[:, k] = expert_id.squeeze(-1)
        
        return weights, indices
