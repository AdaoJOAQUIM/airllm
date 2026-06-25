"""
Sparse Mixture of Experts
========================

Implements sparse activation of experts - only a subset of experts
are active for each token, dramatically reducing active parameters.

Reference: Switch Transformer, Mixtral-8x7B
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExpertConfig:
    """Configuration for sparse mixture of experts."""
    num_experts: int = 8          # Number of experts
    top_k: int = 2               # Number of experts to activate per token
    expert_hidden_dim: int = 14336  # Hidden dimension of each expert (Mixtral default)
    model_dim: int = 4096         # Model dimension
    capacity_factor: float = 1.25  # Expert capacity multiplier
    
    # Routing options
    use_bias: bool = False        # Use bias in routing
    normalize_routing: bool = True  # Normalize routing weights
    
    # Load balancing
    load_balance_weight: float = 0.01  # Weight for load balancing loss
    use_noisy_topk: bool = True   # Add noise for exploration
    noise_std: float = 0.1        # Noise standard deviation


@dataclass
class ExpertStats:
    """Statistics for MoE expert usage."""
    expert_counts: torch.Tensor = None        # Number of tokens per expert
    expert_capacity: int = 0                   # Max tokens per expert
    routing_weights: torch.Tensor = None     # Average routing weights
    load_balance_loss: float = 0.0            # Load balancing loss value
    expert_utilization: List[float] = None    # Utilization per expert (%)


class Expert(nn.Module):
    """
    A single expert network.
    
    Typically a FFN/Gated MLP:
    - Gate projects to expert_hidden_dim
    - Up projects to expert_hidden_dim  
    - Down projects back to model_dim
    """
    
    def __init__(
        self,
        model_dim: int,
        expert_dim: int,
        bias: bool = True,
    ):
        super().__init__()
        
        self.gate = nn.Linear(model_dim, expert_dim, bias=bias)
        self.up = nn.Linear(model_dim, expert_dim, bias=bias)
        self.down = nn.Linear(expert_dim, model_dim, bias=bias)
        
        # Initialize
        nn.init.normal_(self.gate.weight, std=0.02)
        nn.init.normal_(self.up.weight, std=0.02)
        nn.init.zeros_(self.down.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch * seq, model_dim]
            
        Returns:
            [batch * seq, model_dim]
        """
        # SiLU activation (SwiGLU-like)
        gate_out = F.silu(self.gate(x))
        up_out = self.up(x)
        
        # Element-wise product
        intermediate = gate_out * up_out
        
        # Down projection
        return self.down(intermediate)


class ExpertRouter(nn.Module):
    """
    Router that selects top-k experts for each token.
    
    Uses a simple linear layer + softmax for routing decisions.
    Optionally adds noise for exploration (noisy top-k).
    """
    
    def __init__(
        self,
        model_dim: int,
        num_experts: int,
        top_k: int,
        use_bias: bool = False,
        use_noisy_topk: bool = True,
        noise_std: float = 0.1,
    ):
        super().__init__()
        
        self.model_dim = model_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.use_noisy_topk = use_noisy_topk
        self.noise_std = noise_std
        
        # Routing weights
        self.gate = nn.Linear(model_dim, num_experts, bias=use_bias)
        
        # Initialize
        nn.init.normal_(self.gate.weight, std=0.02)
        if use_bias:
            nn.init.zeros_(self.gate.bias)
    
    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Route tokens to experts.
        
        Args:
            x: [batch * seq, model_dim]
            
        Returns:
            top_k_weights: [batch * seq, top_k] - routing weights for top-k experts
            top_k_indices: [batch * seq, top_k] - indices of top-k experts
            load: [num_experts] - expert load for balancing
        """
        batch_size = x.shape[0]
        
        # Compute routing logits
        logits = self.gate(x)  # [batch * seq, num_experts]
        
        # Add noise for exploration (only during training)
        if self.use_noisy_topk and self.training:
            noise = torch.randn_like(logits) * self.noise_std
            logits = logits + noise
        
        # Get top-k experts
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1)
        
        # Normalize weights (softmax)
        top_k_weights = F.softmax(top_k_logits, dim=-1)
        
        # Compute load for load balancing
        # Count how many tokens are routed to each expert
        load = torch.zeros(self.num_experts, device=x.device, dtype=x.dtype)
        for k in range(self.top_k):
            load.scatter_add_(0, top_k_indices[:, k], torch.ones(batch_size, device=x.device))
        
        return top_k_weights, top_k_indices, load
    
    def compute_load_balance_loss(
        self,
        load: torch.Tensor,
        num_tokens: int,
        capacity_factor: float = 1.25,
    ) -> torch.Tensor:
        """
        Compute load balancing loss to ensure expert utilization.
        
        Args:
            load: [num_experts] - tokens per expert
            num_tokens: total number of tokens
            capacity_factor: expert capacity multiplier
            
        Returns:
            Load balancing loss
        """
        # Target: equal load per expert
        target_load = num_tokens / self.num_experts
        
        # Capacity per expert
        capacity = int(num_tokens * capacity_factor / self.num_experts)
        
        # Compute load imbalance
        load_factor = self.num_experts * load / max(num_tokens, 1)
        load_balance_loss = self.num_experts * (load_factor ** 2).mean()
        
        return load_balance_loss


class SparseMixtureOfExperts(nn.Module):
    """
    Sparse Mixture of Experts layer.
    
    For each token, only top-k experts are activated.
    This reduces active parameters from:
        num_experts * expert_size -> top_k * expert_size
    
    For Mixtral-8x7B:
        8 experts * 14336 hidden = 114688 params per token (dense equivalent)
        2 experts * 14336 hidden = 28672 params per token (sparse)
        Reduction: 4x active parameters
    
    Key Innovation: We can store 100B params but only activate 1-2B!
    """
    
    def __init__(
        self,
        config: ExpertConfig,
    ):
        super().__init__()
        
        self.config = config
        
        # Create experts
        self.experts = nn.ModuleList([
            Expert(
                model_dim=config.model_dim,
                expert_dim=config.expert_hidden_dim,
                bias=config.use_bias,
            )
            for _ in range(config.num_experts)
        ])
        
        # Create router
        self.router = ExpertRouter(
            model_dim=config.model_dim,
            num_experts=config.num_experts,
            top_k=config.top_k,
            use_bias=config.use_bias,
            use_noisy_topk=config.use_noisy_topk,
            noise_std=config.noise_std,
        )
        
        # Statistics
        self.stats = ExpertStats(
            expert_counts=torch.zeros(config.num_experts),
            expert_capacity=0,
            routing_weights=torch.zeros(config.num_experts),
        )
    
    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with sparse expert activation.
        
        Args:
            x: [batch, seq, model_dim]
            
        Returns:
            output: [batch, seq, model_dim]
            aux_loss: Auxiliary loss for training (load balancing)
        """
        batch_size, seq_len, model_dim = x.shape
        num_tokens = batch_size * seq_len
        
        # Reshape for processing
        x_flat = x.view(num_tokens, model_dim)  # [batch * seq, model_dim]
        
        # Route to experts
        top_k_weights, top_k_indices, load = self.router(x_flat)
        
        # Update statistics
        self._update_stats(load, top_k_weights, num_tokens)
        
        # Compute load balancing loss
        aux_loss = self.router.compute_load_balance_loss(
            load,
            num_tokens,
            self.config.capacity_factor,
        ) * self.config.load_balance_weight
        
        # Process each token through its experts
        output = self._process_experts(
            x_flat, top_k_weights, top_k_indices, num_tokens
        )
        
        # Reshape output
        output = output.view(batch_size, seq_len, model_dim)
        
        return output, aux_loss
    
    def _process_experts(
        self,
        x: torch.Tensor,
        top_k_weights: torch.Tensor,
        top_k_indices: torch.Tensor,
        num_tokens: int,
    ) -> torch.Tensor:
        """
        Process tokens through their assigned experts.
        
        This is the critical path - must be efficient!
        """
        # Initialize output
        output = torch.zeros_like(x)
        
        # Expert capacity
        capacity = int(num_tokens * self.config.capacity_factor / self.config.num_experts)
        
        # Process each expert
        for expert_idx in range(self.config.num_experts):
            # Find tokens assigned to this expert
            expert_mask = (top_k_indices == expert_idx).any(dim=-1)  # [num_tokens]
            
            if not expert_mask.any():
                continue
            
            # Get indices of tokens for this expert
            expert_token_ids = expert_mask.nonzero(as_tuple=True)[0]
            num_expert_tokens = len(expert_token_ids)
            
            # Cap at capacity
            if num_expert_tokens > capacity:
                expert_token_ids = expert_token_ids[:capacity]
                num_expert_tokens = capacity
            
            if num_expert_tokens == 0:
                continue
            
            # Get input for this expert
            expert_input = x[expert_token_ids]  # [num_expert_tokens, model_dim]
            
            # Forward through expert
            expert_output = self.experts[expert_idx](expert_input)  # [num_expert_tokens, model_dim]
            
            # Weight by routing weight
            # Find which position in top-k this expert occupies for each token
            for pos in range(self.config.top_k):
                pos_mask = (top_k_indices[expert_token_ids, pos] == expert_idx)
                if pos_mask.any():
                    weights = top_k_weights[expert_token_ids, pos][pos_mask]
                    weighted_output = expert_output[pos_mask] * weights.unsqueeze(-1)
                    output[expert_token_ids[pos_mask]] += weighted_output
        
        return output
    
    def _update_stats(
        self,
        load: torch.Tensor,
        routing_weights: torch.Tensor,
        num_tokens: int,
    ) -> None:
        """Update expert usage statistics."""
        if self.training:
            with torch.no_grad():
                self.stats.expert_counts = load.cpu()
                self.stats.routing_weights = routing_weights.mean(0).cpu()
                self.stats.expert_capacity = int(
                    num_tokens * self.config.capacity_factor / self.config.num_experts
                )
    
    def get_utilization(self) -> List[float]:
        """Get expert utilization as percentage."""
        if self.stats.expert_counts.sum() == 0:
            return [0.0] * self.config.num_experts
        
        utilization = (
            self.stats.expert_counts / max(self.stats.expert_counts.sum(), 1)
        ).tolist()
        
        return [u * 100 for u in utilization]
    
    def print_utilization(self) -> None:
        """Print expert utilization statistics."""
        utilization = self.get_utilization()
        
        logger.info("Expert Utilization:")
        for i, u in enumerate(utilization):
            bar = '█' * int(u / 5) + '░' * (20 - int(u / 5))
            logger.info(f"  Expert {i}: {bar} {u:.1f}%")


class DynamicExpertGeneration:
    """
    Dynamic Expert Generation - Generate experts on-demand instead of storing.
    
    KEY INNOVATION: Instead of storing 8 experts, generate them!
    
    ┌─────────────────────────────────────────────────────────────────────┐
    │                    TRADITIONAL vs DYNAMIC MoE                        │
    ├─────────────────────────────────────────────────────────────────────┤
    │                                                                      │
    │  Traditional:                                                       │
    │    ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐│
    │    │ E1  │ │ E2  │ │ E3  │ │ E4  │ │ E5  │ │ E6  │ │ E7  │ │ E8  ││
    │    │stored│ │stored│ │stored│ │stored│ │stored│ │stored│ │stored│ │stored││
    │    └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘│
    │                                                                      │
    │  Dynamic Generation:                                                │
    │    ┌──────────────────────────────────────────────────────┐         │
    │    │           EXPERT GENERATOR (small, stored)            │         │
    │    │     Input: expert_id, seed, task_embedding          │         │
    │    │     Output: expert_weights                           │         │
    │    └──────────────────────────────────────────────────────┘         │
    │                              │                                       │
    │              ┌───────────────┼───────────────┐                      │
    │              ▼               ▼               ▼                       │
    │          ┌─────┐         ┌─────┐         ┌─────┐                    │
    │          │ E3* │         │ E7* │         │ E1* │  ← Generated on-demand │
    │          └─────┘         └─────┘         └─────┘                    │
    │                                                                      │
    └─────────────────────────────────────────────────────────────────────┘
    """
    
    def __init__(
        self,
        model_dim: int,
        expert_dim: int,
        max_experts: int = 8,
        generator_dim: int = 512,
    ):
        self.model_dim = model_dim
        self.expert_dim = expert_dim
        self.max_experts = max_experts
        
        # Generator creates expert weights from seed
        # Input: expert_id (one-hot) + seed + task_embedding
        self.expert_generator = nn.Sequential(
            nn.Linear(max_experts + generator_dim + 64, generator_dim * 2),
            nn.GELU(),
            nn.Linear(generator_dim * 2, generator_dim * 2),
            nn.GELU(),
            nn.Linear(generator_dim * 2, expert_dim * model_dim),  # gate weight
        )
        
        self.up_generator = nn.Sequential(
            nn.Linear(max_experts + generator_dim + 64, generator_dim * 2),
            nn.GELU(),
            nn.Linear(generator_dim * 2, generator_dim * 2),
            nn.GELU(),
            nn.Linear(generator_dim * 2, expert_dim * model_dim),  # up weight
        )
        
        self.down_generator = nn.Sequential(
            nn.Linear(max_experts + generator_dim + 64, generator_dim * 2),
            nn.GELU(),
            nn.Linear(generator_dim * 2, generator_dim * 2),
            nn.GELU(),
            nn.Linear(generator_dim * 2, model_dim * expert_dim),  # down weight
        )
        
        # Cache for generated experts
        self.expert_cache: Dict[int, Tuple[nn.Parameter, nn.Parameter, nn.Parameter]] = {}
        self.cache_size_limit = 4  # Only keep 4 experts in memory
        
        logger.info(f"DynamicExpertGeneration initialized:")
        logger.info(f"  Max experts: {max_experts}")
        logger.info(f"  Expert dim: {expert_dim}")
        logger.info(f"  Cache size: {self.cache_size_limit}")
    
    def generate_expert(
        self,
        expert_id: int,
        seed: Optional[int] = None,
        task_embedding: Optional[torch.Tensor] = None,
    ) -> Tuple[nn.Parameter, nn.Parameter, nn.Parameter]:
        """
        Generate expert weights on-demand.
        
        Args:
            expert_id: Which expert to generate (0 to max_experts-1)
            seed: Random seed for reproducibility
            task_embedding: Optional task-specific conditioning
            
        Returns:
            Tuple of (gate_weight, up_weight, down_weight) as parameters
        """
        # Check cache first
        if expert_id in self.expert_cache:
            return self.expert_cache[expert_id]
        
        # Create input
        expert_onehot = torch.zeros(self.max_experts)
        expert_onehot[expert_id] = 1.0
        
        if seed is not None:
            torch.manual_seed(seed)
        
        random_seed = torch.randn(64)
        
        if task_embedding is not None:
            task_emb = task_embedding.flatten()
            if len(task_emb) > generator_dim:
                task_emb = task_emb[:generator_dim]
            elif len(task_emb) < generator_dim:
                task_emb = F.pad(task_emb, (0, generator_dim - len(task_emb)))
        else:
            task_emb = torch.zeros(generator_dim)
        
        generator_input = torch.cat([expert_onehot, task_emb, random_seed]).unsqueeze(0)
        
        # Generate weights
        gate_w = self.expert_generator(generator_input)
        gate_w = gate_w.view(self.expert_dim, self.model_dim)
        
        up_w = self.up_generator(generator_input)
        up_w = up_w.view(self.expert_dim, self.model_dim)
        
        down_w = self.down_generator(generator_input)
        down_w = down_w.view(self.model_dim, self.expert_dim)
        
        # Create parameters
        gate_param = nn.Parameter(gate_w)
        up_param = nn.Parameter(up_w)
        down_param = nn.Parameter(down_w)
        
        # Cache management - evict if full
        if len(self.expert_cache) >= self.cache_size_limit:
            # Remove oldest
            oldest = next(iter(self.expert_cache))
            del self.expert_cache[oldest]
        
        self.expert_cache[expert_id] = (gate_param, up_param, down_param)
        
        return gate_param, up_param, down_param
    
    def forward_expert(
        self,
        expert_id: int,
        x: torch.Tensor,
        seed: Optional[int] = None,
        task_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward through a generated expert.
        
        Args:
            expert_id: Which expert to use
            x: Input tensor
            seed: Random seed
            task_embedding: Task conditioning
            
        Returns:
            Expert output
        """
        gate, up, down = self.generate_expert(expert_id, seed, task_embedding)
        
        # Forward
        gate_out = F.silu(F.linear(x, gate))
        up_out = F.linear(x, up)
        intermediate = gate_out * up_out
        return F.linear(intermediate, down)


class HybridMoELayer(nn.Module):
    """
    Hybrid MoE layer combining stored and dynamic experts.
    
    Some experts are stored (for common cases),
    Some are generated on-demand (for rare cases).
    """
    
    def __init__(
        self,
        config: ExpertConfig,
        num_stored_experts: int = 4,
        num_dynamic_experts: int = 4,
    ):
        super().__init__()
        
        self.config = config
        self.num_stored = num_stored_experts
        self.num_dynamic = num_dynamic_experts
        
        # Stored experts
        self.stored_experts = nn.ModuleList([
            Expert(
                model_dim=config.model_dim,
                expert_dim=config.expert_hidden_dim,
            )
            for _ in range(num_stored_experts)
        ])
        
        # Dynamic expert generator
        self.dynamic_generator = DynamicExpertGeneration(
            model_dim=config.model_dim,
            expert_dim=config.expert_hidden_dim,
            max_experts=num_dynamic_experts,
        )
        
        # Router that can select from both stored and dynamic
        self.router = ExpertRouter(
            model_dim=config.model_dim,
            num_experts=num_stored_experts + num_dynamic_experts,
            top_k=config.top_k,
        )
        
        # Track which experts are dynamic
        self.dynamic_indices = set(range(num_stored_experts, num_stored_experts + num_dynamic_experts))
    
    def forward(
        self,
        x: torch.Tensor,
        task_embedding: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward with hybrid expert selection."""
        batch_size, seq_len, model_dim = x.shape
        num_tokens = batch_size * seq_len
        
        x_flat = x.view(num_tokens, model_dim)
        
        # Route
        top_k_weights, top_k_indices, load = self.router(x_flat)
        
        # Process stored experts
        output = torch.zeros_like(x_flat)
        
        # Stored experts
        for i, expert in enumerate(self.stored_experts):
            mask = (top_k_indices == i).any(dim=-1)
            if mask.any():
                expert_output = expert(x_flat[mask])
                # Apply weights
                for pos in range(self.config.top_k):
                    pos_mask = mask & (top_k_indices[:, pos] == i)
                    if pos_mask.any():
                        weights = top_k_weights[pos_mask, pos].unsqueeze(-1)
                        output[pos_mask] += expert_output[pos_mask.nonzero(as_tuple=True)[0]] * weights
        
        # Dynamic experts
        for dyn_id in range(self.num_dynamic):
            stored_id = self.num_stored + dyn_id
            mask = (top_k_indices == stored_id).any(dim=-1)
            if mask.any():
                # Generate expert weights
                token_indices = mask.nonzero(as_tuple=True)[0]
                for idx in token_indices:
                    expert_output = self.dynamic_generator.forward_expert(
                        dyn_id, x_flat[idx:idx+1], task_embedding=task_embedding
                    )
                    # Find routing weight
                    for pos in range(self.config.top_k):
                        if (top_k_indices[idx, pos] == stored_id).any():
                            weight = top_k_weights[idx, pos]
                            output[idx] += expert_output.squeeze(0) * weight
        
        return output.view(batch_size, seq_len, model_dim), torch.tensor(0.0)
