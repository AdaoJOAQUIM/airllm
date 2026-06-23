"""
Tensor Parallelism Module
Enables running large models across multiple GPUs.

Based on Megatron-LM style tensor parallelism for transformers.
"""

import os
import gc
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed import ProcessGroup

logger = logging.getLogger(__name__)


class TensorParallelConfig:
    """Configuration for tensor parallelism."""
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    
    # Memory optimization
    use_sequence_parallel: bool = False
    use_flash_attention: bool = True
    
    # Communication
    gradient_checkpointing: bool = False


class TensorParallelManager:
    """
    Manages tensor parallelism for distributed inference.
    
    Features:
    - Column and row parallelism for linear layers
    - Attention parallelism across heads
    - All-reduce communications
    - Gradient checkpointing support
    
    Usage:
        # Initialize
        tp_manager = TensorParallelManager(
            tensor_parallel_size=4,
            world_size=4,
            rank=0
        )
        
        # Wrap a model
        distributed_model = tp_manager.wrap_model(model)
        
        # Run inference
        output = distributed_model(input_ids)
    """
    
    def __init__(
        self,
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        world_size: int = 1,
        rank: int = 0,
        backend: str = "nccl",
    ):
        self.tensor_parallel_size = tensor_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size
        self.world_size = world_size
        self.rank = rank
        self.backend = backend
        
        self.group = None
        self.process_group = None
        
        # Initialize distributed if needed
        if world_size > 1:
            self._init_distributed()
        
        # Module replacements
        self._replaced_modules: List[nn.Module] = []
        
        logger.info(f"TensorParallelManager initialized: TP={tensor_parallel_size}, PP={pipeline_parallel_size}")
    
    def _init_distributed(self) -> None:
        """Initialize torch distributed."""
        if not dist.is_initialized():
            dist.init_process_group(
                backend=self.backend,
                world_size=self.world_size,
                rank=self.rank,
            )
        
        # Create process group for tensor parallelism
        ranks = list(range(self.tensor_parallel_size))
        self.process_group = dist.new_group(ranks)
    
    def wrap_model(self, model: nn.Module) -> nn.Module:
        """
        Wrap a model with tensor parallelism.
        
        Args:
            model: The model to wrap
            
        Returns:
            Wrapped model
        """
        # Apply tensor parallelism to linear layers
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                if "query" in name or "key" in name or "value" in name:
                    # Attention projection - column parallel
                    self._wrap_column_parallel(module, name)
                elif "dense" in name or "wo" in name:
                    # Output projection - row parallel
                    self._wrap_row_parallel(module, name)
                elif "gate_proj" in name or "up_proj" in name:
                    # FFN up projection - column parallel
                    self._wrap_column_parallel(module, name)
                elif "down_proj" in name:
                    # FFN down projection - row parallel
                    self._wrap_row_parallel(module, name)
        
        return model
    
    def _wrap_column_parallel(self, module: nn.Module, name: str) -> None:
        """
        Wrap a linear layer with column parallelism.
        
        Each rank holds a portion of the output features.
        Output is All-Gathered after computation.
        """
        wrapped = ColumnParallelLinear(
            in_features=module.in_features,
            out_features=module.out_features,
            tensor_parallel_size=self.tensor_parallel_size,
            process_group=self.process_group,
        )
        
        # Copy weights
        with torch.no_grad():
            # Split weights along output dimension
            chunk = wrapped.weight.data
            original_weight = module.weight.data
            
            # Distribute weights
            split_size = original_weight.shape[0] // self.tensor_parallel_size
            rank = self.rank
            
            wrapped.weight.data = original_weight[rank * split_size:(rank + 1) * split_size].clone()
            if module.bias is not None:
                wrapped.bias = module.bias[rank * split_size:(rank + 1) * split_size].clone()
        
        self._replace_module(module, wrapped, name)
        self._replaced_modules.append(module)
    
    def _wrap_row_parallel(self, module: nn.Module, name: str) -> None:
        """
        Wrap a linear layer with row parallelism.
        
        Each rank holds a portion of the input features.
        Output is All-Reduced after computation.
        """
        wrapped = RowParallelLinear(
            in_features=module.in_features,
            out_features=module.out_features,
            tensor_parallel_size=self.tensor_parallel_size,
            process_group=self.process_group,
        )
        
        # Copy weights
        with torch.no_grad():
            # Split weights along input dimension
            split_size = module.weight.shape[1] // self.tensor_parallel_size
            rank = self.rank
            
            wrapped.weight.data = module.weight.data[:, rank * split_size:(rank + 1) * split_size].clone()
        
        self._replace_module(module, wrapped, name)
        self._replaced_modules.append(module)
    
    def _replace_module(
        self,
        original: nn.Module,
        replacement: nn.Module,
        name: str,
    ) -> None:
        """Replace a module in the model."""
        parts = name.rsplit('.', 1)
        if len(parts) == 2:
            parent_name, attr_name = parts
            parent = original
            for part in parent_name.split('.'):
                parent = getattr(parent, part)
            setattr(parent, attr_name, replacement)
        else:
            setattr(original, name, replacement)
    
    def cleanup(self) -> None:
        """Cleanup distributed resources."""
        if dist.is_initialized():
            dist.destroy_process_group()


class ColumnParallelLinear(nn.Module):
    """
    Linear layer with column parallelism.
    
    Splits the output features across ranks.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        tensor_parallel_size: int = 1,
        process_group: Optional[ProcessGroup] = None,
        bias: bool = True,
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.tp_size = tensor_parallel_size
        self.process_group = process_group
        
        # Each rank has out_features // tp_size
        self.out_features_per_rank = out_features // tensor_parallel_size
        
        # Local weight
        self.weight = nn.Parameter(
            torch.empty(self.out_features_per_rank, in_features)
        )
        
        if bias:
            self.bias = nn.Parameter(
                torch.empty(self.out_features_per_rank)
            )
        else:
            self.register_parameter('bias', None)
        
        # Initialize
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with all-gather."""
        # Local computation
        output = torch.nn.functional.linear(x, self.weight, self.bias)
        
        # All-gather outputs
        if self.tp_size > 1:
            outputs = [torch.empty_like(output) for _ in range(self.tp_size)]
            dist.all_gather(outputs, output, group=self.process_group)
            output = torch.cat(outputs, dim=-1)
        
        return output


class RowParallelLinear(nn.Module):
    """
    Linear layer with row parallelism.
    
    Splits the input features across ranks.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        tensor_parallel_size: int = 1,
        process_group: Optional[ProcessGroup] = None,
        bias: bool = True,
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.tp_size = tensor_parallel_size
        self.process_group = process_group
        
        # Each rank has in_features // tp_size
        self.in_features_per_rank = in_features // tensor_parallel_size
        
        # Local weight
        self.weight = nn.Parameter(
            torch.empty(out_features, self.in_features_per_rank)
        )
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Initialize
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with all-reduce."""
        # Split input
        if self.tp_size > 1:
            x_chunks = torch.chunk(x, self.tp_size, dim=-1)
            x = x_chunks[self.tp_size - 1]  # Use last chunk
        
        # Local computation (without bias)
        output = torch.nn.functional.linear(x, self.weight)
        
        # All-reduce
        if self.tp_size > 1:
            dist.all_reduce(output, group=self.process_group)
        
        # Add bias
        if self.bias is not None:
            output = output + self.bias
        
        return output


class SequenceParallelLinear(nn.Module):
    """
    Linear layer with sequence parallelism.
    
    Splits the sequence dimension across ranks.
    Used with use_sequence_parallel=True.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        sequence_parallel_size: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.sp_size = sequence_parallel_size
        
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features)
        )
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward with sequence parallelism.
        
        Input: [batch, seq // sp_size, hidden]
        Output: [batch, seq // sp_size, hidden]  # Gathered later if needed
        """
        output = torch.nn.functional.linear(x, self.weight, self.bias)
        
        # All-gather along sequence dimension
        if self.sp_size > 1:
            outputs = [torch.empty_like(output) for _ in range(self.sp_size)]
            dist.all_gather(outputs, output)
            output = torch.cat(outputs, dim=1)
        
        return output
