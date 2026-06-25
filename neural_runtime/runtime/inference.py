"""
Inference Engine
Main execution engine for Neural Runtime.
"""

import os
import gc
import time
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from pathlib import Path
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, Future
import logging

import torch
import torch.nn as nn
from torch import Tensor
from transformers import GenerationConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

from ..core.model import NeuralRuntimeModel, ModelConfig
from ..memory.hierarchy import HierarchicalMemory, MemoryLevel
from ..cache.predictor import AccessPatternPredictor, AdaptivePrefetcher

logger = logging.getLogger(__name__)


@dataclass
class InferenceStats:
    """Statistics for an inference run."""
    total_time_ms: float = 0.0
    prefill_time_ms: float = 0.0
    decode_time_ms: float = 0.0
    memory_load_time_ms: float = 0.0
    compute_time_ms: float = 0.0
    vram_peak_gb: float = 0.0
    layers_processed: int = 0
    
    @property
    def tokens_per_second(self) -> float:
        """Calculate tokens per second."""
        if self.decode_time_ms > 0:
            return 1000.0 / self.decode_time_ms
        return 0.0


class InferenceEngine:
    """
    Main inference engine for Neural Runtime.
    
    Features:
    - Hierarchical memory management
    - Intelligent prefetching
    - Adaptive batching
    - KV cache management
    - Multi-GPU support (via tensor parallelism)
    
    Usage:
        engine = InferenceEngine(model)
        output = engine.generate("Hello, world!")
    """
    
    def __init__(
        self,
        model: NeuralRuntimeModel,
        max_batch_size: int = 1,
        enable_profiling: bool = False,
    ):
        self.model = model
        self.max_batch_size = max_batch_size
        self.enable_profiling = enable_profiling
        
        # Components
        self.memory = getattr(model, 'memory', None)
        self.predictor = getattr(model, 'predictor', None)
        
        # Prefetcher
        if self.predictor:
            self.prefetcher = AdaptivePrefetcher(self.predictor)
        else:
            self.prefetcher = None
        
        # Thread pool for async loading
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._pending_loads: Dict[str, Future] = {}
        
        # Stats
        self.stats = InferenceStats()
        
        # Cache for kv_cache
        self._kv_cache: Optional[List[Tuple[Tensor, Tensor]]] = None
        
        logger.info(f"InferenceEngine initialized")
    
    def generate(
        self,
        prompt: Union[str, List[str]],
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        do_sample: bool = True,
        **kwargs
    ) -> Union[str, List[str]]:
        """
        Generate text from prompt(s).
        
        Args:
            prompt: Input text or list of texts
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Repetition penalty
            do_sample: Whether to sample (False = greedy)
            
        Returns:
            Generated text(s)
        """
        start_time = time.time()
        
        # Tokenize
        if isinstance(prompt, str):
            prompts = [prompt]
        else:
            prompts = prompt
        
        # Tokenize all prompts
        input_ids_list = []
        for p in prompts:
            tokens = self.model.tokenizer(
                p,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.model.config.max_sequence_length
            )
            input_ids_list.append(tokens['input_ids'].squeeze(0))
        
        # Pad to same length
        max_len = max(ids.numel() for ids in input_ids_list)
        padded = []
        for ids in input_ids_list:
            if ids.numel() < max_len:
                pad = torch.full((max_len - ids.numel(),), 
                               self.model.tokenizer.pad_token_id,
                               dtype=ids.dtype)
                ids = torch.cat([ids, pad])
            padded.append(ids)
        
        input_ids = torch.stack(padded).to(self.model.device)
        
        # Generate
        self._reset_stats()
        
        with torch.inference_mode():
            # Prefill phase
            prefill_start = time.time()
            outputs = self._prefill(input_ids)
            self.stats.prefill_time_ms = (time.time() - prefill_start) * 1000
            
            # Decode phase (token by token)
            decode_start = time.time()
            generated_ids = input_ids.clone()
            
            past_key_values = outputs.past_key_values
            
            for step in range(max_new_tokens):
                # Get next token
                logits = outputs.logits[:, -1:, :]
                
                # Apply sampling
                if do_sample:
                    next_token = self._sample(
                        logits,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        repetition_penalty=repetition_penalty,
                    )
                else:
                    next_token = logits.argmax(dim=-1)
                
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                
                # Check for EOS
                if (next_token == self.model.tokenizer.eos_token_id).all():
                    break
                
                # Decode step
                outputs = self._decode_step(
                    next_token,
                    past_key_values=past_key_values,
                )
                past_key_values = outputs.past_key_values
            
            self.stats.decode_time_ms = (time.time() - decode_start) * 1000
        
        self.stats.total_time_ms = (time.time() - start_time) * 1000
        
        # Decode output
        outputs_text = []
        for ids in generated_ids:
            text = self.model.tokenizer.decode(ids, skip_special_tokens=True)
            outputs_text.append(text)
        
        if isinstance(prompt, str):
            return outputs_text[0]
        return outputs_text
    
    def _prefill(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> CausalLMOutputWithPast:
        """
        Prefill phase - process input tokens.
        """
        # Create attention mask if not provided
        if attention_mask is None:
            seq_len = input_ids.shape[1]
            attention_mask = torch.ones(
                seq_len, seq_len,
                dtype=torch.bool,
                device=self.model.device
            ).triu(diagonal=1)
            attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
        
        # Position IDs
        position_ids = torch.arange(
            input_ids.shape[1],
            device=self.model.device
        ).unsqueeze(0)
        
        # Forward through model
        outputs = self._forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
        )
        
        return outputs
    
    def _decode_step(
        self,
        input_ids: Tensor,
        past_key_values: Optional[List] = None,
    ) -> CausalLMOutputWithPast:
        """
        Decode step - generate one token.
        """
        seq_len = input_ids.shape[1]
        
        # Position IDs
        past_len = past_key_values[0][0].shape[2] if past_key_values else 0
        position_ids = torch.arange(
            past_len, past_len + seq_len,
            device=self.model.device
        ).unsqueeze(0)
        
        # Attention mask (causal)
        attention_mask = torch.ones(
            past_len + seq_len, past_len + seq_len,
            dtype=torch.bool,
            device=self.model.device
        ).triu(diagonal=1)
        attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
        
        outputs = self._forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
        
        return outputs
    
    def _forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_values: Optional[List] = None,
        use_cache: bool = True,
        **kwargs
    ) -> CausalLMOutputWithPast:
        """
        Forward pass through the model.
        """
        # This is a placeholder - actual implementation depends on model structure
        # For now, delegate to model's forward
        
        return self.model.model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs
        )
    
    def _sample(
        self,
        logits: Tensor,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
    ) -> Tensor:
        """
        Sample next token from logits.
        """
        # Apply temperature
        if temperature != 1.0:
            logits = logits / temperature
        
        # Apply repetition penalty (simplified)
        # In practice, this should track previously generated tokens
        
        # Top-k filtering
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            values, indices = torch.topk(logits, top_k)
            logits = torch.full_like(logits, float('-inf'))
            logits.scatter_(-1, indices, values)
        
        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumsum_probs = torch.cumsum(
                torch.softmax(sorted_logits, dim=-1),
                dim=-1
            )
            
            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumsum_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False
            
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')
        
        # Sample
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs.squeeze(1), num_samples=1)
        
        return next_token
    
    def _reset_stats(self) -> None:
        """Reset inference statistics."""
        self.stats = InferenceStats()
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    
    def _get_vram_peak(self) -> float:
        """Get peak VRAM usage."""
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024**3)
        return 0.0
    
    def get_stats(self) -> InferenceStats:
        """Get current inference statistics."""
        self.stats.vram_peak_gb = self._get_vram_peak()
        return self.stats


class LayeredInferenceEngine(InferenceEngine):
    """
    Layer-by-layer inference engine (inspired by AirLLM).
    
    This engine processes layers one at a time with intelligent
    memory management and prefetching.
    
    Key differences from base InferenceEngine:
    - Explicit layer-by-layer processing
    - Memory-aware layer loading/unloading
    - Predictive prefetching
    """
    
    def __init__(
        self,
        model: NeuralRuntimeModel,
        max_batch_size: int = 1,
        prefetch_distance: int = 3,
        enable_profiling: bool = False,
    ):
        super().__init__(model, max_batch_size, enable_profiling)
        
        self.prefetch_distance = prefetch_distance
        
        # Layer management
        self.current_layer = 0
        self.layer_states: Dict[int, Dict] = {}
        
        logger.info(f"LayeredInferenceEngine initialized (prefetch={prefetch_distance})")
    
    def _forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_values: Optional[List] = None,
        use_cache: bool = True,
        **kwargs
    ) -> CausalLMOutputWithPast:
        """
        Layer-by-layer forward pass.
        """
        # Track memory
        mem_start = time.time()
        
        # Initialize batch
        batch = [input_ids_unit.unsqueeze(0) for input_ids_unit in input_ids]
        
        # Initialize KV cache list
        kv_cache_list = [] if use_cache else None
        num_layers = len(self.model.layer_info)
        
        for i, layer_info in enumerate(self.model.layer_info):
            layer_name = layer_info.name
            
            # Prefetch next layers
            if self.prefetcher and i < num_layers - self.prefetch_distance:
                next_layers = self.prefetcher.predictor.predict_next(i, self.prefetch_distance)
                for layer in next_layers:
                    if layer not in self._pending_loads:
                        self._pending_loads[layer] = self._executor.submit(
                            self._load_layer, layer
                        )
            
            # Load layer
            layer_state = self._load_layer(layer_name)
            self.stats.memory_load_time_ms += layer_state['load_time']
            
            # Execute layer
            layer = layer_state['layer']
            
            for j, seq in enumerate(batch):
                if layer_info.is_embedding:
                    batch[j] = layer(seq)
                elif layer_info.is_transformer:
                    outputs = layer(
                        seq,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values[i] if past_key_values else None,
                        use_cache=use_cache,
                    )
                    new_seq = outputs[0]
                    
                    if use_cache and len(outputs) > 1:
                        kv = outputs[1] if isinstance(outputs[1], tuple) else (outputs[1], outputs[2])
                        if kv_cache_list is not None:
                            kv_cache_list.append(kv)
                    
                    batch[j] = new_seq
                elif layer_info.is_norm:
                    batch[j] = layer(seq)
                elif layer_info.is_lm_head:
                    batch[j] = layer(seq)
            
            # Unload layer
            self._unload_layer(layer_name)
            
            self.stats.layers_processed += 1
        
        self.stats.memory_load_time_ms += (time.time() - mem_start) * 1000
        
        # Combine batch
        logits = torch.cat(batch, 0)
        
        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=tuple(kv_cache_list) if kv_cache_list else None,
        )
    
    def _load_layer(self, layer_name: str) -> Dict[str, Any]:
        """
        Load a layer into memory.
        """
        load_start = time.time()
        
        # Check if already loaded
        if layer_name in self.layer_states:
            self.layer_states[layer_name]['last_access'] = time.time()
            return self.layer_states[layer_name]
        
        # Try to get from memory system
        layer_data = None
        if self.memory:
            layer_data = self.memory.get(layer_name)
        
        # If not in memory, load from disk/model
        if layer_data is None:
            layer_data = self._load_layer_from_source(layer_name)
        
        load_time = (time.time() - load_start) * 1000
        
        state = {
            'layer': layer_data,
            'load_time': load_time,
            'last_access': time.time(),
        }
        
        self.layer_states[layer_name] = state
        
        return state
    
    def _load_layer_from_source(self, layer_name: str) -> nn.Module:
        """Load a layer from the model or checkpoint."""
        # Find layer in model
        layer_idx = None
        for i, info in enumerate(self.model.layer_info):
            if info.name == layer_name:
                layer_idx = i
                break
        
        if layer_idx is None:
            raise ValueError(f"Unknown layer: {layer_name}")
        
        # Return layer from model's layers list
        # This is a simplified version
        return self.model.model.model.layers[layer_idx]
    
    def _unload_layer(self, layer_name: str) -> None:
        """
        Unload a layer from memory.
        """
        if layer_name in self.layer_states:
            state = self.layer_states[layer_name]
            
            # Move layer to meta device
            if hasattr(state['layer'], 'to'):
                state['layer'].to('meta')
            
            # Clear from states
            del self.layer_states[layer_name]
            
            # Update memory system
            if self.memory:
                self.memory.store(layer_name, None, MemoryLevel.HBM)
            
            # Clean memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def _cleanup(self) -> None:
        """Cleanup resources."""
        # Wait for pending loads
        for future in self._pending_loads.values():
            future.cancel()
        
        self._pending_loads.clear()
        self.layer_states.clear()
        gc.collect()
