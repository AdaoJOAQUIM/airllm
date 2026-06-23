"""
Task Scheduler
============

Schedules neural computation tasks based on priority, resources, and dependencies.
"""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
from queue import PriorityQueue
import time
import threading

import torch
import logging

logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """Task priority levels."""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5


@dataclass
class ScheduledTask:
    """A task scheduled for execution."""
    task_id: str
    priority: TaskPriority
    data: Any
    dependencies: List[str]
    created_at: float
    estimated_duration_ms: float
    status: str = "pending"


class TaskScheduler:
    """
    Scheduler for neural computation tasks.
    
    Features:
    - Priority-based scheduling
    - Dependency management
    - Resource tracking
    - Concurrent execution
    """
    
    def __init__(self, max_concurrent: int = 4):
        self.max_concurrent = max_concurrent
        self.queue = PriorityQueue()
        self.running_tasks: Dict[str, ScheduledTask] = {}
        self.completed_tasks: Dict[str, ScheduledTask] = {}
        self.lock = threading.Lock()
        
        # Resource tracking
        self.estimated_vram_usage: float = 0.0
        self.estimated_ram_usage: float = 0.0
    
    def schedule(
        self,
        task_id: str,
        data: Any,
        priority: TaskPriority = TaskPriority.NORMAL,
        dependencies: Optional[List[str]] = None,
        estimated_duration_ms: float = 1000.0,
    ) -> None:
        """Schedule a task for execution."""
        task = ScheduledTask(
            task_id=task_id,
            priority=priority,
            data=data,
            dependencies=dependencies or [],
            created_at=time.time(),
            estimated_duration_ms=estimated_duration_ms,
        )
        
        self.queue.put((priority.value, task))
        
        logger.debug(f"Scheduled task {task_id} with priority {priority.name}")
    
    def get_next_task(self) -> Optional[ScheduledTask]:
        """Get the next task ready for execution."""
        with self.lock:
            # Check concurrent limit
            if len(self.running_tasks) >= self.max_concurrent:
                return None
            
            # Peek at queue
            if self.queue.empty():
                return None
            
            priority, task = self.queue.get()
            
            # Check dependencies
            for dep_id in task.dependencies:
                if dep_id not in self.completed_tasks:
                    # Put back in queue
                    self.queue.put((priority, task))
                    return None
            
            # Start task
            task.status = "running"
            self.running_tasks[task.task_id] = task
            
            return task
    
    def complete_task(self, task_id: str, result: Any = None) -> None:
        """Mark a task as completed."""
        with self.lock:
            if task_id in self.running_tasks:
                task = self.running_tasks.pop(task_id)
                task.status = "completed"
                self.completed_tasks[task_id] = task
                
                logger.debug(f"Completed task {task_id}")
    
    def cancel_task(self, task_id: str) -> None:
        """Cancel a task."""
        with self.lock:
            if task_id in self.running_tasks:
                task = self.running_tasks.pop(task_id)
                task.status = "cancelled"
                self.completed_tasks[task_id] = task
    
    def get_status(self) -> Dict[str, Any]:
        """Get scheduler status."""
        return {
            "queued": self.queue.qsize(),
            "running": len(self.running_tasks),
            "completed": len(self.completed_tasks),
            "max_concurrent": self.max_concurrent,
        }


class ContinuousBatcher:
    """
    Continuous batching for efficient inference.
    
    Batches requests dynamically as they arrive.
    """
    
    def __init__(
        self,
        max_batch_size: int = 8,
        max_wait_ms: float = 50.0,
    ):
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.pending_requests: List[Tuple[str, Any]] = []
        self.lock = threading.Lock()
        self.last_batch_time: float = 0
    
    def add_request(
        self,
        request_id: str,
        data: Any,
    ) -> None:
        """Add a request to the batch."""
        with self.lock:
            self.pending_requests.append((request_id, data))
    
    def get_batch(self) -> Optional[Tuple[List, float]]:
        """
        Get a batch if ready.
        
        Returns:
            Tuple of (batch_data, wait_time_ms) or None
        """
        with self.lock:
            if not self.pending_requests:
                return None
            
            # Check if batch is full
            if len(self.pending_requests) >= self.max_batch_size:
                batch = self.pending_requests[:self.max_batch_size]
                self.pending_requests = self.pending_requests[self.max_batch_size:]
                return [r[1] for r in batch], 0.0
            
            # Check timeout
            wait_time = (time.time() - self.last_batch_time) * 1000
            if wait_time >= self.max_wait_ms and self.pending_requests:
                batch = self.pending_requests
                self.pending_requests = []
                self.last_batch_time = time.time()
                return [r[1] for r in batch], wait_time
            
            return None
    
    def size(self) -> int:
        """Get number of pending requests."""
        with self.lock:
            return len(self.pending_requests)


class AdaptiveScheduler:
    """
    Scheduler that adapts based on system state.
    
    Adjusts batching and priority based on:
    - GPU utilization
    - Memory pressure
    - Request patterns
    """
    
    def __init__(self, base_scheduler: TaskScheduler):
        self.scheduler = base_scheduler
        
        # Adaptive parameters
        self.gpu_utilization_target = 0.9
        self.memory_threshold_gb = 20.0
        
        # Monitoring
        self.gpu_utilization_history: List[float] = []
        self.memory_history: List[float] = []
    
    def update_system_state(self) -> Dict[str, float]:
        """Update and return current system state."""
        state = {}
        
        # GPU utilization
        if torch.cuda.is_available():
            # Rough estimate based on memory
            mem_used = torch.cuda.memory_allocated() / (1024**3)
            mem_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            gpu_util = mem_used / mem_total if mem_total > 0 else 0
            state["gpu_utilization"] = gpu_util
            self.gpu_utilization_history.append(gpu_util)
        else:
            state["gpu_utilization"] = 0.0
        
        # Memory usage
        try:
            import psutil
            mem = psutil.virtual_memory()
            state["ram_used_gb"] = mem.used / (1024**3)
        except:
            state["ram_used_gb"] = 0.0
        
        # Keep history bounded
        if len(self.gpu_utilization_history) > 100:
            self.gpu_utilization_history = self.gpu_utilization_history[-50:]
        
        return state
    
    def should_adjust(self) -> bool:
        """Check if scheduling should be adjusted."""
        if not self.gpu_utilization_history:
            return False
        
        recent_util = self.gpu_utilization_history[-10:]
        avg_util = sum(recent_util) / len(recent_util)
        
        # Adjust if far from target
        return abs(avg_util - self.gpu_utilization_target) > 0.2
    
    def get_adjusted_batch_size(self) -> int:
        """Get batch size adjusted for current state."""
        state = self.update_system_state()
        gpu_util = state["gpu_utilization"]
        
        # Reduce batch if high utilization
        if gpu_util > 0.95:
            return max(1, self.scheduler.max_concurrent // 2)
        
        # Increase if low utilization
        if gpu_util < 0.5:
            return min(16, self.scheduler.max_concurrent * 2)
        
        return self.scheduler.max_concurrent
