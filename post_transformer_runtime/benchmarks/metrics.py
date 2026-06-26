"""
Measurement primitives for the Post-Transformer Runtime benchmark harness.

Design contract (from docs/AIRLLM_AUTOPSY.md, Phase 0 guardrails):

  * NEVER fabricate a metric. If a quantity cannot be measured on the current
    machine (no GPU, no RAPL energy counters, ...), the corresponding field is
    returned as ``None`` -- not as a guess.
  * Stdlib-only fallbacks. ``psutil`` / ``torch`` / ``nvidia-smi`` are used when
    present, but everything degrades to the standard library (``resource``,
    ``/proc``, ``/sys``) so the harness runs on a bare Raspberry Pi too.
  * The unit of every field is encoded in its name (``_bytes``, ``_joules``,
    ``_seconds``).

This module has no third-party hard dependency and is importable (and
self-testable) without torch installed.
"""

from __future__ import annotations

import os
import time
import threading
import subprocess
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


# --------------------------------------------------------------------------- #
# Optional dependencies, probed once.                                         #
# --------------------------------------------------------------------------- #

try:
    import psutil  # type: ignore

    _HAS_PSUTIL = True
except Exception:  # pragma: no cover - environment dependent
    _HAS_PSUTIL = False

try:
    import torch  # type: ignore

    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


# --------------------------------------------------------------------------- #
# RAM (resident set size).                                                    #
# --------------------------------------------------------------------------- #

def process_rss_bytes() -> Optional[int]:
    """Resident set size of the current process, in bytes, or None.

    Tries psutil, then /proc/self/statm (Linux), then resource.getrusage
    (ru_maxrss, which is already a *peak* and is in KiB on Linux / bytes on
    macOS). Returns None only if every method fails.
    """
    if _HAS_PSUTIL:
        try:
            return int(psutil.Process(os.getpid()).memory_info().rss)
        except Exception:
            pass

    # Linux: /proc/self/statm -> resident pages * page size
    try:
        with open("/proc/self/statm", "r") as fh:
            resident_pages = int(fh.read().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE")
    except Exception:
        pass

    # Portable last resort: peak RSS via getrusage.
    try:
        import resource
        import sys

        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports KiB, macOS/BSD report bytes.
        return ru * 1024 if sys.platform != "darwin" else ru
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# VRAM (NVIDIA only for now).                                                  #
# --------------------------------------------------------------------------- #

def gpu_mem_used_bytes() -> Optional[int]:
    """Currently used GPU memory in bytes, or None if no GPU is visible.

    Prefers torch (reports the allocator's view) and falls back to parsing
    ``nvidia-smi``. Returns None on machines with no NVIDIA GPU (e.g. a Pi).
    """
    if _HAS_TORCH:
        try:
            if torch.cuda.is_available():
                # memory_reserved reflects what the caching allocator holds,
                # which is closer to real footprint than memory_allocated.
                return int(torch.cuda.memory_reserved())
        except Exception:
            pass

    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        # Sum across GPUs, values are in MiB.
        mib = sum(int(x) for x in out.decode().split())
        return mib * 1024 * 1024
    except Exception:
        return None


def torch_reset_peak_vram() -> None:
    """Reset torch's peak VRAM counter, if torch+cuda are available."""
    if _HAS_TORCH:
        try:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass


def torch_peak_vram_bytes() -> Optional[int]:
    """torch's recorded peak reserved VRAM since the last reset, or None."""
    if _HAS_TORCH:
        try:
            if torch.cuda.is_available():
                return int(torch.cuda.max_memory_reserved())
        except Exception:
            pass
    return None


# --------------------------------------------------------------------------- #
# Disk footprint.                                                              #
# --------------------------------------------------------------------------- #

def dir_size_bytes(path: str | os.PathLike) -> Optional[int]:
    """Total size of all files under ``path`` (following the tree), or None.

    Used to measure the on-disk footprint of the splitted model -- the storage
    cost that AirLLM does *not* reduce (Phase 0, section 2.6).
    """
    p = Path(path)
    if not p.exists():
        return None
    total = 0
    try:
        for root, _dirs, files in os.walk(p):
            for name in files:
                fp = Path(root) / name
                try:
                    total += fp.stat().st_size
                except OSError:
                    continue
        return total
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Energy (Intel RAPL CPU package + NVIDIA GPU integration).                    #
# --------------------------------------------------------------------------- #

_RAPL_ROOT = "/sys/class/powercap"


def _read_rapl_energy_uj() -> Optional[int]:
    """Sum of all Intel RAPL package energy counters in microjoules, or None."""
    root = Path(_RAPL_ROOT)
    if not root.exists():
        return None
    total = 0
    found = False
    try:
        for entry in root.glob("intel-rapl:*"):
            ej = entry / "energy_uj"
            if ej.exists():
                try:
                    total += int(ej.read_text().strip())
                    found = True
                except (OSError, ValueError):
                    continue
        return total if found else None
    except Exception:
        return None


class EnergyMeter:
    """Best-effort energy measurement across a code block.

    CPU energy via Intel RAPL (microjoule counters, handles the documented
    32-bit-ish wraparound by clamping negative deltas to None). GPU energy by
    integrating ``nvidia-smi`` instantaneous power draw on a background thread.

    Every field is None when the corresponding counter is unavailable -- a Pi
    has neither RAPL nor nvidia-smi, and that is reported honestly.
    """

    def __init__(self, gpu_poll_seconds: float = 0.1):
        self._gpu_poll = gpu_poll_seconds
        self._rapl_start: Optional[int] = None
        self._gpu_energy_j = 0.0
        self._gpu_samples = 0
        self._gpu_thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.cpu_energy_joules: Optional[float] = None
        self.gpu_energy_joules: Optional[float] = None

    def _gpu_loop(self) -> None:
        last = time.time()
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=power.draw",
                     "--format=csv,noheader,nounits"],
                    stderr=subprocess.DEVNULL, timeout=2,
                )
                watts = sum(float(x) for x in out.decode().split())
                now = time.time()
                self._gpu_energy_j += watts * (now - last)
                last = now
                self._gpu_samples += 1
            except Exception:
                # GPU power query failed -> give up on GPU energy entirely.
                self._gpu_samples = -1
                return
            self._stop.wait(self._gpu_poll)

    def __enter__(self) -> "EnergyMeter":
        self._rapl_start = _read_rapl_energy_uj()
        # Probe nvidia-smi once before spawning the thread.
        if gpu_mem_used_bytes() is not None:
            self._stop.clear()
            self._gpu_thread = threading.Thread(target=self._gpu_loop, daemon=True)
            self._gpu_thread.start()
        return self

    def __exit__(self, *exc) -> None:
        rapl_end = _read_rapl_energy_uj()
        if self._rapl_start is not None and rapl_end is not None:
            delta_uj = rapl_end - self._rapl_start
            # Negative delta => counter wraparound; we can't recover it cleanly.
            self.cpu_energy_joules = delta_uj / 1e6 if delta_uj >= 0 else None

        if self._gpu_thread is not None:
            self._stop.set()
            self._gpu_thread.join(timeout=3)
            if self._gpu_samples > 0:
                self.gpu_energy_joules = self._gpu_energy_j


# --------------------------------------------------------------------------- #
# Peak-RAM / peak-VRAM sampler (background polling).                           #
# --------------------------------------------------------------------------- #

class PeakMemorySampler:
    """Context manager that polls RSS (and GPU memory) to capture peaks.

    AirLLM's whole point is a low *peak* memory; a single before/after read can
    miss the spike during a layer's matmul. This samples on a background thread.
    For VRAM, torch's own peak counter is more accurate, so we reset it on enter
    and read it on exit in addition to the polled value.
    """

    def __init__(self, poll_seconds: float = 0.05):
        self._poll = poll_seconds
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.peak_rss_bytes: Optional[int] = None
        self.peak_gpu_bytes: Optional[int] = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            rss = process_rss_bytes()
            if rss is not None:
                self.peak_rss_bytes = rss if self.peak_rss_bytes is None \
                    else max(self.peak_rss_bytes, rss)
            gpu = gpu_mem_used_bytes()
            if gpu is not None:
                self.peak_gpu_bytes = gpu if self.peak_gpu_bytes is None \
                    else max(self.peak_gpu_bytes, gpu)
            self._stop.wait(self._poll)

    def __enter__(self) -> "PeakMemorySampler":
        torch_reset_peak_vram()
        self.peak_rss_bytes = process_rss_bytes()
        self.peak_gpu_bytes = gpu_mem_used_bytes()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        # Prefer torch's exact peak if it exceeds what polling caught.
        torch_peak = torch_peak_vram_bytes()
        if torch_peak is not None:
            self.peak_gpu_bytes = torch_peak if self.peak_gpu_bytes is None \
                else max(self.peak_gpu_bytes, torch_peak)


# --------------------------------------------------------------------------- #
# Host description (so results are interpretable / reproducible).             #
# --------------------------------------------------------------------------- #

@dataclass
class HostInfo:
    platform: str
    python_version: str
    cpu_count: Optional[int]
    total_ram_bytes: Optional[int]
    has_torch: bool
    torch_version: Optional[str]
    cuda_available: bool
    gpu_name: Optional[str]
    has_rapl: bool

    @staticmethod
    def collect() -> "HostInfo":
        import platform
        import sys

        total_ram = None
        if _HAS_PSUTIL:
            try:
                total_ram = int(psutil.virtual_memory().total)
            except Exception:
                pass
        if total_ram is None:
            try:
                total_ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
            except Exception:
                pass

        torch_version = None
        cuda_available = False
        gpu_name = None
        if _HAS_TORCH:
            try:
                torch_version = torch.__version__
                cuda_available = bool(torch.cuda.is_available())
                if cuda_available:
                    gpu_name = torch.cuda.get_device_name(0)
            except Exception:
                pass
        if gpu_name is None:
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=name",
                     "--format=csv,noheader"],
                    stderr=subprocess.DEVNULL, timeout=5,
                )
                gpu_name = out.decode().strip().splitlines()[0] or None
            except Exception:
                pass

        return HostInfo(
            platform=platform.platform(),
            python_version=sys.version.split()[0],
            cpu_count=os.cpu_count(),
            total_ram_bytes=total_ram,
            has_torch=_HAS_TORCH,
            torch_version=torch_version,
            cuda_available=cuda_available,
            gpu_name=gpu_name,
            has_rapl=Path(_RAPL_ROOT).exists()
            and any(Path(_RAPL_ROOT).glob("intel-rapl:*")),
        )

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Result container.                                                           #
# --------------------------------------------------------------------------- #

@dataclass
class BenchmarkResult:
    """One benchmark run. Any field may be None == 'could not be measured'."""

    # identification
    engine: str = "airllm"
    model_id: Optional[str] = None
    device: Optional[str] = None
    compression: Optional[str] = None
    prompt_tokens: Optional[int] = None
    generated_tokens: Optional[int] = None

    # timing
    load_seconds: Optional[float] = None
    generate_seconds: Optional[float] = None
    seconds_per_token: Optional[float] = None
    tokens_per_second: Optional[float] = None

    # memory
    peak_rss_bytes: Optional[int] = None
    peak_gpu_bytes: Optional[int] = None

    # storage
    model_disk_bytes: Optional[int] = None
    splitted_disk_bytes: Optional[int] = None

    # energy
    cpu_energy_joules: Optional[float] = None
    gpu_energy_joules: Optional[float] = None
    joules_per_token: Optional[float] = None

    # quality
    perplexity: Optional[float] = None
    eval_text_tokens: Optional[int] = None

    # context
    host: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Pretty helpers.                                                             #
# --------------------------------------------------------------------------- #

def human_bytes(n: Optional[int]) -> str:
    if n is None:
        return "n/a"
    step = 1024.0
    units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]
    f = float(n)
    for u in units:
        if f < step or u == units[-1]:
            return f"{f:.2f} {u}"
        f /= step
    return f"{f:.2f} PiB"
