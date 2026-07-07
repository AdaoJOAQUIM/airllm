"""
E15 - two hands-on tests of the conservation law and of pointer overhead.

A) TRASH TEST (filesystem): moving a file to a trash dir is a pointer
   rename - nothing shrinks; restore works while bytes exist; after
   real deletion, they don't.
B) OBJECT/POINTER OVERHEAD: the same 1M records stored as class
   instances / __slots__ / tuples / contiguous numpy arrays -
   memory and access-speed measured. "Replacing pointers, objects,
   classes" = structure-of-arrays, which is what the engine already
   does with tensors.
"""
import hashlib, os, shutil, time, tracemalloc
import numpy as np

BASE = os.path.join(os.environ.get("SCRATCH", "/tmp"), "e15")
shutil.rmtree(BASE, ignore_errors=True); os.makedirs(BASE)

def free_bytes():
    st = os.statvfs(BASE); return st.f_bavail * st.f_frsize

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''): h.update(c)
    return h.hexdigest()[:16]

print("=== A) trash test (100 MB file) ===")
f = os.path.join(BASE, "big.bin")
with open(f, 'wb') as fh:
    for _ in range(100): fh.write(os.urandom(1 << 20))
h0, free0 = sha(f), free_bytes()
trash = os.path.join(BASE, ".trash"); os.makedirs(trash)
os.rename(f, os.path.join(trash, "big.bin"))          # "move to trash"
free1 = free_bytes()
print(f"free space after move-to-trash: {(free1-free0)/1e6:+.1f} MB  (pointer move frees NOTHING)")
os.rename(os.path.join(trash, "big.bin"), f)          # restore
print(f"restored intact: {sha(f) == h0}  (restorable because bytes never left)")
os.remove(f)                                           # empty trash
print(f"free space after real delete:  {(free_bytes()-free0)/1e6:+.1f} MB  (space back only when bytes go)")
print(f"restore after delete: {'impossible' if not os.path.exists(f) else '?'} - "
      f"restorable <=> bytes exist. Conservation, verified.\n")

print("=== B) 1,000,000 records: objects vs pointers vs arrays ===")
N = 1_000_000
def bench(name, build, access):
    tracemalloc.start()
    t0 = time.perf_counter(); data = build(); t_build = time.perf_counter() - t0
    mem = tracemalloc.get_traced_memory()[1]; tracemalloc.stop()
    t0 = time.perf_counter(); s = access(data); t_acc = time.perf_counter() - t0
    print(f"{name:22s} {mem/N:8.1f} B/rec {t_build:7.2f}s build {t_acc*1000:9.1f} ms access  (sum={s})")

class Rec:
    def __init__(s, a, b): s.a = a; s.b = b
class RecS:
    __slots__ = ('a', 'b')
    def __init__(s, a, b): s.a = a; s.b = b

bench("class instances",  lambda: [Rec(i, i*2) for i in range(N)],
                          lambda d: sum(r.a for r in d) % 97)
bench("__slots__ objects", lambda: [RecS(i, i*2) for i in range(N)],
                          lambda d: sum(r.a for r in d) % 97)
bench("tuples",           lambda: [(i, i*2) for i in range(N)],
                          lambda d: sum(t[0] for t in d) % 97)
bench("numpy SoA arrays", lambda: (np.arange(N, dtype=np.int64), np.arange(N, dtype=np.int64)*2),
                          lambda d: int(d[0].sum()) % 97)
shutil.rmtree(BASE, ignore_errors=True)
print("\npointers/objects pay metadata per record; contiguous arrays pay ~0.")
print("Same information, lighter container - the engine's tensors/safetensors")
print("and factstore packing are exactly this replacement.")
