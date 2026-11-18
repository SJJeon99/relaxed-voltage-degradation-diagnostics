from __future__ import annotations

import hashlib
from typing import Any



DEFAULT_PROJECT_RANDOM_SEED = 42
MAX_NUMPY_SEED = 2**32 - 1


def normalize_seed(seed: Any = None, *, default: int | None = DEFAULT_PROJECT_RANDOM_SEED) -> int | None:
    if seed is None:
        if default is None:
            return None
        seed = default
    seed_int = int(seed)
    seed_int %= MAX_NUMPY_SEED
    if seed_int == 0:
        seed_int = 1
    return seed_int


def derive_seed(base_seed: Any = None, *parts: Any, default: int | None = DEFAULT_PROJECT_RANDOM_SEED) -> int | None:
    seed_int = normalize_seed(base_seed, default=default)
    if seed_int is None:
        return None

    hasher = hashlib.blake2b(digest_size=8)
    hasher.update(str(seed_int).encode("utf-8"))
    for part in parts:
        hasher.update(b"|")
        hasher.update(str(part).encode("utf-8"))

    derived = (seed_int + int.from_bytes(hasher.digest(), "little")) % MAX_NUMPY_SEED
    if derived == 0:
        derived = 1
    return int(derived)




