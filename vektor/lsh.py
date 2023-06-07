import numpy as np
import hashlib
import struct
import random
import collections
import json

mersenne = (1 << 61) - 1
max_hash = (1 << 32) - 1
hash_range = (1 << 32)

def integrate(f: object, a: float, b: float) -> float:
    """
    integrate (area under the curve) of f(x)
    """

    area = 0.0
    x = a
    while x < b:
        area += f(x + 0.5 * 1e-3) * 1e-3
        x += 1e-3
    return area

class LSH:
    """
    LSH (locality sensitive hashing) implementation
    """

    def __init__(
        self,
        threshold: float = 0.1,
        n_perms: int = 128,
        weights = (0.5, 0.5)
    ) -> None:
        self.n_perms = n_perms
        pos_weight, neg_weight = weights
        b, r = self._optimal(threshold, pos_weight, neg_weight)
        self.tables = [collections.defaultdict(list) for _ in range(b)]
        self.ranges = [(i * r, (i + 1) * r) for i in range(b)]
        self.swap = lambda x: bytes(x.byteswap().data)
        self.store = dict()

    def _optimal(self, threshold: float, pos_weight: float, neg_weight: float) -> tuple:
        min_error = float("inf")
        params = (0, 0)
        for b in range(1, self.n_perms + 1):
            max_r = int(self.n_perms / b)
            for r in range(1, max_r + 1):
                pos_prob_fn = lambda x: 1 - (1 - x ** float(r)) ** float(b)
                neg_prob_fn = lambda x: 1 - (1 - (1 - x ** float(r)) ** float(b))
                pos_prob = integrate(pos_prob_fn, 0.0, threshold)
                neg_prob = integrate(neg_prob_fn, threshold, 1.0)
                error = pos_prob * pos_weight + neg_prob * neg_weight
                if error < min_error:
                    min_error = error
                    params = (b, r)
        return params

    def _generate_hash(self, vector: np.array) -> None:
        generator = random.Random(1337)
        values = np.ones(self.n_perms, np.uint64) * max_hash
        a, b = np.array([
            (generator.randint(1, mersenne), generator.randint(0, mersenne))
            for _ in range(self.n_perms)
        ], np.uint64).T
        unpacked = struct.unpack("<I", hashlib.sha1(vector.tobytes()).digest()[:4])[0]
        return np.minimum(np.bitwise_and((a * values + b) % mersenne, np.uint64(max_hash)), values)

    def index(self, vector: np.array, reference: object) -> None:
        reference = json.dumps(reference)
        hashed = self._generate_hash(vector)
        self.store[reference] = [self.swap(hashed[s:e]) for s, e in self.ranges]
        for hash_, table in zip(self.store[reference], self.tables):
            table[hash_].append(reference)

    def query(self, vector: np.array) -> None:
        candidates = set()
        hashed = self._generate_hash(vector)
        for (s, e), table in zip(self.ranges, self.tables):
            hash_ = self.swap(hashed[s:e])
            if hash_ in table:
                for key in table[hash_]:
                    candidates.add(key)
        return list(candidates)
