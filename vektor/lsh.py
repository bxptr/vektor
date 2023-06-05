import numpy as np
import hashlib
import struct
import random
import collections
import json

generator = random.Random()
generator.seed(1337)

mersenne = (1 << 61) - 1
max_hash = (1 << 32) - 1
hash_range = (1 << 32)

def generate_hash(n_perms: int, vector: np.array) -> np.array:
    permutations = np.array([
        (generator.randint(1, mersenne), generator.randint(0, mersenne))
        for _ in range(n_perms)
    ], np.uint64).T
    sha = haslib.sha1(vector)
    values = struct.unpack("<I", self.hashobj(sha).digest()[:4])[0]
    a, b = permutations
    perm_values = np.bitwise_and((a * values + b) % mersenne, np.uint64(max_hash))
    return np.minimum(perm_values, np.ones(n_perms, np.uint64) * max_hash)

def integrate(fn: object, a: float, b: float) -> float:
    area = 0.0
    x = a
    while x < b:
        area += fn(x + 0.5 * 1e-3) * 1e-3
        x += 1e-3
    return area

class LSH:
    """
    LSH (locality sensitive hashing) implementation
    """

    def __init__(
        self,
        dims: int,
        threshold: float = 0,
        n_perms: int = 128,
        weights = (0.5, 0.5)
    ) -> None:
        self.dims = dims
        self.threshold = threshold
        self.n_perms = n_perms
        self.weights = weights
        self.store = dict()
        self.b, self.r = self._optimal()
        self.tables = [collections.defaultdict(list) for _ in range(self.b)]
        self.ranges = [(i * self.r, (i + 1) * self.r) for i in range(self.b)]

    def _optimal(self) -> tuple:
        min_error = float("inf")
        params = (0, 0)
        for b in range(1, self.n_perms + 1):
            max_r = int(self.n_perms / b)
            for r in range(1, max_r + 1):
                pos_prob_fn = lambda x: 1 - (1 - x ** float(r)) ** float(b)
                neg_prob_fn = lambda x: 1 - (1 - (1 - x ** float(r)) ** float(b))
                pos_prob = integrate(pos_prob_fn, 0.0, self.threshold)
                neg_prob = integrate(neg_prob_fn, self.threshold, 1.0)
                error = pos_prob * self.weights[0] + neg_prob * self.weights[1]
                if error < min_error:
                    min_error = error
                    params = (b, r)
        return params

    def index(self, vector: np.array, reference: object) -> None:
        reference = json.dumps(reference)
        self.store[reference] = [generate_hash(self.dims, vector)]
        for hash_, table in zip(self.store[reference], self.tables):
            table[hash_].append(reference)

    def query(self, vector: np.array) -> None:
        candidates = set()
        for (s, e), table in zip(self.ranges, self.tables):
            hash_ = generate_hash(self.dims, vector)
            print(hash_)
            print(table.keys())
            if hash_ in table:
                for key in table[hash_]: candidates.add(key)
        return [json.dumps(c) for c in list(candidates)]
