import numpy as np
import collections

def generate_hash(dims: int, vector: np.array) -> str:
    projections = np.dot(np.random.randn(64, dims), vector.flatten())
    return "".join(["1" if i > 0 else "0" for i in projections])

def integrate(fn: object, a: float, b: float) -> float:
    area = 0.0
    x = a
    while x < b:
        area += f(x + 0.5 * 1e-3) * 1e-3
        x += 1e-3
    return area

class LSH:
    """
    LSH (locality sensitive hashing) implementation)
    """

    def __init__(self, threshold: float = 0.9, perms: int = 128, weights = (0.5, 0.5)) -> None:
        self.threshold = threshold
        self.perms = perms
        self.weights = weights
        self.store = dict()
        self.b, self.r = self._optimal()
        self.tables = [defaultdict(list) for _ in range(self.b)]
        self.ranges = [(i * self.r, (i + 1) * self.r) for i in range(self.b)]

    def _optimal(self) -> tuple:
        min_error = float("inf")
        params = (0, 0)
        for b in range(1, self.perms + 1):
            max_r = int(self.perms / b)
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

    def index(self, vector: np.array, reference: str) -> None:
        self.keys[reference] = [generate_hash(vector)[i:j] for i, j in self.ranges]
        for hash_, table in zip(self.keys[reference], self.tables):
            table[hash_].append(reference)

    def query(self, vector: np.array) -> None:
        candidates = set()
        for (s, e), table in zip(self.ranges, self.tables):
            hash_ = generate_hash(vector[s:e])
            if hash_ in table:
                for key in table[hash_]: candidates.add(key)
        return list(candidates)
