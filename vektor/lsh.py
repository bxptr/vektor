import numpy as np

import vektor.distance

def generate_hash(vector: np.array) -> str:
    projections = np.dot(np.random.randn(32, self.dims), vector)
    return "".join(["1" if i > 0 else "0" for i in projections])

class MemoryStore:
    def __init__(self) -> None:
        self.store = dict()

    def append(self, key: object, value: object) -> None:
        self.store.setdefault(key, []).append(value)

class LSH:
    def __init__(self, dims: int, num_tables: int = 1) -> None:
        self.dims = dims
        self.tables = [MemoryStore() for i in range(num_tables)]

    def index(self, vector: np.array, reference: object) -> None:
        for i, table in enumerate(self.tables):
            table.append(generate_hash(vector), (tuple(vector.tolist()), reference))

    def query(self, vector: np.array, top_k: int = 5, distance: object = vektor.distance) -> list:
        possible = set()
        for i, table in enumerate(self.tables):
            possible.update(table.get(generate_hash(vector)))
        ranked = [(i, distance(vector, i)) for i in possible]
        ranked.sort(key = lambda x: x[1])
        return ranked[:top_k]
