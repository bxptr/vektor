import numpy as np

import vektor.distance

def generate_hash(dims: int, vector: np.array) -> str:
    """
    compute 32bit binary hash for given vector
    """

    projections = np.dot(np.random.randn(32, dims), vector.flatten())
    return "".join(["1" if i > 0 else "0" for i in projections])

class MemoryStore:
    """
    in-memory store
    """

    def __init__(self) -> None:
        self.store = dict()

    def append(self, key: object, value: object) -> None:
        self.store.setdefault(key, []).append(value)

class LSH:
    """
    LSH (locality sensitive hashing) implementation
    """

    def __init__(self, dims: int, num_tables: int = 1) -> None:
        self.dims = dims
        self.tables = [MemoryStore() for i in range(num_tables)]

    def index(self, vector: np.array, reference: object) -> None:
        for i, table in enumerate(self.tables):
            table.append(generate_hash(self.dims, vector), (tuple(vector.tolist()), reference))

    def query(self, vector: np.array, distance: object) -> list:
        possible = set()
        hashed = generate_hash(self.dims, vector)
        for i, table in enumerate(self.tables):
            possible.update(table.get(hashed))
        ranked = [(i, distance(vector, i)) for i in possible]
        ranked.sort(key = lambda x: x[1])
        return ranked[:top_k]
