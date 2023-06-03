import numpy as np

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
        projections = np.dot(np.random.randn(32, self.dims), vector)
        hashed = "".join(["1" if i > 0 else "0" for i in projections])
        for i, table in enumerate(self.tables):
            table.append(hashed, (tuple(vector.tolist()), reference))

    def query(self, vector: np.array, top_k: int = 5, distance: object = None) -> list:
        pass
