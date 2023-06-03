import numpy as np

class LSH:
    def __init__(self, dims: int) -> None:
        self.dims = dims
        self.tables = 1

    def index(self, vector: np.array, reference: object) -> None:
        pass

    def query(self, vector: np.array, top_k: int = 5, distance: object = None) -> list:
        pass
