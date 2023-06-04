"""
this is the LSH (locality sensitive hashing) implementation

it is widely used and at the core of many top tech companies (Google!)

the first approach to come to mind may be brute forcing through the database and
computing the distance between all vectors to find the best; however, this easily
increases the time complexity of one search to O(n^2), possibly O(n).

using LSH, even in-memory, can allow us to achieve a sub-linear time complexity and
limit the search scope.
"""

import numpy as np

import vektor.distance

def generate_hash(dims: int, vector: np.array) -> str:
    # compute a random projection of the vector with 32 because we want a 32-bit hash
    projections = np.dot(np.random.randn(32, dims), vector.flatten())
    return "".join(["1" if i > 0 else "0" for i in projections]) # convert into binary

class MemoryStore:
    """
    in-memory store

    although it may seem as simply a dict wrapper, a dict has the goal to
    minimize collisions, but an LSH wants to maximize collisions and can
    be imagined as bucketing similar vectors to reduce the search space.
    """

    def __init__(self) -> None:
        self.store = dict()

    def append(self, key: object, value: object) -> None:
        self.store.setdefault(key, []).append(value)

    def get(self, key: object) -> object:
        return self.store.get(key, [])

class LSH:
    def __init__(self, dims: int, num_tables: int = 1) -> None:
        self.dims = dims
        self.tables = [MemoryStore() for i in range(num_tables)] # setup tables (for the examples we only need 1)

    def index(self, vector: np.array, reference: object) -> None:
        for i, table in enumerate(self.tables):
            # generate a hash for the vector and store it's reference in all tables for redundancy
            table.append(generate_hash(self.dims, vector), (tuple(vector.tolist()), reference))

    def query(self, vector: np.array, top_k: int, distance: object) -> list:
        possible = set()
        hashed = generate_hash(self.dims, vector)
        for i, table in enumerate(self.tables):
            # here is the magic, we maximized collisions so each table should point to multiple vectors
            # we store all of the them into a set to remove duplicates
            possible.update(table.get(hashed))
        ranked = [(i, distance(vector, i)) for i in possible] # compute the distance between input and possible
        ranked.sort(key = lambda x: x[1]) # sort by distance
        return ranked[:top_k] # return the top k results

"""
congrats! you officially have learned how most vector databases work!

for the interested, you can see ./vektor/distance.py for distance functions
"""
