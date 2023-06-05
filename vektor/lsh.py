import numpy as np

import vektor.distance

def shingle(sentence: str, k: int = 5) -> set:
    return set([sentence[i:i + k] for i in range(len(sentence) - k)])


