import numpy as np
import transformers
import torch # yes, this is just for the type-checking

tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")
model = transformers.BertModel.from_pretrained("bert-base-uncased")

def embedding(sentence: str) -> torch.Tensor:
    inputs = tokenizer(sentence, return_tensors = "pt")
    outputs = model(**inputs)
    return outputs.last_hidden_state

def euclidean(vectors: np.array, query: np.array) -> float:
    similarities = np.linalg.norm(vectors - query, axis = 1)
    return 1 / (1 + similarities)

class Vektor:
    def __init__(
        self,
        vectors: np.array = np.array([]),
        similarity: object = euclidean
    ) -> None:
        self.vectors = vectors
        self.embedding = None
        self.similarity = similarity
        self.items = []

    def from_source(self, source: list) -> None:
        vectors = np.array(self.embedding(source)).astype(np.float32)
        for v, item in zip(vectors, source):
            self.vectors = np.vstack([self.vectors, v]).astype(np.float32)
            self.items.append(item)

    def query(self, vector: np.array, top_k: int = 5) -> list:
        vector = self.embedding([vector])[0]
        similarities = self.similarity(self.vectors, vector)
        top = np.argsort(similarities, axis = 0)[-top_k:][::-1]
        return list(zip([self.items[i] for i in top.flatten()], similarities[top].flatten()))

