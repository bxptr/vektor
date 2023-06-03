import numpy as np
import transformers
import pickle

import torch # yes, this is just for type-checking

tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")
model = transformers.BertModel.from_pretrained("bert-base-uncased")

def bert_embedding(sentence: str) -> torch.Tensor:
    inputs = tokenizer(
        sentence,
        return_tensors = "pt",
        padding = "max_length",
        max_length = 100,
        truncation = True
    )
    outputs = model(**inputs)
    return outputs.last_hidden_state.detach()

def euclidean(vectors: np.array, query: np.array) -> list:
    #distances = np.empty((1, 100, 768), dtype = np.float32)
    #for i, v in enumerate(vectors):
        #dist = np.sqrt(np.sum((a - b) ** 2 for a, b in zip(v, query)))
        #distances = np.vstack((distances, dist))
    return [np.linalg.norm(query - vec) for vec in vectors]

class Vektor:
    def __init__(
        self,
        vectors: np.array = np.empty((1, 100, 768), dtype = np.float32),
        embedding: object = bert_embedding,
        similarity: object = euclidean
    ) -> None:
        self.vectors = vectors
        self.embedding = embedding
        self.similarity = similarity
        self.references = []

    def from_source(self, source: list) -> None:
        for ref in source:
            print("from_source:", self.vectors.shape)
            vector = np.array(self.embedding(ref["info"]["description"])).astype(np.float32)
            self.vectors = np.vstack((self.vectors, vector))
            self.references.append(ref)

    def save(self, filename: str) -> None:
        with open(filename, "wb") as handler:
            pickle.dump({
                "vectors": self.vectors,
                "references": self.references
            }, handler)

    def load(self, filename: str) -> None:
        with open(filename, "rb") as handler:
            data = pickle.load(handler)
        self.vectors = data["vectors"].astype(np.float32)
        self.references = data["references"]

    def query(self, query: str, top_k: int = 5) -> list:
        vector = np.array(self.embedding(query))
        similarities = self.similarity(self.vectors, vector)
        table = {j: i for i, j in enumerate(similarities)}
        similarities.sort(reverse = True)
        top = [self.references[table[i] - 1] for i in similarities][:top_k]
        return top
