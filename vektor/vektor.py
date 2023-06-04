"""
vektor
a mini vector database implementation that intends to be educational and interpretable

welcome to vektor's source. if you're here you want to learn how vector databases work and
what makes VCs invest millions of dollars into them.

interestingly enough, it is not that hard to make a functional and performant implementation.
"""

import numpy as np
import transformers
import pickle
import tqdm

import vektor.lsh
import vektor.distance

import torch # yes, this is just for type-checking

transformers.logging.set_verbosity_error() # i just don't like the message
tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")
model = transformers.BertModel.from_pretrained("bert-base-uncased")

"""
the above code simply imports different libraries and initializes BERT.

the core of vector databases lies in the embedding model of choice, and since that is a different
concept, I opted to use an existing implementation. The choices were either FSE, OpenAI's model, or BERT.

I initially opted for FSE but it didn't compile on my macbook so I went with BERT because OpenAI isn't too 
open anymore...
"""

def bert_embedding(sentence: str) -> torch.Tensor:
    """
    convert a string sentence into a BERT embedding
    """

    inputs = tokenizer(
        sentence,
        return_tensors = "pt",
        padding = "max_length",
        max_length = 100,
        truncation = True
    )
    outputs = model(**inputs)
    return outputs.last_hidden_state.detach()

class Vektor:
    """
    database class that encompasses the entire functionality
    """

    def __init__(
        self,
        vectors: np.array = np.empty((1, 100, 768), dtype = np.float32), # (1, 100, 768) is the shape of BERT embeddings
        embedding: object = bert_embedding,
        distance: object = vektor.distance.cosine # distance algorithm to find the difference of two vectors
    ) -> None:
        self.embedding = embedding
        self.distance = distance
        self.lsh = vektor.lsh.LSH(1 * 100 * 768) # learn more at ./vektor/lsh.py

    def from_source(self, source: list, key_fn: object = lambda x: x) -> None:
        for ref in (bar := tqdm.tqdm(source)):
            # convert a source list of sentences into np.array's with BERT and choosing what to 
            # parse with a key function
            vector = np.array(self.embedding(key_fn(ref))).astype(np.float32)
            self.lsh.index(vector, ref) # index the new vector with LSH to search from later
            bar.set_description("from_source")

    def save(self, filename: str) -> None:
        with open(filename, "wb") as handler:
            pickle.dump(self.lsh, handler) # pickle dump the LSH to increase repeat runtime

    def load(self, filename: str) -> None:
        with open(filename, "rb") as handler:
            self.lsh = pickle.load(handler)

    def query(self, sentence: str, top_k: int = 5) -> list:
        # np.array the BERT embedding of the query sentence to have all vectors as the same type
        vector = np.array(self.embedding(sentence)).astype(np.float32)
        return self.lsh.query(vector, top_k, self.distance) # return the LSH query

"""
after reading this file, go to ./vektor/lsh.py to learn about the indexing and 
reference system of the database!
"""
