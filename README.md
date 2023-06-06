# vektor
a mini vector database implementation that intends to be educational and interpretable

many current open source vector database implementations tend to be sprawling and over-complicated 
and the general AI enthusiast cannot parse through the tens of files and hundreds of lines to learn.

vektor is the first serious mini vector database implementation that is understandable -- the
combination of a low count of lines and expressive variable names makes it feasible for anyone 
to learn to love the same technology VCs do!

as an overview, the implemenation currently uses BERT as the embedding model and LSH to
achive sub-linear time complexity on searches, with Jaccard similarity to find neighboring vectors.

`vektor/vektor.py` is very beginner-friendly and is simply a wrapper for the LSH class located in
`vektor/lsh.py`. the LSH implementation is not as complex as it seems, but a some Google
searches ( which also use a vector database :) ) should clear most doubts.

there are some examples in `examples/`. `examples/periodic_table.py` is a end-to-end example of 
how one would use the classes and `examples/lsh.py` is just focused on the LSH (and can also be used
as a unittest for the class)

have fun!

references:
1. https://www.pinecone.io/learn/vector-database/
2. https://www.pinecone.io/learn/locality-sensitive-hashing/
