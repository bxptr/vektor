# vektor
a mini vector database implementation that intends to be educational and interpretable

many current open source implementations tend to be sprawling and over-complicated and the
general AI enthusiast cannot parse through the tens of files and hundreds of lines to learn.

vektor is the first mini vector database implementation and is also the first to demystify 
how the code works with comments throughout to explain what each section does.

as an overview, the implemenation currently uses BERT as the embedding model, LSH to
achive sub-linear time complexity on searches, and cosine distance to compare vectors.

to start, read `vektor/vektor.py` and follow the comments to `vektor/lsh.py` and optionally
view `vektor/distance.py`. this will guide you through decisions and implementation details.

have fun!

references:
1. https://www.pinecone.io/learn/vector-database/
2. https://www.pinecone.io/learn/locality-sensitive-hashing/
3. https://www.pinecone.io/learn/vector-similarity/

note: notes i took when learning how these work are in `notes`
