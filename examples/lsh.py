import vektor

a = vektor.bert_embedding("aarush is very cool")
b = vektor.bert_embedding("aarush is smooth")
c = vektor.bert_embedding("aarush is very cool")

lsh = vektor.lsh.LSH()
lsh.index(a, "aarush is very cool")
lsh.index(b, "aarush is smooth")
result = lsh.query(c)

print(result)
