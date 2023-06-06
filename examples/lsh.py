import vektor

a = vektor.bert_embedding("aarush is very cool")
b = vektor.bert_embedding("aarush is smooth")
c = vektor.bert_embedding("aarush is very cool")

lsh = vektor.lsh.LSH()
lsh.index(a, "m2")
lsh.index(b, "m3")
result = lsh.query(c)

print(result)
