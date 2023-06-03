import vektor
import json
import os.path

import time

source = []
with open("./examples/references/periodic_table.json") as handler:
    table = json.load(handler)
    for element in table["elements"]:
        source.append(element)

print("importing periodic table")
start = time.time()

database = vektor.Vektor()
if os.path.isfile("db.bin"):
    database.load("db.bin")
else:
    database.from_source(source, lambda x: x["summary"])
    database.save("db.bin")

end = time.time()
print(f"time: {(end - start) * 10 ** 3}ms")

start = time.time()
results = database.query("Lightest element")
end = time.time()
print(f"time: {(end - start) * 10 ** 3}ms")

print(results)
