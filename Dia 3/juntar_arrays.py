import json

import pandas as pd

df = pd.read_csv("dados/saida_4_lematizado.csv")

vocabulario = set()
for tokens in df["tokens"]:
    vocabulario.update(json.loads(tokens))

vocabulario = sorted(vocabulario)

print(f"{len(set(vocabulario))} palavras únicas juntando os 1500 arrays.")
print(vocabulario[:20])
