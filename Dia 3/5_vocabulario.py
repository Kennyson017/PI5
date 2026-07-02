import json
from collections import Counter

import pandas as pd

CAMINHO_ENTRADA = "dados/saida_4_lematizado.csv"
CAMINHO_SAIDA_CSV = "dados/vocabulario.csv"
CAMINHO_SAIDA_PY = "vocabulario.py"

df = pd.read_csv(CAMINHO_ENTRADA)
df["tokens"] = df["tokens"].apply(json.loads)

frequencia_total = Counter()
frequencia_documentos = Counter()

for tokens in df["tokens"]:
    frequencia_total.update(tokens)
    frequencia_documentos.update(set(tokens))

vocabulario = sorted(frequencia_total)

# CSV com o vocabulário normalizado: uma palavra por linha, ordenada por frequência.
vocab_df = pd.DataFrame({
    "palavra": vocabulario,
    "frequencia_total": [frequencia_total[p] for p in vocabulario],
    "frequencia_documentos": [frequencia_documentos[p] for p in vocabulario],
}).sort_values(by="frequencia_total", ascending=False, ignore_index=True)

vocab_df.to_csv(CAMINHO_SAIDA_CSV, index=False)

# Array Python com o vocabulário, para ser importado em outras etapas (ex.: bag of words).
with open(CAMINHO_SAIDA_PY, "w", encoding="utf-8") as arquivo:
    arquivo.write("VOCABULARIO = [\n")
    for palavra in vocabulario:
        arquivo.write(f"    {palavra!r},\n")
    arquivo.write("]\n")

print(f"{len(vocabulario)} palavras únicas no vocabulário.")
print(f"Vocabulário salvo em '{CAMINHO_SAIDA_CSV}' e '{CAMINHO_SAIDA_PY}'.")
print("Top 10 palavras mais frequentes:")
print(vocab_df.head(10))
