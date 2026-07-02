import json
import nltk
import pandas as pd
from nltk.corpus import stopwords

# Baixa a lista de stop words
nltk.download('stopwords')

CAMINHO_ENTRADA = "dados/saida_2_tokenizado.csv"
CAMINHO_SAIDA = "dados/saida_3_sem_stopwords.csv"

df = pd.read_csv(CAMINHO_ENTRADA)
df["tokens"] = df["tokens"].apply(json.loads)

stop_words = set(stopwords.words('portuguese'))


def remover_stopwords(tokens: list[str]) -> list[str]:
    return [token for token in tokens if token.lower() not in stop_words]


df["tokens"] = df["tokens"].apply(remover_stopwords)
df[["tokens"]].assign(tokens=df["tokens"].apply(json.dumps)).to_csv(CAMINHO_SAIDA, index=False)

print(f"{len(df)} textos filtrados. Saída salva em '{CAMINHO_SAIDA}'.")
print("Exemplo (linha 0):")
print(df["tokens"].iloc[0][:20])
