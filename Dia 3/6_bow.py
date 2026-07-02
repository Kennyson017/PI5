import json

import pandas as pd
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import CountVectorizer

CAMINHO_ENTRADA = "dados/saida_4_lematizado.csv"
CAMINHO_SAIDA_MATRIZ = "dados/bow_matriz.npz"
CAMINHO_SAIDA_VOCABULARIO = "dados/bow_vocabulario.csv"
TOKEN_PATTERN = r"(?u)\b[a-zA-ZÀ-ÿ][a-zA-ZÀ-ÿ]{2,}\b"

df = pd.read_csv(CAMINHO_ENTRADA)
df["tokens"] = df["tokens"].apply(json.loads)

# Cada linha de "tokens" já contém uma lista de palavras normalizadas (lematizadas).
# Algoritmos como CountVectorizer, TF-IDF e K-Means geralmente trabalham com uma coleção
# de documentos no formato de strings (texto completo), e não com listas de tokens.
# Por isso, cada lista de lemas é "reconvertida" para texto, juntando as palavras com espaço entre elas.
documentos = [" ".join(tokens) for tokens in df["tokens"]]

print("Exemplo de documento reconvertido (linha 0):")
print(documentos[0][:300])

vectorize = CountVectorizer(token_pattern=TOKEN_PATTERN)
X = vectorize.fit_transform(documentos)

vocabulario = vectorize.get_feature_names_out()

save_npz(CAMINHO_SAIDA_MATRIZ, X)
pd.DataFrame({"palavra": vocabulario}).to_csv(CAMINHO_SAIDA_VOCABULARIO, index=False)

print(f"\nMatriz BoW: {X.shape[0]} documentos x {X.shape[1]} palavras.")
print(f"Matriz salva em '{CAMINHO_SAIDA_MATRIZ}' e vocabulário em '{CAMINHO_SAIDA_VOCABULARIO}'.")
print("Amostra do vocabulário:")
print(vocabulario[:20])
