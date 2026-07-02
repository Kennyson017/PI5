import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

CAMINHO_ENTRADA = "dados/saida_4_lematizado.csv"
CAMINHO_SAIDA_GRAFICO = "dados/kmeans_silhouette.png"

K_MINIMO = 2
K_MAXIMO = 15
K_ALTERNATIVO = 4
TOKEN_PATTERN = r"(?u)\b[a-zA-ZÀ-ÿ][a-zA-ZÀ-ÿ]{2,}\b"
KMEANS_CONFIG = {
    "n_init": 20,
    "max_iter": 300,
    "algorithm": "lloyd",
    "random_state": 42,
}

df = pd.read_csv(CAMINHO_ENTRADA)
df["tokens"] = df["tokens"].apply(json.loads)

documentos = [" ".join(tokens) for tokens in df["tokens"]]

# Mesma vetorização usada no cotovelo (7_kmeans_cotovelo.py), para poder comparar os resultados.
vectorizer = TfidfVectorizer(min_df=5, max_df=0.9, token_pattern=TOKEN_PATTERN)
X = vectorizer.fit_transform(documentos)

scores = []
for k in range(K_MINIMO, K_MAXIMO + 1):
    kmeans = KMeans(n_clusters=k, **KMEANS_CONFIG)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    scores.append(score)
    print(f"k={k}: silhouette={score:.4f}")

melhor_k = range(K_MINIMO, K_MAXIMO + 1)[scores.index(max(scores))]
print(f"\nMelhor k pelo silhouette score: {melhor_k} (score={max(scores):.4f})")

fig, ax = plt.subplots(figsize=(8, 5), facecolor="#f6f8fa")
ax.set_facecolor("#ffffff")
ks = list(range(K_MINIMO, K_MAXIMO + 1))
ax.plot(ks, scores, marker="o", color="#0969da")
indice_melhor = ks.index(melhor_k)
ax.scatter(
    [melhor_k],
    [scores[indice_melhor]],
    color="red",
    s=90,
    zorder=5,
    label=f"melhor k={melhor_k}",
)
indice_alternativo = ks.index(K_ALTERNATIVO)
ax.scatter(
    [K_ALTERNATIVO],
    [scores[indice_alternativo]],
    color="green",
    s=90,
    zorder=5,
    label=f"k={K_ALTERNATIVO} interpretável",
)
ax.legend(facecolor="#ffffff", labelcolor="#1f2328")
ax.set_xlabel("Número de clusters (k)", color="#57606a")
ax.set_ylabel("Silhouette score", color="#57606a")
ax.set_title("Silhouette score por número de clusters - K-means", color="#1f2328")
ax.set_xticks(range(K_MINIMO, K_MAXIMO + 1))
ax.tick_params(colors="#57606a")
ax.grid(True, color="#d0d7de", alpha=0.8)
for spine in ax.spines.values():
    spine.set_color("#d0d7de")
plt.savefig(CAMINHO_SAIDA_GRAFICO, dpi=150, bbox_inches="tight")

print(f"Gráfico salvo em '{CAMINHO_SAIDA_GRAFICO}'.")
