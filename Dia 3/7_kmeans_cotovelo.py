import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

CAMINHO_ENTRADA = "dados/saida_4_lematizado.csv"
CAMINHO_SAIDA_GRAFICO = "dados/kmeans_cotovelo.png"

K_MINIMO = 2
K_MAXIMO = 15
K_ESCOLHIDO = 4
TOKEN_PATTERN = r"(?u)\b[a-zA-ZÀ-ÿ][a-zA-ZÀ-ÿ]{2,}\b"
KMEANS_CONFIG = {
    "n_init": 20,
    "max_iter": 300,
    "algorithm": "lloyd",
    "random_state": 42,
}

df = pd.read_csv(CAMINHO_ENTRADA)
df["tokens"] = df["tokens"].apply(json.loads)

# TF-IDF funciona melhor que contagem simples (BoW) para o K-means, pois
# reduz o peso de palavras muito comuns e destaca as mais características de cada texto.
documentos = [" ".join(tokens) for tokens in df["tokens"]]

vectorizer = TfidfVectorizer(min_df=5, max_df=0.9, token_pattern=TOKEN_PATTERN)
X = vectorizer.fit_transform(documentos)

print(f"Matriz TF-IDF: {X.shape[0]} documentos x {X.shape[1]} palavras.")

inercias = []
for k in range(K_MINIMO, K_MAXIMO + 1):
    kmeans = KMeans(n_clusters=k, **KMEANS_CONFIG)
    kmeans.fit(X)
    inercias.append(kmeans.inertia_)
    print(f"k={k}: inércia={kmeans.inertia_:.2f}")

fig, ax = plt.subplots(figsize=(8, 5), facecolor="#f6f8fa")
ax.set_facecolor("#ffffff")
ks = list(range(K_MINIMO, K_MAXIMO + 1))
ax.plot(ks, inercias, marker="o", color="#0969da")
indice_escolhido = ks.index(K_ESCOLHIDO)
ax.scatter(
    [K_ESCOLHIDO],
    [inercias[indice_escolhido]],
    color="red",
    s=90,
    zorder=5,
    label=f"k={K_ESCOLHIDO} escolhido",
)
ax.legend(facecolor="#ffffff", labelcolor="#1f2328")
ax.set_xlabel("Número de clusters (k)", color="#57606a")
ax.set_ylabel("Inércia", color="#57606a")
ax.set_title("Método do cotovelo - K-means", color="#1f2328")
ax.set_xticks(range(K_MINIMO, K_MAXIMO + 1))
ax.tick_params(colors="#57606a")
ax.grid(True, color="#d0d7de", alpha=0.8)
for spine in ax.spines.values():
    spine.set_color("#d0d7de")
plt.savefig(CAMINHO_SAIDA_GRAFICO, dpi=150, bbox_inches="tight")

print(f"\nGráfico do cotovelo salvo em '{CAMINHO_SAIDA_GRAFICO}'.")
print("Escolha o k onde a curva 'dobra' (deixa de cair rapidamente).")
