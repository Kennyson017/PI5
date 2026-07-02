import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score

CAMINHO_ENTRADA = "dados/saida_4_lematizado.csv"
CAMINHO_SAIDA_CSV = "dados/kmeans_metricas_extras.csv"
CAMINHO_SAIDA_GRAFICO = "dados/kmeans_metricas_extras.png"

K_MINIMO = 2
K_MAXIMO = 15
K_ALTERNATIVO = 4
N_COMPONENTES_SVD = 100
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

vectorizer = TfidfVectorizer(min_df=5, max_df=0.9, token_pattern=TOKEN_PATTERN)
X_tfidf = vectorizer.fit_transform(documentos)

# Calinski-Harabasz e Davies-Bouldin trabalham melhor com matriz densa.
# SVD reduz a dimensionalidade mantendo a estrutura principal dos textos.
n_componentes = min(N_COMPONENTES_SVD, X_tfidf.shape[1] - 1)
svd = TruncatedSVD(n_components=n_componentes, random_state=42)
X = svd.fit_transform(X_tfidf)

print(
    f"Matriz TF-IDF reduzida por SVD: {X.shape[0]} documentos x {X.shape[1]} componentes "
    f"({svd.explained_variance_ratio_.sum():.2%} da variância)."
)

resultados = []
for k in range(K_MINIMO, K_MAXIMO + 1):
    kmeans = KMeans(n_clusters=k, **KMEANS_CONFIG)
    labels = kmeans.fit_predict(X)

    calinski = calinski_harabasz_score(X, labels)
    davies = davies_bouldin_score(X, labels)

    resultados.append({
        "k": k,
        "calinski_harabasz": calinski,
        "davies_bouldin": davies,
    })
    print(f"k={k}: Calinski-Harabasz={calinski:.2f} | Davies-Bouldin={davies:.4f}")

metricas = pd.DataFrame(resultados)
metricas.to_csv(CAMINHO_SAIDA_CSV, index=False)

melhor_calinski = metricas.loc[metricas["calinski_harabasz"].idxmax()]
melhor_davies = metricas.loc[metricas["davies_bouldin"].idxmin()]

print(
    f"\nMelhor k por Calinski-Harabasz: {int(melhor_calinski['k'])} "
    f"(score={melhor_calinski['calinski_harabasz']:.2f})"
)
print(
    f"Melhor k por Davies-Bouldin: {int(melhor_davies['k'])} "
    f"(score={melhor_davies['davies_bouldin']:.4f})"
)

fig, ax1 = plt.subplots(figsize=(9, 5), facecolor="#f6f8fa")
ax1.set_facecolor("#ffffff")

ax1.plot(metricas["k"], metricas["calinski_harabasz"], marker="o", color="#2764d8")
ax1.scatter(
    [melhor_calinski["k"]],
    [melhor_calinski["calinski_harabasz"]],
    color="red",
    s=90,
    zorder=5,
    label=f"Calinski melhor k={int(melhor_calinski['k'])}",
)
calinski_alternativo = metricas.loc[metricas["k"] == K_ALTERNATIVO].iloc[0]
ax1.scatter(
    [calinski_alternativo["k"]],
    [calinski_alternativo["calinski_harabasz"]],
    color="green",
    s=90,
    zorder=5,
    label=f"Calinski k={K_ALTERNATIVO} interpretável",
)
ax1.set_xlabel("Número de clusters (k)")
ax1.set_ylabel("Calinski-Harabasz (maior é melhor)", color="#2764d8")
ax1.tick_params(axis="y", labelcolor="#2764d8")
ax1.tick_params(axis="x", colors="#57606a")
ax1.set_xticks(metricas["k"])
ax1.grid(True, color="#d0d7de", alpha=0.8)
for spine in ax1.spines.values():
    spine.set_color("#d0d7de")

ax2 = ax1.twinx()
ax2.plot(metricas["k"], metricas["davies_bouldin"], marker="s", color="#c2413b")
ax2.scatter(
    [melhor_davies["k"]],
    [melhor_davies["davies_bouldin"]],
    color="red",
    s=90,
    zorder=5,
    label=f"Davies melhor k={int(melhor_davies['k'])}",
)
davies_alternativo = metricas.loc[metricas["k"] == K_ALTERNATIVO].iloc[0]
ax2.scatter(
    [davies_alternativo["k"]],
    [davies_alternativo["davies_bouldin"]],
    color="green",
    s=90,
    zorder=5,
    label=f"Davies k={K_ALTERNATIVO} interpretável",
)
ax2.set_ylabel("Davies-Bouldin (menor é melhor)", color="#c2413b")
ax2.tick_params(axis="y", labelcolor="#c2413b")
for spine in ax2.spines.values():
    spine.set_color("#d0d7de")

linhas1, rotulos1 = ax1.get_legend_handles_labels()
linhas2, rotulos2 = ax2.get_legend_handles_labels()
ax1.legend(linhas1 + linhas2, rotulos1 + rotulos2, loc="best", facecolor="#ffffff", labelcolor="#1f2328")

plt.title("Métricas extras para escolha de k - K-means", color="#1f2328")
fig.tight_layout()
plt.savefig(CAMINHO_SAIDA_GRAFICO, dpi=150, bbox_inches="tight")

print(f"\nResultados salvos em '{CAMINHO_SAIDA_CSV}'.")
print(f"Gráfico salvo em '{CAMINHO_SAIDA_GRAFICO}'.")
