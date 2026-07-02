import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

CAMINHO_ENTRADA = "dados/saida_4_lematizado.csv"
CAMINHO_SAIDA_GRAFICO = "dados/kmeans_gap_statistic.png"

K_MINIMO = 2
K_MAXIMO = 15
K_ALTERNATIVO = 4
TOKEN_PATTERN = r"(?u)\b[a-zA-ZÀ-ÿ][a-zA-ZÀ-ÿ]{2,}\b"
N_COMPONENTES_SVD = 100
KMEANS_CONFIG = {
    "n_init": 5,
    "max_iter": 300,
    "algorithm": "lloyd",
    "random_state": 42,
}
B = 5  # número de datasets de referência (distribuição nula, sem estrutura de cluster)

df = pd.read_csv(CAMINHO_ENTRADA)
df["tokens"] = df["tokens"].apply(json.loads)
documentos = [" ".join(tokens) for tokens in df["tokens"]]

# Mesma vetorização usada no cotovelo e no silhouette, para poder comparar os resultados.
vectorizer = TfidfVectorizer(min_df=5, max_df=0.9, token_pattern=TOKEN_PATTERN)
X_tfidf = vectorizer.fit_transform(documentos)

# Gap Statistic exige dados densos para gerar referências uniformes.
# A redução SVD mantém a análise viável para texto esparso de alta dimensão.
n_componentes = min(N_COMPONENTES_SVD, X_tfidf.shape[1] - 1)
svd = TruncatedSVD(n_components=n_componentes, random_state=42)
X = svd.fit_transform(X_tfidf).astype(np.float32)

print(
    f"Matriz TF-IDF reduzida por SVD: {X.shape[0]} documentos x {X.shape[1]} componentes "
    f"({svd.explained_variance_ratio_.sum():.2%} da variância)."
)

rng = np.random.default_rng(42)
minimos = X.min(axis=0)
maximos = X.max(axis=0)


def dispersao(dados, k):
    kmeans = KMeans(n_clusters=k, **KMEANS_CONFIG)
    kmeans.fit(dados)
    return kmeans.inertia_


gaps = []
erros = []
for k in range(K_MINIMO, K_MAXIMO + 1):
    w_real = dispersao(X, k)

    # Gera B conjuntos de dados aleatórios (sem estrutura) dentro dos mesmos limites
    # de cada palavra/coluna, e mede a dispersão do K-means neles.
    log_w_referencia = []
    for _ in range(B):
        referencia = rng.uniform(minimos, maximos, size=X.shape).astype(np.float32)
        w_ref = dispersao(referencia, k)
        log_w_referencia.append(np.log(w_ref))

    gap = np.mean(log_w_referencia) - np.log(w_real)
    sd = np.std(log_w_referencia)
    erro = sd * np.sqrt(1 + 1 / B)

    gaps.append(gap)
    erros.append(erro)
    print(f"k={k}: gap={gap:.4f} (erro={erro:.4f})")

# Critério de Tibshirani et al. (2001): menor k tal que Gap(k) >= Gap(k+1) - erro(k+1)
melhor_k = K_MAXIMO
for i in range(len(gaps) - 1):
    if gaps[i] >= gaps[i + 1] - erros[i + 1]:
        melhor_k = K_MINIMO + i
        break

print(f"\nMelhor k pelo Gap Statistic: {melhor_k}")

fig, ax = plt.subplots(figsize=(8, 5), facecolor="#f6f8fa")
ax.set_facecolor("#ffffff")
ks = list(range(K_MINIMO, K_MAXIMO + 1))
ax.errorbar(ks, gaps, yerr=erros, marker="o", capsize=3, color="#0969da")
indice_melhor = ks.index(melhor_k)
ax.scatter(
    [melhor_k],
    [gaps[indice_melhor]],
    color="red",
    s=90,
    zorder=5,
    label=f"melhor k={melhor_k}",
)
indice_alternativo = ks.index(K_ALTERNATIVO)
ax.scatter(
    [K_ALTERNATIVO],
    [gaps[indice_alternativo]],
    color="green",
    s=90,
    zorder=5,
    label=f"k={K_ALTERNATIVO} interpretável",
)
ax.legend(facecolor="#ffffff", labelcolor="#1f2328")
ax.set_xlabel("Número de clusters (k)", color="#57606a")
ax.set_ylabel("Gap statistic", color="#57606a")
ax.set_title("Gap Statistic - K-means", color="#1f2328")
ax.set_xticks(range(K_MINIMO, K_MAXIMO + 1))
ax.tick_params(colors="#57606a")
ax.grid(True, color="#d0d7de", alpha=0.8)
for spine in ax.spines.values():
    spine.set_color("#d0d7de")
plt.savefig(CAMINHO_SAIDA_GRAFICO, dpi=150, bbox_inches="tight")

print(f"Gráfico salvo em '{CAMINHO_SAIDA_GRAFICO}'.")
