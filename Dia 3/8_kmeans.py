import json

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

CAMINHO_ENTRADA = "dados/saida_4_lematizado.csv"
CAMINHO_SAIDA = "dados/saida_5_kmeans.csv"
CAMINHO_SAIDA_TOPICOS = "dados/saida_5_topicos_clusters.csv"

N_CLUSTERS = 4
TOP_N_TERMOS = 12
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
X = vectorizer.fit_transform(documentos)

kmeans = KMeans(n_clusters=N_CLUSTERS, **KMEANS_CONFIG)
df["cluster"] = kmeans.fit_predict(X)

df[["cluster"]].assign(tokens=[json.dumps(t) for t in df["tokens"]]).to_csv(CAMINHO_SAIDA, index=False)

print(f"{N_CLUSTERS} clusters gerados. Saída salva em '{CAMINHO_SAIDA}'.")
print("\nTamanho de cada cluster:")
print(df["cluster"].value_counts().sort_index())

print("\nPalavras mais características de cada cluster:")
termos = vectorizer.get_feature_names_out()
centros_ordenados = kmeans.cluster_centers_.argsort()[:, ::-1]
topicos = []
for cluster_id in range(N_CLUSTERS):
    top_termos = [termos[i] for i in centros_ordenados[cluster_id, :TOP_N_TERMOS]]
    topicos.append({
        "cluster": cluster_id,
        "quantidade_textos": int((df["cluster"] == cluster_id).sum()),
        "termos_principais": ", ".join(top_termos),
    })
    print(f"Cluster {cluster_id}: {', '.join(top_termos)}")

pd.DataFrame(topicos).to_csv(CAMINHO_SAIDA_TOPICOS, index=False)
print(f"\nTÃ³picos dos clusters salvos em '{CAMINHO_SAIDA_TOPICOS}'.")
