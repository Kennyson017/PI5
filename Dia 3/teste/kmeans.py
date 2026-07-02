import pandas as pd
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np

# 1. CARREGAMENTO E PRÉ-PROCESSAMENTO
# Assumindo que a coluna de texto no CSV se chama 'texto'
df = pd.read_csv('docs/Base_dia_3(1).csv')
nlp = spacy.load("pt_core_news_sm")

def preprocess_text(text):
    # Processa o texto (tokenização, lematização, remoção de stop words e pontuação)
    doc = nlp(str(text).lower())
    lemmas = [
        token.lemma_ for token in doc 
        if not token.is_stop and not token.is_punct and token.is_alpha
    ]
    return " ".join(lemmas)

print("Iniciando pré-processamento das 1500 linhas...")
df['texto_processado'] = df['text'].apply(preprocess_text)

# 2. MODELO BAG OF WORDS (BoW) - Item A do Slide
# min_df=5 (palavra deve aparecer em pelo menos 5 docs)
# max_df=0.85 (ignora palavras presentes em mais de 85% dos docs)
vectorizer = CountVectorizer(min_df=5, max_df=0.85)
X_bow = vectorizer.fit_transform(df['texto_processado'])

# 3. K-MEANS E NÚMERO IDEAL DE GRUPOS - Itens B e C do Slide
inertias = []
silhouette_scores = []
K_range = range(2, 16)

print("Calculando o número ideal de clusters...")
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_bow)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_bow, kmeans.labels_))

# Plotagem para justificar a escolha (Método do Cotovelo e Silhouette)
fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(K_range, inertias, marker='o', color='b', label='Inércia (Cotovelo)')
ax1.set_xlabel('Número de Clusters (k)')
ax1.set_ylabel('Inércia', color='b')

ax2 = ax1.twinx()
ax2.plot(K_range, silhouette_scores, marker='s', color='r', label='Silhouette Score')
ax2.set_ylabel('Silhouette Score', color='r')

plt.title('Avaliação do Número Ideal de Tópicos')
plt.show()

# Baseado no gráfico, defina o melhor K (substitua o valor abaixo pelo resultado visível no Silhouette/Cotovelo)
BEST_K = np.argmax(silhouette_scores) + 2 # +2 pois o range começa em 2
print(f"Número ideal matemático detectado: {BEST_K}")

# Executa o modelo final com o K ideal
kmeans_final = KMeans(n_clusters=BEST_K, random_state=42, n_init=10)
df['cluster'] = kmeans_final.fit_predict(X_bow)

# 4. QUAIS SÃO OS TÓPICOS? - Item D do Slide
print("\n--- TÓPICOS DE CADA GRUPO ---")
vocabulario = np.array(vectorizer.get_feature_names_out())
centroides = kmeans_final.cluster_centers_

for i in range(BEST_K):
    # Pega os índices das 10 palavras com maiores pesos neste cluster
    top_indices = centroides[i].argsort()[::-1][:10]
    top_palavras = vocabulario[top_indices]
    print(f"Grupo {i}: {', '.join(top_palavras)}")