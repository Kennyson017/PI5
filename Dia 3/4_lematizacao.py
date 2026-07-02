import json
import pandas as pd
import spacy
from spacy.cli import download

# O parser e o NER são desabilitados pois só precisamos do lematizador (mais rápido).
try:
    nlp = spacy.load("pt_core_news_sm", disable=["parser", "ner"])
except OSError:
    download("pt_core_news_sm")
    nlp = spacy.load("pt_core_news_sm", disable=["parser", "ner"])

CAMINHO_ENTRADA = "dados/saida_3_sem_stopwords.csv"
CAMINHO_SAIDA = "dados/saida_4_lematizado.csv"

df = pd.read_csv(CAMINHO_ENTRADA)
df["tokens"] = df["tokens"].apply(json.loads)

# O spaCy trabalha com texto em string, não com listas de tokens
textos = df["tokens"].apply(lambda tokens: " ".join(tokens))

lemas = [
    [token.lemma_.lower().strip() for token in doc if token.lemma_.strip()]
    for doc in nlp.pipe(textos, batch_size=50)
]

df["tokens"] = [json.dumps(lista) for lista in lemas]
df.to_csv(CAMINHO_SAIDA, index=False)

print(f"{len(df)} textos lematizados. Saída salva em '{CAMINHO_SAIDA}'.")
print("Exemplo (linha 0):")
print(lemas[0][:20])
