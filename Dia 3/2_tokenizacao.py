import json
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize

# Baixa o modelo necessário para tokenizar
nltk.download('punkt')
nltk.download('punkt_tab')

CAMINHO_ENTRADA = "dados/saida_1_sem_pontuacao.csv"
CAMINHO_SAIDA = "dados/saida_2_tokenizado.csv"

df = pd.read_csv(CAMINHO_ENTRADA)

df["tokens"] = df["text"].astype(str).apply(
    lambda texto: word_tokenize(texto, language='portuguese')
)

# Listas de tokens são salvas como JSON para poderem ser lidas na próxima etapa
df[["tokens"]].assign(tokens=df["tokens"].apply(json.dumps)).to_csv(CAMINHO_SAIDA, index=False)

print(f"{len(df)} textos tokenizados. Saída salva em '{CAMINHO_SAIDA}'.")
print("Exemplo (linha 0):")
print(df["tokens"].iloc[0][:20])
