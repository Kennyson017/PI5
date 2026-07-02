import string
import pandas as pd


CAMINHO_ENTRADA = "docs/Base_dia_3(1).csv"
CAMINHO_SAIDA = "dados/saida_1_sem_pontuacao.csv"

df = pd.read_csv(CAMINHO_ENTRADA)


def remover_pontuacao(texto: str) -> str:
    # Remove todos os sinais de pontuação e converte para minúsculas
    return texto.translate(str.maketrans('', '', string.punctuation)).lower()


df["text"] = df["text"].astype(str).apply(remover_pontuacao)
df.to_csv(CAMINHO_SAIDA, index=False)

print(f"{len(df)} textos processados. Saída salva em '{CAMINHO_SAIDA}'.")
print("Exemplo (linha 0):")
print(df["text"].iloc[0][:300])
