"""
================================================================================
 CLASSIFICAÇÃO BINÁRIA DE IMAGENS TÉRMICAS DE MOTOR
 Projeto Integrador 5 — Exercício 02
================================================================================

OBJETIVO
--------
Classificar imagens térmicas de um motor industrial em duas classes:

    classe 0  ->  "M" (Médio)  -> carga NORMAL
    classe 1  ->  "F" (Full)   -> carga MÁXIMA / sobrecarga

Esse é exatamente o problema apresentado no roteiro da aula: usar visão térmica
+ aprendizado de máquina para identificar, em tempo real, se o motor está
operando em sobrecarga (evitando ter que escolher entre "parar tudo" ou
"arriscar quebrar o motor").

PIPELINE (5 ETAPAS)
--------------------
 1) Carregar as imagens das pastas de treino/teste e pré-processar
    (escala de cinza + redimensionamento)
 2) Vetorizar cada imagem (flatten: matriz -> vetor de pixels) e padronizar
    (StandardScaler) -- necessário pois os modelos (KNN, SVM) são sensíveis
    à escala dos atributos
 3) Treinar 5 algoritmos de classificação, cada um com busca de
    hiperparâmetros via GridSearchCV (validação cruzada)
 4) Avaliar cada modelo no conjunto de TESTE (nunca usado no treino):
    acurácia, F1-score, matriz de confusão e relatório de classificação
 5) Combinar os 5 modelos em um ensemble por voto da maioria (hard voting)
    e comparar com os modelos individuais

PERSISTÊNCIA DE RESULTADOS
---------------------------
Ao final, todos os resultados relevantes (tabela de métricas, matrizes de
confusão, hiperparâmetros escolhidos, predições, amostras de imagens) são
salvos em disco na pasta `resultados/`. Esses arquivos são lidos pelo
dashboard interativo `dash.py`, que deve ser executado separadamente
(`streamlit run dash.py`) para a apresentação dos resultados.

COMO RODAR
----------
    python pipeline_classificacao_termica.py

Pré-requisitos (instalar uma vez):
    pip install numpy pandas matplotlib pillow scikit-learn

Estrutura de pastas esperada (relativa a este arquivo):
    imgs/Bases_Img_Termica_Motor/
        ├── Treino_H_F/   (imagens de treino, carga MÁXIMA)
        ├── Treino_H_M/   (imagens de treino, carga NORMAL)
        ├── Teste_H_F/    (imagens de teste,  carga MÁXIMA)
        └── Teste_H_M/    (imagens de teste,  carga NORMAL)
================================================================================
"""

import os
import glob
import json
import pickle
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


# ==============================================================================
# CONFIGURAÇÕES GERAIS
# ==============================================================================

# Semente fixa -> garante que os resultados sejam reprodutíveis entre execuções
# (mesma divisão de validação cruzada, mesma inicialização de Random Forest etc.)
np.random.seed(42)

# Pasta raiz onde estão as subpastas de imagens (ver estrutura no cabeçalho)
BASE_DIR = "imgs/Bases_Img_Termica_Motor"

# Tamanho fixo (largura, altura) para o qual TODAS as imagens são redimensionadas.
# Isso é necessário porque os modelos de ML clássico (KNN, SVM, Árvore, RF)
# exigem que todas as amostras tenham o MESMO número de atributos (pixels).
# Imagem original de referência: 440x280 (proporção ~1/3). Aqui usamos um
# tamanho reduzido para diminuir dimensionalidade e tempo de treino.
IMG_SIZE = (110, 70)  # (largura, altura)

# Mapeamento pasta -> rótulo da classe
#   0 = carga normal   (pastas "..._H_M", M de "Médio")
#   1 = carga máxima   (pastas "..._H_F", F de "Full")
PASTAS = {
    "train": {"F": "Treino_H_F", "M": "Treino_H_M"},
    "test": {"F": "Teste_H_F", "M": "Teste_H_M"},
}

NOME_CLASSES = ["Normal (0)", "Máxima (1)"]

# Pasta de saída onde tudo que o dashboard precisa será salvo
OUT_DIR = "resultados"
os.makedirs(OUT_DIR, exist_ok=True)


# ==============================================================================
# ETAPA 1 — CARREGAMENTO E PRÉ-PROCESSAMENTO DAS IMAGENS
# ==============================================================================

def carregar_pasta(caminho: str, label: int):
    """
    Lê todas as imagens .jpg de uma pasta, converte para escala de cinza
    e redimensiona para IMG_SIZE.

    Parâmetros
    ----------
    caminho : str
        Caminho da pasta contendo os arquivos .jpg
    label : int
        Rótulo (0 ou 1) a ser atribuído a todas as imagens dessa pasta

    Retorna
    -------
    imagens : list[np.ndarray]  -> cada item é uma matriz (altura, largura)
    labels  : list[int]         -> rótulo repetido para cada imagem
    """
    imagens, labels = [], []
    arquivos = sorted(glob.glob(os.path.join(caminho, "*.jpg")))

    for arq in arquivos:
        # "L" = escala de cinza (1 canal, valores 0-255) -- equivalente ao que
        # foi visto no roteiro (cada pixel é um valor único de intensidade)
        img = Image.open(arq).convert("L")
        img = img.resize(IMG_SIZE)  # padroniza dimensões entre amostras
        imagens.append(np.array(img, dtype=np.uint8))
        labels.append(label)

    return imagens, labels


def montar_conjunto(split: str):
    """
    Monta o conjunto completo (treino OU teste), juntando as imagens de
    carga máxima (F -> 1) e carga normal (M -> 0).

    Parâmetros
    ----------
    split : str
        "train" ou "test"

    Retorna
    -------
    imagens : np.ndarray, shape (N, altura, largura)
    labels  : np.ndarray, shape (N,)
    """
    imgs_f, lab_f = carregar_pasta(os.path.join(BASE_DIR, PASTAS[split]["F"]), 1)
    imgs_m, lab_m = carregar_pasta(os.path.join(BASE_DIR, PASTAS[split]["M"]), 0)

    imagens = imgs_f + imgs_m
    labels = lab_f + lab_m
    return np.array(imagens), np.array(labels)


# ==============================================================================
# ETAPA 4 (função auxiliar usada em treino e ensemble) — AVALIAÇÃO DE MODELO
# ==============================================================================

def avaliar_modelo(nome: str, modelo, X_test, y_test, salvar_figura=True):
    """
    Avalia um modelo já treinado no conjunto de teste:
        - calcula acurácia, F1, precisão e recall
        - imprime o classification_report (métricas por classe)
        - gera e salva a figura da matriz de confusão
        - retorna um dicionário com tudo (usado depois para montar a
          tabela-resumo e para alimentar o dashboard)
    """
    y_pred = modelo.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precisao = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"=== {nome} ===")
    print(f"Acurácia: {acc:.4f} | F1-score: {f1:.4f} | "
          f"Precisão: {precisao:.4f} | Recall: {recall:.4f}")
    print(classification_report(y_test, y_pred, target_names=NOME_CLASSES))

    if salvar_figura:
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1], NOME_CLASSES)
        ax.set_yticks([0, 1], NOME_CLASSES)
        ax.set_xlabel("Predito")
        ax.set_ylabel("Real")
        ax.set_title(f"Matriz de Confusão — {nome}")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        nome_arquivo = nome.lower().replace(" ", "_").replace("(", "").replace(")", "")
        fig.savefig(os.path.join(OUT_DIR, f"cm_{nome_arquivo}.png"), dpi=120)
        plt.close(fig)

    return {
        "modelo": nome,
        "acuracia": acc,
        "f1": f1,
        "precisao": precisao,
        "recall": recall,
        "matriz_confusao": cm.tolist(),  # listas -> serializável em JSON
        "y_pred": y_pred.tolist(),
    }


# ==============================================================================
# FUNÇÃO PRINCIPAL — executa o pipeline completo, ponta a ponta
# ==============================================================================

def main():
    t0 = time.time()

    # ---------------------------------------------------------------
    # ETAPA 1: carregar imagens
    # ---------------------------------------------------------------
    print(">> Etapa 1/5: carregando imagens...")
    imgs_train, y_train = montar_conjunto("train")
    imgs_test, y_test = montar_conjunto("test")

    print(f"Treino: {imgs_train.shape[0]} imagens | Teste: {imgs_test.shape[0]} imagens")
    print(f"Dimensão de cada imagem: {imgs_train.shape[1:]}")
    print(f"Distribuição treino -> normal(0): {(y_train == 0).sum()} | "
          f"máxima(1): {(y_train == 1).sum()}")
    print(f"Distribuição teste  -> normal(0): {(y_test == 0).sum()} | "
          f"máxima(1): {(y_test == 1).sum()}")

    # Guarda algumas imagens de amostra (cru, sem padronização) para o
    # dashboard conseguir exibir exemplos visuais de cada classe
    idx_normal = np.where(y_train == 0)[0][:4]
    idx_maxima = np.where(y_train == 1)[0][:4]
    np.savez_compressed(
        os.path.join(OUT_DIR, "amostras_imagens.npz"),
        normais=imgs_train[idx_normal],
        maximas=imgs_train[idx_maxima],
    )

    # ---------------------------------------------------------------
    # ETAPA 2: vetorização (flatten) + padronização
    # ---------------------------------------------------------------
    print("\n>> Etapa 2/5: vetorizando e padronizando...")

    # flatten: (N, altura, largura) -> (N, altura*largura)
    # cada linha passa a ser um "vetor de características" (1 atributo por pixel)
    X_train = imgs_train.reshape(imgs_train.shape[0], -1).astype(np.float64)
    X_test = imgs_test.reshape(imgs_test.shape[0], -1).astype(np.float64)

    # StandardScaler: centraliza (média 0) e normaliza (desvio padrão 1) cada
    # atributo (pixel). É AJUSTADO SOMENTE NO TREINO (fit_transform) e depois
    # apenas APLICADO no teste (transform) -- isso evita "vazamento de
    # informação" (data leakage) do conjunto de teste para o treino.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("X_train:", X_train_scaled.shape)
    print("X_test :", X_test_scaled.shape)

    # ---------------------------------------------------------------
    # ETAPA 3: treinamento com busca de hiperparâmetros (GridSearchCV)
    # ---------------------------------------------------------------
    print("\n>> Etapa 3/5: treinando modelos (GridSearchCV)...")

    # Para cada algoritmo, definimos o estimador-base e a grade de
    # hiperparâmetros a testar. O GridSearchCV testa TODAS as combinações
    # usando validação cruzada (cv=5) e fica com a combinação de melhor
    # acurácia média.
    param_grids = {
        "KNN": {
            "estimator": KNeighborsClassifier(),
            "param_grid": {
                "n_neighbors": [3, 5, 7, 9, 11],
                "weights": ["uniform", "distance"],
                "metric": ["euclidean", "manhattan"],
            },
        },
        "Arvore de Decisao": {
            "estimator": DecisionTreeClassifier(random_state=42),
            "param_grid": {
                "max_depth": [3, 5, 8, 12, None],
                "min_samples_split": [2, 5, 10],
                "criterion": ["gini", "entropy"],
            },
        },
        "Random Forest": {
            "estimator": RandomForestClassifier(random_state=42),
            "param_grid": {
                "n_estimators": [100, 200, 300],
                "max_depth": [5, 10, None],
                "min_samples_split": [2, 5],
            },
        },
        "SVM Linear": {
            "estimator": SVC(kernel="linear", random_state=42),
            "param_grid": {
                "C": [0.01, 0.1, 1, 10, 100],
            },
        },
        "SVM Polinomial (grau 2)": {
            "estimator": SVC(kernel="poly", degree=2, random_state=42),
            "param_grid": {
                "C": [0.1, 1, 10],
                "coef0": [0, 1, 5],
                "gamma": ["scale", "auto"],
            },
        },
    }

    modelos_treinados = {}
    melhores_params = {}
    cv_scores = {}

    for nome, config in param_grids.items():
        print(f"Treinando: {nome} ...")
        grid = GridSearchCV(
            estimator=config["estimator"],
            param_grid=config["param_grid"],
            cv=5,                 # 5-fold cross-validation interna
            scoring="accuracy",
            n_jobs=-1,             # usa todos os núcleos de CPU disponíveis
        )
        grid.fit(X_train_scaled, y_train)

        modelos_treinados[nome] = grid.best_estimator_
        melhores_params[nome] = grid.best_params_
        cv_scores[nome] = grid.best_score_

        print(f"  Melhores parâmetros: {grid.best_params_}")
        print(f"  Acurácia média (CV): {grid.best_score_:.4f}\n")

    # ---------------------------------------------------------------
    # ETAPA 4: avaliação individual no conjunto de TESTE
    # ---------------------------------------------------------------
    print(">> Etapa 4/5: avaliando modelos no conjunto de teste...")
    resultados = []
    for nome, modelo in modelos_treinados.items():
        r = avaliar_modelo(nome, modelo, X_test_scaled, y_test)
        r["acuracia_cv"] = cv_scores[nome]  # guarda também a acc. de validação
        resultados.append(r)

    # ---------------------------------------------------------------
    # ETAPA 5: ensemble por voto da maioria (hard voting)
    # ---------------------------------------------------------------
    print("\n>> Etapa 5/5: montando ensemble (voto da maioria)...")

    ensemble = VotingClassifier(
        estimators=[(nome, modelo) for nome, modelo in modelos_treinados.items()],
        voting="hard",  # "hard" = vota na CLASSE prevista por cada modelo
                          # (não usa probabilidades / "soft" voting)
    )
    # Os estimadores já vêm ajustados (best_estimator_ do GridSearchCV);
    # o .fit() do VotingClassifier reaproveita esse ajuste e organiza os
    # rótulos de classe internamente.
    ensemble.fit(X_train_scaled, y_train)

    resultado_ensemble = avaliar_modelo(
        "Ensemble (Voto Majoritario)", ensemble, X_test_scaled, y_test
    )
    resultado_ensemble["acuracia_cv"] = None  # ensemble não passou por GridSearch
    resultados.append(resultado_ensemble)

    # ---------------------------------------------------------------
    # TABELA-RESUMO FINAL
    # ---------------------------------------------------------------
    df_resultados = pd.DataFrame(
        [
            {
                "Modelo": r["modelo"],
                "Acuracia": r["acuracia"],
                "F1-score": r["f1"],
                "Precisao": r["precisao"],
                "Recall": r["recall"],
                "Acuracia_CV": r["acuracia_cv"],
            }
            for r in resultados
        ]
    ).sort_values("Acuracia", ascending=False).reset_index(drop=True)

    print("\n===== TABELA-RESUMO =====")
    print(df_resultados.to_string(index=False))

    # ---------------------------------------------------------------
    # PERSISTÊNCIA — tudo que o dashboard (dash.py) vai consumir
    # ---------------------------------------------------------------
    print(f"\nSalvando resultados em ./{OUT_DIR}/ ...")

    df_resultados.to_csv(os.path.join(OUT_DIR, "tabela_resultados.csv"), index=False)

    with open(os.path.join(OUT_DIR, "matrizes_confusao.json"), "w", encoding="utf-8") as f:
        json.dump(
            {r["modelo"]: r["matriz_confusao"] for r in resultados},
            f, ensure_ascii=False, indent=2,
        )

    with open(os.path.join(OUT_DIR, "melhores_params.json"), "w", encoding="utf-8") as f:
        json.dump(melhores_params, f, ensure_ascii=False, indent=2, default=str)

    with open(os.path.join(OUT_DIR, "predicoes.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "y_test": y_test.tolist(),
                "predicoes": {r["modelo"]: r["y_pred"] for r in resultados},
            },
            f, ensure_ascii=False, indent=2,
        )

    info_dataset = {
        "img_size": IMG_SIZE,
        "n_treino": int(imgs_train.shape[0]),
        "n_teste": int(imgs_test.shape[0]),
        "treino_normal": int((y_train == 0).sum()),
        "treino_maxima": int((y_train == 1).sum()),
        "teste_normal": int((y_test == 0).sum()),
        "teste_maxima": int((y_test == 1).sum()),
        "n_atributos_por_imagem": int(X_train.shape[1]),
        "tempo_execucao_segundos": round(time.time() - t0, 1),
    }
    with open(os.path.join(OUT_DIR, "info_dataset.json"), "w", encoding="utf-8") as f:
        json.dump(info_dataset, f, ensure_ascii=False, indent=2)

    # Modelos + scaler ficam salvos em pickle, caso se queira classificar
    # uma nova imagem depois (ex: Exercício Extra 2 do roteiro)
    with open(os.path.join(OUT_DIR, "modelos_treinados.pkl"), "wb") as f:
        pickle.dump(
            {"modelos": modelos_treinados, "ensemble": ensemble, "scaler": scaler,
             "img_size": IMG_SIZE},
            f,
        )

    print("Concluído! Para visualizar os resultados, rode:")
    print("    streamlit run dash.py")


if __name__ == "__main__":
    main()
