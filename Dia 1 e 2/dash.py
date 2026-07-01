"""
================================================================================
 DASHBOARD — Classificação Binária de Imagens Térmicas de Motor
 Projeto Integrador 5
================================================================================

O QUE É ESTE ARQUIVO
---------------------
Um dashboard interativo (Streamlit) para apresentar os resultados do
pipeline de classificação (`pipeline_classificacao_termica.py`):

    - Ranking de acurácia / F1 / precisão / recall por modelo
    - Matrizes de confusão interativas (por modelo)
    - Tabela comparativa simplificada
    - Gráfico de barras comparando todos os modelos
    - Análise de erros (falso positivo x falso negativo) com insight sobre
      qual erro é mais "caro" no contexto do problema (proteção do motor)
    - Hiperparâmetros escolhidos por cada modelo (GridSearchCV)
    - Galeria de amostras de imagens usadas no treino
    - Insights automáticos gerados a partir dos números (texto dinâmico)

COMO RODAR
----------
    pip install streamlit plotly pandas numpy

    1) Rode primeiro o pipeline para gerar a pasta `resultados/`:
           python pipeline_classificacao_termica.py

    2) Depois rode o dashboard:
           streamlit run dash.py

MODO DEMO
---------
Se a pasta `resultados/` (gerada pelo pipeline) ainda não existir, o
dashboard carrega automaticamente um conjunto de dados SIMULADO apenas
para fins de demonstração da interface -- útil para testar o dashboard
antes de ter os resultados reais, ou para a apresentação caso o
treinamento ainda não tenha sido rodado na máquina atual.
================================================================================
"""

import os
import json

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ------------------------------------------------------------------------
# CONFIGURAÇÃO GERAL DA PÁGINA
# ------------------------------------------------------------------------
st.set_page_config(
    page_title="Classificação Térmica do Motor — Dashboard",
    page_icon="🌡️",
    layout="wide",
)

RESULT_DIR = "resultados"
NOME_CLASSES = ["Normal (0)", "Máxima (1)"]


# ==========================================================================
# CARREGAMENTO DE DADOS (real, se existir; senão, demo simulada)
# ==========================================================================

def _gerar_dados_demo():
    """
    Gera um conjunto de resultados FICTÍCIO, apenas para que o dashboard
    possa ser demonstrado/testado mesmo sem ter rodado o pipeline real
    ainda. Os números aqui são plausíveis, mas não reais.
    """
    rng = np.random.default_rng(42)

    modelos = [
        "KNN", "Arvore de Decisao", "Random Forest",
        "SVM Linear", "SVM Polinomial (grau 2)", "Ensemble (Voto Majoritario)",
    ]
    # acurácias "plausíveis" por tipo de modelo (apenas para visual de demo)
    acc_base = [0.82, 0.78, 0.90, 0.88, 0.85, 0.92]

    linhas = []
    matrizes = {}
    n_teste = 60  # tamanho fictício do conjunto de teste (30/30)

    for nome, acc in zip(modelos, acc_base):
        acc_jit = float(np.clip(acc + rng.normal(0, 0.01), 0, 1))
        f1 = float(np.clip(acc_jit + rng.normal(0, 0.02), 0, 1))
        precisao = float(np.clip(acc_jit + rng.normal(0, 0.02), 0, 1))
        recall = float(np.clip(acc_jit + rng.normal(0, 0.02), 0, 1))

        # monta uma matriz de confusão fictícia coerente com a acurácia
        acertos = int(round(acc_jit * n_teste))
        erros = n_teste - acertos
        fn = erros // 2          # falso negativo: máxima predita como normal
        fp = erros - fn          # falso positivo: normal predita como máxima
        vp = n_teste // 2 - fn   # verdadeiro positivo (classe "máxima")
        vn = n_teste // 2 - fp   # verdadeiro negativo (classe "normal")
        cm = [[max(vn, 0), max(fp, 0)], [max(fn, 0), max(vp, 0)]]

        linhas.append({
            "Modelo": nome, "Acuracia": acc_jit, "F1-score": f1,
            "Precisao": precisao, "Recall": recall,
            "Acuracia_CV": float(np.clip(acc_jit - 0.02, 0, 1)),
        })
        matrizes[nome] = cm

    df_resultados = pd.DataFrame(linhas).sort_values("Acuracia", ascending=False).reset_index(drop=True)

    melhores_params = {
        "KNN": {"n_neighbors": 5, "weights": "distance", "metric": "euclidean"},
        "Arvore de Decisao": {"max_depth": 8, "min_samples_split": 5, "criterion": "gini"},
        "Random Forest": {"n_estimators": 200, "max_depth": 10, "min_samples_split": 2},
        "SVM Linear": {"C": 1},
        "SVM Polinomial (grau 2)": {"C": 1, "coef0": 1, "gamma": "scale"},
    }

    info_dataset = {
        "img_size": [110, 70],
        "n_treino": 200,
        "n_teste": n_teste,
        "treino_normal": 100,
        "treino_maxima": 100,
        "teste_normal": n_teste // 2,
        "teste_maxima": n_teste // 2,
        "n_atributos_por_imagem": 110 * 70,
        "tempo_execucao_segundos": 0,
    }

    return df_resultados, matrizes, melhores_params, info_dataset, None


@st.cache_data
def carregar_dados():
    """
    Tenta carregar os arquivos reais gerados por `pipeline_classificacao_termica.py`.
    Caso não existam, cai automaticamente no modo demo (dados simulados).

    Retorna
    -------
    df_resultados   : DataFrame com Modelo / Acuracia / F1 / Precisao / Recall
    matrizes        : dict {nome_modelo: matriz 2x2 (lista de listas)}
    melhores_params : dict {nome_modelo: dict de hiperparâmetros}
    info_dataset    : dict com metadados do dataset (tamanho, contagens etc.)
    amostras        : dict com arrays de imagens de amostra, ou None
    """
    caminho_csv = os.path.join(RESULT_DIR, "tabela_resultados.csv")

    if not os.path.exists(caminho_csv):
        return _gerar_dados_demo() + (True,)  # último item = flag "é_demo"

    df_resultados = pd.read_csv(caminho_csv)

    with open(os.path.join(RESULT_DIR, "matrizes_confusao.json"), encoding="utf-8") as f:
        matrizes = json.load(f)

    with open(os.path.join(RESULT_DIR, "melhores_params.json"), encoding="utf-8") as f:
        melhores_params = json.load(f)

    with open(os.path.join(RESULT_DIR, "info_dataset.json"), encoding="utf-8") as f:
        info_dataset = json.load(f)

    amostras = None
    caminho_amostras = os.path.join(RESULT_DIR, "amostras_imagens.npz")
    if os.path.exists(caminho_amostras):
        npz = np.load(caminho_amostras)
        amostras = {"normais": npz["normais"], "maximas": npz["maximas"]}

    return df_resultados, matrizes, melhores_params, info_dataset, amostras, False


df_resultados, matrizes, melhores_params, info_dataset, amostras, eh_demo = carregar_dados()


# ==========================================================================
# CABEÇALHO
# ==========================================================================
st.title("🌡️ Classificação Binária de Imagens Térmicas do Motor")
st.caption("Carga Normal (0) × Carga Máxima / Sobrecarga (1) — Projeto Integrador 5")

if eh_demo:
    st.warning(
        "⚠️ Pasta `resultados/` não encontrada — exibindo **dados de demonstração "
        "simulados**. Rode `python pipeline_classificacao_termica.py` primeiro para "
        "ver os resultados reais do seu dataset.",
        icon="⚠️",
    )

st.divider()


# ==========================================================================
# BLOCO 1 — VISÃO GERAL / KPIs
# ==========================================================================
melhor = df_resultados.iloc[0]
melhor_individual = df_resultados[~df_resultados["Modelo"].str.contains("Ensemble")].iloc[0]
ensemble_row = df_resultados[df_resultados["Modelo"].str.contains("Ensemble")]

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Imagens de treino", info_dataset["n_treino"])
col2.metric("Imagens de teste", info_dataset["n_teste"])
col3.metric("Tamanho da imagem Adaptada", f"{info_dataset['img_size'][0]}×{info_dataset['img_size'][1]}")
col4.metric("Tamanho da imagem Original", f"440x280")
col5.metric("Atributos (pixels)", f"{info_dataset['n_atributos_por_imagem']:,}".replace(",", "."))
col6.metric("🏆 Melhor modelo", melhor["Modelo"], f"{melhor['Acuracia']:.1%} acurácia")

st.divider()


# ==========================================================================
# BLOCO 2 — TABELA COMPARATIVA + GRÁFICO DE BARRAS
# ==========================================================================
st.header("📊 Comparação entre Modelos")

c1, c2 = st.columns([1.1, 1])

with c1:
    st.subheader("Tabela-resumo")
    df_show = df_resultados.copy()
    for col in ["Acuracia", "F1-score", "Precisao", "Recall", "Acuracia_CV"]:
        if col in df_show.columns:
            df_show[col] = df_show[col].apply(lambda x: f"{x:.1%}" if pd.notnull(x) else "—")
    st.dataframe(
        df_show.rename(columns={"Acuracia_CV": "Acurácia (validação cruzada)"}),
        use_container_width=True,
        hide_index=True,
    )

with c2:
    st.subheader("Acurácia × F1-score por modelo")
    df_plot = df_resultados.melt(
        id_vars="Modelo",
        value_vars=["Acuracia", "F1-score"],
        var_name="Métrica",
        value_name="Valor",
    )
    fig_bar = px.bar(
        df_plot, x="Modelo", y="Valor", color="Métrica",
        barmode="group", text_auto=".1%",
        color_discrete_map={"Acuracia": "#1f77b4", "F1-score": "#ff7f0e"},
    )
    fig_bar.update_layout(yaxis_tickformat=".0%", xaxis_title="", legend_title="",
                           height=420, margin=dict(t=10))
    st.plotly_chart(fig_bar, use_container_width=True)

st.divider()


# ==========================================================================
# BLOCO 3 — MATRIZ DE CONFUSÃO INTERATIVA
# ==========================================================================
st.header("🧩 Matriz de Confusão por Modelo")

modelo_selecionado = st.selectbox("Selecione o modelo:", df_resultados["Modelo"].tolist())

cm = np.array(matrizes[modelo_selecionado])
linha_modelo = df_resultados[df_resultados["Modelo"] == modelo_selecionado].iloc[0]

cc1, cc2 = st.columns([1, 1.3])

with cc1:
    fig_cm = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=NOME_CLASSES, y=NOME_CLASSES,
            colorscale="Blues",
            text=cm, texttemplate="%{text}",
            textfont={"size": 20},
            showscale=False,
        )
    )
    fig_cm.update_layout(
        title=f"Matriz de Confusão — {modelo_selecionado}",
        xaxis_title="Predito", yaxis_title="Real",
        yaxis=dict(autorange="reversed"),
        height=420,
    )
    st.plotly_chart(fig_cm, use_container_width=True)

with cc2:
    vn, fp = cm[0]
    fn, vp = cm[1]
    total = cm.sum()

    st.metric("Acurácia", f"{linha_modelo['Acuracia']:.1%}")
    m1, m2 = st.columns(2)
    m1.metric("F1-score", f"{linha_modelo['F1-score']:.1%}")
    m2.metric("Recall (classe Máxima)", f"{linha_modelo['Recall']:.1%}")

    st.markdown(
        f"""
        **Leitura da matriz** (total de {total} imagens de teste):
        - ✅ Verdadeiro Normal: **{vn}**
        - ✅ Verdadeiro Máxima: **{vp}**
        - ⚠️ Falso Positivo (normal classificado como máxima): **{fp}**
        - 🔴 Falso Negativo (máxima classificado como normal): **{fn}**
        """
    )

    if fn > fp:
        st.error(
            f"⚠️ **Atenção**: este modelo erra mais por **falso negativo** "
            f"({fn} casos) — ou seja, classifica o motor como NORMAL quando "
            f"na verdade está em SOBRECARGA. Esse é o erro mais perigoso no "
            f"contexto do problema (risco de quebra do motor sem alerta)."
        )
    elif fp > fn:
        st.info(
            f"ℹ️ Este modelo erra mais por **falso positivo** ({fp} casos) — "
            f"classifica o motor como em sobrecarga quando está normal. É um "
            f"erro mais conservador (gera alarme falso), porém mais seguro "
            f"para a integridade do equipamento do que o falso negativo."
        )
    else:
        st.success("Erros equilibrados entre falso positivo e falso negativo.")

st.divider()


# ==========================================================================
# BLOCO 4 — HIPERPARÂMETROS ESCOLHIDOS (GridSearchCV)
# ==========================================================================
st.header("⚙️ Hiperparâmetros Selecionados (GridSearchCV)")

linhas_params = []
for nome, params in melhores_params.items():
    linha = {"Modelo": nome}
    linha.update(params)
    linhas_params.append(linha)

df_params = pd.DataFrame(linhas_params).set_index("Modelo")
st.dataframe(df_params, use_container_width=True)

st.divider()


# ==========================================================================
# BLOCO 5 — AMOSTRAS DE IMAGENS (se disponíveis)
# ==========================================================================
if amostras is not None:
    st.header("🖼️ Amostras do Conjunto de Treino")
    g1, g2 = st.columns(2)

    with g1:
        st.caption("Carga Normal (classe 0)")
        cols = st.columns(len(amostras["normais"]))
        for c, img in zip(cols, amostras["normais"]):
            c.image(img, use_container_width=True, clamp=True)

    with g2:
        st.caption("Carga Máxima (classe 1)")
        cols = st.columns(len(amostras["maximas"]))
        for c, img in zip(cols, amostras["maximas"]):
            c.image(img, use_container_width=True, clamp=True)

    st.divider()


# ==========================================================================
# BLOCO 6 — INSIGHTS 
# ==========================================================================
st.header("💡 Insights")

ensemble_existe = not ensemble_row.empty
texto_insights = []

texto_insights.append(
    f"- O modelo com **melhor acurácia individual** foi **{melhor_individual['Modelo']}** "
    f"({melhor_individual['Acuracia']:.1%}), com F1-score de {melhor_individual['F1-score']:.1%}."
)

if ensemble_existe:
    acc_ens = ensemble_row.iloc[0]["Acuracia"]
    if acc_ens > melhor_individual["Acuracia"]:
        diff = acc_ens - melhor_individual["Acuracia"]
        texto_insights.append(
            f"- O **ensemble (voto majoritário)** superou o melhor modelo individual "
            f"em **{diff:.1%}** de acurácia ({acc_ens:.1%}) — indício de que os modelos "
            f"cometem erros pouco correlacionados entre si, então a combinação ajuda."
        )
    elif acc_ens < melhor_individual["Acuracia"]:
        diff = melhor_individual["Acuracia"] - acc_ens
        texto_insights.append(
            f"- O **ensemble** ficou **{diff:.1%}** abaixo do melhor modelo individual "
            f"({acc_ens:.1%} vs {melhor_individual['Acuracia']:.1%}) — sinal de que algum "
            f"modelo mais fraco do grupo está \"puxando\" a decisão final para baixo."
        )
    else:
        texto_insights.append(
            "- O ensemble teve desempenho **equivalente** ao melhor modelo individual."
        )

# Acha o modelo com maior risco de falso negativo (sob ótica de segurança do motor)
fn_por_modelo = {nome: np.array(m)[1][0] for nome, m in matrizes.items()}
modelo_mais_fn = max(fn_por_modelo, key=fn_por_modelo.get)
texto_insights.append(
    f"- O modelo com **mais falsos negativos** (motor em sobrecarga classificado "
    f"como normal) é **{modelo_mais_fn}**, com {fn_por_modelo[modelo_mais_fn]} casos. "
    f"Em um cenário real de proteção de equipamento, esse é o erro mais caro — "
    f"vale priorizar modelos com **alto recall** na classe 'Máxima', mesmo que "
    f"a acurácia geral seja um pouco menor."
)

variacao = df_resultados["Acuracia"].max() - df_resultados["Acuracia"].min()
if variacao < 0.05:
    texto_insights.append(
        f"- A variação de acurácia entre os modelos foi pequena (**{variacao:.1%}**), "
        f"sugerindo que o problema é relativamente bem separável com o "
        f"pré-processamento atual (tamanho de imagem, padronização)."
    )
else:
    texto_insights.append(
        f"- Houve uma variação relevante de acurácia entre os modelos (**{variacao:.1%}**), "
        f"o que sugere sensibilidade ao tipo de algoritmo — vale testar outros valores "
        f"de `IMG_SIZE` ou engenharia de atributos adicional."
    )

for linha in texto_insights:
    st.markdown(linha)

st.caption(
    "Dashboard gerado para apresentação prática do Exercício 02 — "
    "Projeto Integrador 5."
)