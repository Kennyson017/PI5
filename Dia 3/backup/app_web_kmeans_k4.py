"""
Dashboard Web Interativo - Agrupamento de Textos com K-means
============================================================
Como rodar:
  1. Abra o terminal na pasta PI_5.1
  2. Execute:  python app_web_kmeans.py
  3. Abra no navegador: http://127.0.0.1:8051
============================================================
"""

import json
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

import dash
from dash import Input, Output, dash_table, dcc, html


# Paleta (mesma linha visual do dashboard PI_5)
BG = "#f6f8fa"
CARD = "#ffffff"
BORDA = "#d0d7de"
VERDE = "#1a7f37"
VERMELHO = "#cf222e"
AZUL = "#0969da"
AMARELO = "#9a6700"
ROXO = "#8250df"
BRANCO = "#1f2328"
CINZA = "#57606a"
HEADER = "#0f3460"

CLUSTER_CORES = {
    0: ROXO,
    1: AZUL,
    2: VERDE,
    3: VERMELHO,
}

CLUSTER_NOMES = {
    0: "Blog / astronomia",
    1: "Esporte / futebol",
    2: "Gastronomia",
    3: "Ciência / pesquisa",
}

K_MIN = 2
K_MAX = 15
KS = list(range(K_MIN, K_MAX + 1))

INERCIAS = [
    1409.64, 1395.06, 1382.52, 1371.76, 1362.58, 1355.35, 1350.96,
    1343.74, 1338.03, 1331.52, 1328.15, 1322.65, 1318.61, 1315.28,
]

SILHOUETTE = [
    0.0141, 0.0185, 0.0224, 0.0153, 0.0170, 0.0187, 0.0168,
    0.0212, 0.0217, 0.0224, 0.0195, 0.0203, 0.0213, 0.0227,
]

GAP = [
    1.7375, 1.7541, 1.7730, 1.7924, 1.8065, 1.8165, 1.8312,
    1.8462, 1.8633, 1.8811, 1.8887, 1.8971, 1.9113, 1.9281,
]

GAP_ERRO = [
    0.0017, 0.0018, 0.0030, 0.0025, 0.0014, 0.0026, 0.0037,
    0.0027, 0.0017, 0.0034, 0.0041, 0.0026, 0.0031, 0.0034,
]


def carregar_dados():
    """Carrega as saídas principais geradas pelo pipeline."""
    topicos = pd.read_csv("dados/saida_5_topicos_clusters.csv")
    kmeans = pd.read_csv("dados/saida_5_kmeans.csv")
    vocabulario = pd.read_csv("dados/vocabulario.csv")
    metricas_extras = pd.read_csv("dados/kmeans_metricas_extras.csv")
    lematizado = pd.read_csv("dados/saida_4_lematizado.csv")

    return {
        "topicos": topicos,
        "kmeans": kmeans,
        "vocabulario": vocabulario,
        "metricas_extras": metricas_extras,
        "lematizado": lematizado,
    }


def preparar_projecao(lematizado, kmeans):
    """Reduz os textos para 2 dimensões para visualizar os clusters."""
    textos_tokens = lematizado["tokens"].apply(json.loads)
    documentos = [" ".join(tokens) for tokens in textos_tokens]

    vectorizer = TfidfVectorizer(
        min_df=5,
        max_df=0.9,
        token_pattern=r"(?u)\b[a-zA-ZÀ-ÿ][a-zA-ZÀ-ÿ]{2,}\b",
    )
    X = vectorizer.fit_transform(documentos)
    svd = TruncatedSVD(n_components=2, random_state=42)
    coords = svd.fit_transform(X)

    projecao = pd.DataFrame({
        "x": coords[:, 0],
        "y": coords[:, 1],
        "cluster": kmeans["cluster"].astype(int),
    })
    return projecao


dados = carregar_dados()
projecao = preparar_projecao(dados["lematizado"], dados["kmeans"])

topicos_df = dados["topicos"]
metricas_extras_df = dados["metricas_extras"]
total_textos = len(dados["kmeans"])
total_vocabulario = len(dados["vocabulario"])
total_bow = 36112
total_tfidf = 7970
n_clusters = len(topicos_df)


def card(valor, label, cor):
    return html.Div(style={
        "backgroundColor": CARD,
        "border": f"2px solid {cor}",
        "borderRadius": "8px",
        "padding": "14px 20px",
        "minWidth": "170px",
        "textAlign": "center",
        "flex": "1",
    }, children=[
        html.Div(str(valor), style={
            "color": cor,
            "fontSize": "28px",
            "fontWeight": "bold",
            "lineHeight": "1.1",
        }),
        html.Div(label, style={
            "color": CINZA,
            "fontSize": "11px",
            "marginTop": "5px",
            "whiteSpace": "pre-line",
        }),
    ])


def badge(texto, cor=VERDE):
    return html.Code(texto, style={
        "display": "block",
        "backgroundColor": "#dafbe1",
        "border": "1px solid #aceebb",
        "borderRadius": "6px",
        "padding": "6px 9px",
        "color": "#116329",
        "fontSize": "12px",
        "minWidth": "150px",
    })


def fluxo_linha(rotulo, texto):
    return html.Div(style={
        "display": "grid",
        "gridTemplateColumns": "170px 1fr",
        "gap": "12px",
        "alignItems": "start",
        "marginBottom": "12px",
    }, children=[
        badge(rotulo),
        html.Span(texto, style={"color": BRANCO, "fontSize": "14px", "lineHeight": "1.45"}),
    ])


def bloco_tfidf_geral():
    return html.Div(style={
        "display": "grid",
        "gridTemplateColumns": "minmax(0, 1fr) minmax(0, 1fr)",
        "gap": "16px",
        "marginTop": "16px",
    }, children=[
        html.Div(style={
            "backgroundColor": CARD,
            "border": f"1px solid {BORDA}",
            "borderRadius": "8px",
            "padding": "16px",
        }, children=[
            html.H3("Como o TF-IDF entra no processo", style={"margin": "0 0 12px 0", "color": BRANCO, "fontSize": "17px"}),
            html.P([
                html.B("O TF-IDF nao define os clusters diretamente. "),
                "Ele prepara os textos para que o K-Means consiga comparar documentos por semelhanca."
            ], style={"color": BRANCO, "fontSize": "14px", "lineHeight": "1.55", "marginTop": 0}),
            html.Div(style={
                "backgroundColor": "#f8fafc",
                "border": f"1px solid {BORDA}",
                "borderLeft": f"5px solid {AZUL}",
                "borderRadius": "8px",
                "padding": "12px 14px",
                "color": BRANCO,
                "fontSize": "14px",
                "lineHeight": "1.55",
            }, children=[
                "Ordem usada: textos -> limpeza -> TF-IDF -> K-Means -> clusters. ",
                "Depois disso, os topicos sao lidos pelos termos com maior peso no centro de cada grupo."
            ]),
        ]),
        html.Div(style={
            "backgroundColor": CARD,
            "border": f"1px solid {BORDA}",
            "borderRadius": "8px",
            "padding": "16px",
        }, children=[
            html.H3("Configuracao usada", style={"margin": "0 0 12px 0", "color": BRANCO, "fontSize": "17px"}),
            fluxo_linha("TF-IDF", html.Span([
                html.B("min_df=5"),
                " remove termos raros; ",
                html.B("max_df=0.9"),
                " remove termos comuns demais."
            ])),
            fluxo_linha("KMeans", html.Span([
                html.B("n_clusters=4"),
                ", ",
                html.B("n_init=20"),
                ", ",
                html.B("max_iter=300"),
                ", ",
                html.B("algorithm=lloyd"),
                ", ",
                html.B("random_state=42"),
                "."
            ])),
            fluxo_linha("Topicos", "Os termos principais sao as palavras com maior peso no centro de cada cluster."),
        ]),
    ])


def nota_tfidf_metrica():
    return html.Div(style={
        "backgroundColor": "#f8fafc",
        "border": f"1px solid {BORDA}",
        "borderLeft": f"4px solid {VERDE}",
        "borderRadius": "8px",
        "padding": "10px 12px",
        "color": BRANCO,
        "fontSize": "13px",
        "lineHeight": "1.45",
        "marginTop": "10px",
    }, children=[
        html.B("Papel do TF-IDF: "),
        "transforma as palavras em pesos numericos. As metricas testam o K-Means usando esses pesos; elas nao criam os clusters sozinhas."
    ])


def fig_distribuicao_clusters():
    fig = go.Figure()
    df = topicos_df.copy()
    df["nome"] = df["cluster"].map(CLUSTER_NOMES)

    fig.add_trace(go.Bar(
        x=[f"C{c}<br>{n}" for c, n in zip(df["cluster"], df["nome"])],
        y=df["quantidade_textos"],
        marker_color=[CLUSTER_CORES[int(c)] for c in df["cluster"]],
        text=df["quantidade_textos"],
        textposition="outside",
        textfont=dict(color=BRANCO, size=12),
        hovertemplate="%{x}<br>%{y} textos<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Distribuição dos textos por cluster", font=dict(size=14, color=BRANCO)),
        paper_bgcolor=BG,
        plot_bgcolor=CARD,
        font=dict(color=BRANCO),
        xaxis=dict(gridcolor=BORDA, linecolor=BORDA, tickfont=dict(color=CINZA)),
        yaxis=dict(gridcolor=BORDA, linecolor=BORDA, tickfont=dict(color=CINZA)),
        height=360,
        margin=dict(t=55, b=35, l=40, r=20),
        showlegend=False,
    )
    return fig


def fig_como_ler_divisao():
    pontos = pd.DataFrame({
        "cluster": [0, 1, 2, 3],
        "nome": [CLUSTER_NOMES[i] for i in range(4)],
        "x": [0.0, 2.4, 0.8, 3.3],
        "y": [1.55, 1.75, 0.55, 0.65],
        "tamanho": [120, 165, 175, 175],
    })

    fig = go.Figure()
    for _, linha in pontos.iterrows():
        cluster = int(linha["cluster"])
        fig.add_trace(go.Scatter(
            x=[linha["x"]],
            y=[linha["y"]],
            mode="markers+text",
            marker=dict(
                size=linha["tamanho"],
                color=CLUSTER_CORES[cluster],
                opacity=0.16,
                line=dict(color=CLUSTER_CORES[cluster], width=4),
            ),
            text=[f"<b>C{cluster}</b><br>{linha['nome']}"],
            textfont=dict(color=BRANCO, size=13),
            hovertemplate=f"C{cluster}<br>{linha['nome']}<extra></extra>",
            showlegend=False,
        ))

    fig.update_layout(
        title=dict(text="Como ler a divisão", font=dict(size=16, color=BRANCO)),
        paper_bgcolor=BG,
        plot_bgcolor=CARD,
        height=360,
        margin=dict(t=55, b=15, l=15, r=15),
        xaxis=dict(visible=False, range=[-0.75, 4.15]),
        yaxis=dict(visible=False, range=[-0.25, 2.6], scaleanchor="x", scaleratio=1),
    )
    return fig


def fig_projecao_clusters():
    fig = go.Figure()

    for cluster_id in sorted(projecao["cluster"].unique()):
        parte = projecao[projecao["cluster"] == cluster_id]
        fig.add_trace(go.Scatter(
            x=parte["x"],
            y=parte["y"],
            mode="markers",
            name=f"C{cluster_id} - {CLUSTER_NOMES[cluster_id]}",
            marker=dict(
                size=7,
                color=CLUSTER_CORES[cluster_id],
                opacity=0.72,
                line=dict(width=0.5, color="#ffffff"),
            ),
            hovertemplate=(
                f"Cluster {cluster_id}<br>"
                "SVD 1: %{x:.3f}<br>"
                "SVD 2: %{y:.3f}<extra></extra>"
            ),
        ))

    fig.update_layout(
        title=dict(text="Visualização 2D dos textos agrupados", font=dict(size=14, color=BRANCO)),
        paper_bgcolor=BG,
        plot_bgcolor=CARD,
        font=dict(color=BRANCO),
        legend=dict(bgcolor=CARD, bordercolor=BORDA, borderwidth=1, font=dict(color=BRANCO, size=11)),
        xaxis=dict(title="Componente SVD 1", gridcolor=BORDA, linecolor=BORDA, tickfont=dict(color=CINZA)),
        yaxis=dict(title="Componente SVD 2", gridcolor=BORDA, linecolor=BORDA, tickfont=dict(color=CINZA)),
        height=420,
        margin=dict(t=55, b=45, l=50, r=20),
    )
    return fig


def fig_top_palavras():
    top = dados["vocabulario"].head(12).sort_values("frequencia_total")
    fig = go.Figure(go.Bar(
        x=top["frequencia_total"],
        y=top["palavra"],
        orientation="h",
        marker_color=AZUL,
        text=top["frequencia_total"],
        textposition="outside",
        textfont=dict(color=BRANCO, size=11),
        hovertemplate="%{y}: %{x} ocorrências<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Palavras mais frequentes após o pré-processamento", font=dict(size=14, color=BRANCO)),
        paper_bgcolor=BG,
        plot_bgcolor=CARD,
        font=dict(color=BRANCO),
        xaxis=dict(gridcolor=BORDA, linecolor=BORDA, tickfont=dict(color=CINZA)),
        yaxis=dict(gridcolor=BORDA, linecolor=BORDA, tickfont=dict(color=CINZA)),
        height=390,
        margin=dict(t=55, b=30, l=80, r=30),
    )
    return fig


def fig_metrica(nome_metrica):
    info = metricas_info()[nome_metrica]
    fig = go.Figure()

    if nome_metrica == "cotovelo":
        y = INERCIAS
        fig.add_trace(go.Scatter(x=KS, y=y, mode="lines+markers", marker_color=AZUL, name="Inércia"))
    elif nome_metrica == "silhouette":
        y = SILHOUETTE
        fig.add_trace(go.Scatter(x=KS, y=y, mode="lines+markers", marker_color=AZUL, name="Silhouette"))
    elif nome_metrica == "gap":
        y = GAP
        fig.add_trace(go.Scatter(
            x=KS,
            y=y,
            mode="lines+markers",
            marker_color=AZUL,
            error_y=dict(type="data", array=GAP_ERRO, visible=True),
            name="Gap",
        ))
    elif nome_metrica == "calinski":
        y = metricas_extras_df["calinski_harabasz"].tolist()
        fig.add_trace(go.Scatter(
            x=metricas_extras_df["k"],
            y=y,
            mode="lines+markers",
            marker_color=AZUL,
            name="Calinski-Harabasz",
        ))
    else:
        y = metricas_extras_df["davies_bouldin"].tolist()
        fig.add_trace(go.Scatter(
            x=metricas_extras_df["k"],
            y=y,
            mode="lines+markers",
            marker_color=VERMELHO,
            name="Davies-Bouldin",
        ))

    k_destaque = info["k"]
    if nome_metrica in {"calinski", "davies"}:
        y_destaque = metricas_extras_df.loc[metricas_extras_df["k"] == k_destaque, info["coluna"]].iloc[0]
    else:
        y_destaque = y[KS.index(k_destaque)]

    fig.add_trace(go.Scatter(
        x=[k_destaque],
        y=[y_destaque],
        mode="markers",
        marker=dict(color="red", size=14, line=dict(color="#ffffff", width=1)),
        name=f"Ponto destacado: k={k_destaque}",
    ))

    k_interpretavel = 4
    if k_destaque != k_interpretavel:
        if nome_metrica in {"calinski", "davies"}:
            y_interpretavel = metricas_extras_df.loc[
                metricas_extras_df["k"] == k_interpretavel, info["coluna"]
            ].iloc[0]
        else:
            y_interpretavel = y[KS.index(k_interpretavel)]

        fig.add_trace(go.Scatter(
            x=[k_interpretavel],
            y=[y_interpretavel],
            mode="markers",
            marker=dict(color="green", size=14, line=dict(color="#ffffff", width=1)),
            name=f"Alternativa interpretável: k={k_interpretavel}",
        ))

    fig.update_layout(
        title=dict(text=info["titulo_grafico"], font=dict(size=14, color=BRANCO)),
        paper_bgcolor=BG,
        plot_bgcolor=CARD,
        font=dict(color=BRANCO),
        legend=dict(bgcolor=CARD, bordercolor=BORDA, borderwidth=1, font=dict(color=BRANCO, size=11)),
        xaxis=dict(title="Número de clusters (k)", gridcolor=BORDA, linecolor=BORDA, tickfont=dict(color=CINZA)),
        yaxis=dict(title=info["eixo_y"], gridcolor=BORDA, linecolor=BORDA, tickfont=dict(color=CINZA)),
        height=430,
        margin=dict(t=55, b=45, l=60, r=25),
    )
    return fig


def metricas_info():
    return {
        "cotovelo": {
            "titulo": "Cotovelo",
            "titulo_grafico": "Método do cotovelo",
            "eixo_y": "Inércia",
            "k": 4,
            "coluna": None,
            "melhor": "Ponto vermelho: k = 4, usado como leitura clara dos macrotemas.",
            "texto": "Mostra quanto os textos ficam próximos dos centros. A curva ajuda a perceber quando aumentar k já traz pouco ganho.",
            "leitura": "Não há uma dobra perfeita, mas k = 4 separa assuntos fáceis de reconhecer sem fragmentar demais a base.",
        },
        "silhouette": {
            "titulo": "Silhouette",
            "titulo_grafico": "Silhouette por número de clusters",
            "eixo_y": "Silhouette score",
            "k": 15,
            "coluna": None,
            "melhor": "Ponto vermelho: k = 15, maior score no intervalo testado.",
            "texto": "Compara se cada texto está mais perto do próprio grupo do que dos outros grupos.",
            "leitura": "O maior valor aparece em k = 15, mas a diferença para k = 4 é pequena. O ponto verde marca k = 4 como alternativa mais simples para visão geral.",
        },
        "gap": {
            "titulo": "Gap Statistic",
            "titulo_grafico": "Gap Statistic",
            "eixo_y": "Gap",
            "k": 15,
            "coluna": None,
            "melhor": "Ponto vermelho: k = 15, maior gap observado.",
            "texto": "Compara a estrutura dos textos reais com uma referência aleatória.",
            "leitura": "A métrica favorece uma divisão mais detalhada. O ponto verde marca k = 4 como alternativa mais simples para explicar os macrotemas.",
        },
        "calinski": {
            "titulo": "Calinski-Harabasz",
            "titulo_grafico": "Calinski-Harabasz",
            "eixo_y": "Score",
            "k": 2,
            "coluna": "calinski_harabasz",
            "melhor": "Ponto vermelho: k = 2, maior valor da métrica.",
            "texto": "Valor alto indica grupos compactos por dentro e separados entre si.",
            "leitura": "O k = 2 une temas demais. O ponto verde marca k = 4 como alternativa com mais detalhes e ainda fácil de ler.",
        },
        "davies": {
            "titulo": "Davies-Bouldin",
            "titulo_grafico": "Davies-Bouldin",
            "eixo_y": "Score",
            "k": 14,
            "coluna": "davies_bouldin",
            "melhor": "Ponto vermelho: k = 14, menor valor da métrica.",
            "texto": "Valor baixo indica clusters menos parecidos entre si.",
            "leitura": "O k = 14 cria grupos mais finos. O ponto verde marca k = 4 como alternativa mais direta para contar a história principal.",
        },
    }


def tabela_topicos():
    rows = []
    for _, linha in topicos_df.iterrows():
        cluster = int(linha["cluster"])
        rows.append({
            "Cluster": f"C{cluster}",
            "Interpretação": CLUSTER_NOMES[cluster],
            "Textos": int(linha["quantidade_textos"]),
            "Termos principais": linha["termos_principais"],
        })

    return dash_table.DataTable(
        data=rows,
        columns=[{"name": c, "id": c} for c in rows[0].keys()],
        style_table={"overflowX": "auto"},
        style_cell={
            "backgroundColor": CARD,
            "color": BRANCO,
            "border": f"1px solid {BORDA}",
            "textAlign": "left",
            "padding": "9px 12px",
            "fontFamily": "Arial, sans-serif",
            "fontSize": "13px",
            "whiteSpace": "normal",
            "height": "auto",
        },
        style_header={
            "backgroundColor": HEADER,
            "color": "#ffffff",
            "fontWeight": "bold",
            "border": f"1px solid {BORDA}",
            "textAlign": "left",
        },
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": "#f8fafc"},
            {"if": {"filter_query": '{Cluster} = "C0"', "column_id": "Cluster"}, "color": ROXO, "fontWeight": "bold"},
            {"if": {"filter_query": '{Cluster} = "C1"', "column_id": "Cluster"}, "color": AZUL, "fontWeight": "bold"},
            {"if": {"filter_query": '{Cluster} = "C2"', "column_id": "Cluster"}, "color": VERDE, "fontWeight": "bold"},
            {"if": {"filter_query": '{Cluster} = "C3"', "column_id": "Cluster"}, "color": VERMELHO, "fontWeight": "bold"},
        ],
        sort_action="native",
    )


def cards_clusters():
    cards = []
    for _, linha in topicos_df.iterrows():
        cluster = int(linha["cluster"])
        termos = [t.strip() for t in linha["termos_principais"].split(",")[:8]]
        cards.append(html.Div(style={
            "backgroundColor": CARD,
            "border": f"2px solid {CLUSTER_CORES[cluster]}",
            "borderRadius": "8px",
            "padding": "14px",
            "flex": "1",
            "minWidth": "210px",
        }, children=[
            html.Div(f"Cluster {cluster}", style={
                "color": "#ffffff",
                "backgroundColor": CLUSTER_CORES[cluster],
                "display": "inline-block",
                "padding": "4px 9px",
                "borderRadius": "999px",
                "fontSize": "12px",
                "fontWeight": "bold",
                "marginBottom": "10px",
            }),
            html.H3(CLUSTER_NOMES[cluster], style={"margin": "0 0 6px 0", "fontSize": "16px", "color": BRANCO}),
            html.Div(str(int(linha["quantidade_textos"])), style={
                "color": CLUSTER_CORES[cluster],
                "fontSize": "28px",
                "fontWeight": "bold",
                "lineHeight": "1.1",
            }),
            html.Div("textos", style={"color": CINZA, "fontSize": "12px", "marginBottom": "10px"}),
            html.Div([
                html.Span(termo, style={
                    "border": f"1px solid {BORDA}",
                    "borderRadius": "999px",
                    "padding": "4px 8px",
                    "margin": "3px",
                    "display": "inline-block",
                    "fontSize": "12px",
                    "color": BRANCO,
                    "backgroundColor": "#f8fafc",
                }) for termo in termos
            ]),
        ]))
    return cards


def aba_visao_geral():
    return html.Div(style={"padding": "20px 0"}, children=[
        html.Div(style={"display": "flex", "gap": "16px", "marginBottom": "20px", "flexWrap": "wrap"}, children=[
            card(total_textos, "Textos analisados", VERDE),
            card(total_bow, "Termos no BoW filtrado", AZUL),
            card(total_tfidf, "Termos usados no TF-IDF", AMARELO),
            card(n_clusters, "Clusters escolhidos", ROXO),
        ]),
        html.Div(style={
            "backgroundColor": "#ddf4ff",
            "border": f"1px solid {BORDA}",
            "borderLeft": f"5px solid {AZUL}",
            "borderRadius": "8px",
            "padding": "12px 16px",
            "marginBottom": "16px",
            "color": BRANCO,
        }, children=[
            html.B("Leitura principal: "),
            "k = 4 separa os textos em macrotemas fáceis de explicar: blog/astronomia, esporte, gastronomia e ciência.",
        ]),
        html.Div(style={"marginBottom": "16px"}, children=[
            dcc.Graph(figure=fig_como_ler_divisao(), config={"displayModeBar": False}),
        ]),
        html.Div(style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}, children=[
            dcc.Graph(figure=fig_distribuicao_clusters(), style={"flex": "1", "minWidth": "360px"}, config={"displayModeBar": False}),
            dcc.Graph(figure=fig_projecao_clusters(), style={"flex": "1", "minWidth": "360px"}, config={"displayModeBar": False}),
        ]),
        bloco_tfidf_geral(),
        html.Div(style={"marginTop": "16px"}, children=[
            dcc.Graph(figure=fig_top_palavras(), config={"displayModeBar": False}),
        ]),
    ])


def aba_metricas():
    return html.Div(style={"padding": "20px 0"}, children=[
        html.Div(style={"marginBottom": "14px"}, children=[
            dcc.RadioItems(
                id="metrica",
                options=[
                    {"label": " Cotovelo ", "value": "cotovelo"},
                    {"label": " Silhouette ", "value": "silhouette"},
                    {"label": " Gap Statistic ", "value": "gap"},
                    {"label": " Calinski-Harabasz ", "value": "calinski"},
                    {"label": " Davies-Bouldin ", "value": "davies"},
                ],
                value="cotovelo",
                inline=True,
                style={"color": BRANCO, "fontSize": "14px"},
                inputStyle={"marginRight": "6px", "marginLeft": "14px", "cursor": "pointer"},
                labelStyle={
                    "cursor": "pointer",
                    "padding": "8px 12px",
                    "border": f"1px solid {BORDA}",
                    "borderRadius": "6px",
                    "backgroundColor": CARD,
                    "marginBottom": "8px",
                },
            ),
        ]),
        html.Div(style={"display": "flex", "gap": "16px", "alignItems": "stretch", "flexWrap": "wrap"}, children=[
            html.Div(id="metrica-texto", style={
                "backgroundColor": CARD,
                "border": f"1px solid {BORDA}",
                "borderRadius": "8px",
                "padding": "16px",
                "width": "330px",
                "minWidth": "300px",
            }),
            dcc.Graph(id="grafico-metrica", style={"flex": "1", "minWidth": "420px"}, config={"displayModeBar": False}),
        ]),
    ])


def aba_topicos():
    return html.Div(style={"padding": "20px 0"}, children=[
        html.Div(style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "18px"}, children=cards_clusters()),
        html.H3("Resumo dos tópicos", style={"color": BRANCO, "fontSize": "15px", "marginBottom": "10px"}),
        tabela_topicos(),
        html.H3("Fluxo do processamento", style={"color": BRANCO, "fontSize": "15px", "margin": "22px 0 10px 0"}),
        html.Div(style={"display": "grid", "gridTemplateColumns": "repeat(4, minmax(180px, 1fr))", "gap": "10px"}, children=[
            etapa("1", "Limpeza", "Remove pontuação e padroniza o texto."),
            etapa("2", "Tokenização", "Quebra cada texto em palavras."),
            etapa("3", "Stopwords", "Remove termos muito comuns e pouco informativos."),
            etapa("4", "Lematização", "Reduz palavras para formas-base."),
            etapa("5", "BoW", "Cria a matriz de contagem de palavras."),
            etapa("6", "TF-IDF", "Valoriza termos mais característicos."),
            etapa("7", "K-means", "Agrupa textos por semelhança."),
            etapa("8", "Tópicos", "Lê os termos centrais de cada grupo."),
        ]),
    ])


def etapa(numero, titulo, texto):
    return html.Div(style={
        "backgroundColor": "#f8fafc",
        "border": f"1px solid {BORDA}",
        "borderRadius": "8px",
        "padding": "12px",
    }, children=[
        html.Div(numero, style={
            "backgroundColor": HEADER,
            "color": "#ffffff",
            "width": "28px",
            "height": "28px",
            "borderRadius": "50%",
            "display": "inline-flex",
            "alignItems": "center",
            "justifyContent": "center",
            "fontSize": "13px",
            "fontWeight": "bold",
            "marginBottom": "8px",
        }),
        html.H4(titulo, style={"margin": "0 0 5px 0", "fontSize": "14px", "color": BRANCO}),
        html.P(texto, style={"margin": 0, "fontSize": "12px", "color": CINZA}),
    ])


app = dash.Dash(__name__, title="Dashboard K-means - Textos")

TAB_STYLE = {
    "backgroundColor": BG,
    "color": CINZA,
    "border": f"1px solid {BORDA}",
    "borderRadius": "6px 6px 0 0",
    "padding": "10px 22px",
    "fontWeight": "600",
    "fontSize": "13px",
}

TAB_SEL = {
    **TAB_STYLE,
    "backgroundColor": CARD,
    "color": BRANCO,
    "borderBottom": f"3px solid {AZUL}",
}

app.layout = html.Div(style={
    "backgroundColor": BG,
    "minHeight": "100vh",
    "fontFamily": "Arial, sans-serif",
    "padding": "20px",
}, children=[
    html.Div(style={
        "backgroundColor": HEADER,
        "borderRadius": "8px",
        "padding": "16px 20px",
        "marginBottom": "8px",
    }, children=[
        html.H1("Dashboard - Agrupamento de Textos com K-means", style={
            "color": "#ffffff",
            "margin": "0",
            "fontSize": "22px",
            "fontWeight": "bold",
        }),
        html.P("BoW | TF-IDF | Métricas para escolha de k | Tópicos por cluster", style={
            "color": "#a8c4e0",
            "margin": "4px 0 0 0",
            "fontSize": "12px",
        }),
    ]),
    dcc.Tabs(
        value="visao",
        colors={"border": BORDA, "primary": AZUL, "background": BG},
        children=[
            dcc.Tab(label="Visão Geral", value="visao", style=TAB_STYLE, selected_style=TAB_SEL, children=[aba_visao_geral()]),
            dcc.Tab(label="Métricas do K", value="metricas", style=TAB_STYLE, selected_style=TAB_SEL, children=[aba_metricas()]),
            dcc.Tab(label="Tópicos e Pipeline", value="topicos", style=TAB_STYLE, selected_style=TAB_SEL, children=[aba_topicos()]),
        ],
    ),
])


@app.callback(
    Output("grafico-metrica", "figure"),
    Output("metrica-texto", "children"),
    Input("metrica", "value"),
)
def atualizar_metrica(metrica):
    info = metricas_info()[metrica]
    bloco = html.Div(children=[
        html.H3(info["titulo"], style={"marginTop": "0", "color": BRANCO}),
        html.P(info["texto"], style={"color": CINZA, "fontSize": "13px"}),
        html.Div(info["melhor"], style={
            "backgroundColor": "#fff5f5",
            "border": "1px solid #ffc9c9",
            "borderRadius": "8px",
            "padding": "10px 12px",
            "color": "#82071e",
            "fontWeight": "bold",
            "fontSize": "13px",
            "marginBottom": "10px",
        }),
        html.Div(info["leitura"], style={
            "backgroundColor": "#f8fafc",
            "border": f"1px solid {BORDA}",
            "borderRadius": "8px",
            "padding": "10px 12px",
            "color": BRANCO,
            "fontSize": "13px",
        }),
        nota_tfidf_metrica(),
    ])
    return fig_metrica(metrica), bloco


if __name__ == "__main__":
    print("=" * 60)
    print("  Acesse no navegador: http://127.0.0.1:8051")
    print("  Para parar: Ctrl + C")
    print("=" * 60)
    app.run(debug=False, host="127.0.0.1", port=8051)
