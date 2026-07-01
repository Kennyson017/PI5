import gradio as gr
import matplotlib.pyplot as plt
import numpy as np

# Dados simulados representando a saída real do seu pipeline scikit-learn.
# Matriz: [[Verdadeiro Negativo, Falso Positivo], [Falso Negativo, Verdadeiro Positivo]]
# Foco estratégico: Expor a anomalia térmica (Minimizar o Falso Negativo no índice [1][0])
resultados_modelos = {
    "KNN": {"cm": [[40, 10], [15, 35]], "acc": "75%", "recall": "70%"},
    "Árvore de Decisão": {"cm": [[45, 5], [10, 40]], "acc": "85%", "recall": "80%"},
    "Random Forest": {"cm": [[48, 2], [1, 49]], "acc": "97%", "recall": "98%"}, # O melhor para risco industrial
    "SVM Linear": {"cm": [[50, 0], [12, 38]], "acc": "88%", "recall": "76%"},  # Perigoso: Deixa motores quebrarem
    "SVM Polinomial": {"cm": [[47, 3], [8, 42]], "acc": "89%", "recall": "84%"},
    "Ensemble (Maioria)": {"cm": [[49, 1], [2, 48]], "acc": "97%", "recall": "96%"}
}

def analisar_modelo(nome_modelo):
    """Processa a matriz de confusão e calcula o risco financeiro/operacional."""
    dados = resultados_modelos[nome_modelo]
    cm = np.array(dados["cm"])
    
    # Renderização da Matriz de Confusão em tempo real
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(cm, cmap=plt.cm.Reds) # Paleta de alerta térmico
    plt.title(f"Matriz Analítica: {nome_modelo}\n", pad=20, fontweight='bold')
    fig.colorbar(cax)
    
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Predito:\nNormal', 'Predito:\nSobrecarga'])
    ax.set_yticklabels(['Real:\nNormal', 'Real:\nSobrecarga'], rotation=90, va='center')
    
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i][j]), va='center', ha='center', 
                    color='white' if cm[i][j] > (cm.max()/2) else 'black', 
                    fontsize=16, fontweight='bold')
            
    # Lógica de negócio dura: Falso Negativo = Motor estourado na fábrica
    falsos_negativos = cm[1][0]
    
    if falsos_negativos > 5:
        risco_critico = f"⚠️ INACEITÁVEL: {falsos_negativos} falhas ocultas. O modelo ignora motores prestes a fundir."
    elif falsos_negativos > 0:
        risco_critico = f"⚠️ ALERTA: {falsos_negativos} motores com anomalia térmica passaram despercebidos."
    else:
        risco_critico = "✅ SEGURO: Zero motores em risco ignorados pelo modelo."
        
    return fig, dados["acc"], dados["recall"], risco_critico

# Arquitetura do Dashboard em Blocos
with gr.Blocks(theme=gr.themes.Default()) as dashboard:
    gr.Markdown("# 🏭 Painel de Risco Preditivo: Motores Industriais")
    gr.Markdown("Pare de olhar para a Acurácia. Avalie o impacto direto de **Falsos Negativos** (motores superaquecidos classificados como normais) na linha de produção.")
    
    with gr.Row():
        with gr.Column(scale=1):
            seletor = gr.Radio(list(resultados_modelos.keys()), label="Selecione o Modelo Treinado", value="Random Forest")
            acuracia_txt = gr.Textbox(label="Acurácia Geral (Métrica de Vaidade)")
            recall_txt = gr.Textbox(label="Recall da Classe Sobrecarga (O que importa)")
            risco_txt = gr.Textbox(label="Diagnóstico de Risco Operacional")
        
        with gr.Column(scale=2):
            grafico_matriz = gr.Plot(label="Mapa de Classificação")
            
    # Lógica de reatividade: ao mudar o Radio, atualiza as saídas
    seletor.change(fn=analisar_modelo, inputs=seletor, outputs=[grafico_matriz, acuracia_txt, recall_txt, risco_txt])
    
    # Carregamento inicial do estado
    dashboard.load(fn=analisar_modelo, inputs=seletor, outputs=[grafico_matriz, acuracia_txt, recall_txt, risco_txt])

# O inline=True faz a mágica de injetar o web app direto na saída da sua célula no Jupyter.
dashboard.launch(inline=True, show_error=True)