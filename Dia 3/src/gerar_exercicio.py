import math
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

pre_processado = {
    1: ["aluno", "estudar", "matemática", "universidade"],
    2: ["aluno", "estudar", "matemática", "escola"],
    3: ["professor", "ensinar", "matemática", "universidade"],
    4: ["professor", "ensinar", "física", "universidade"],
    5: ["cachorro", "correr", "rapidamente", "parque"],
}

vocabulario = sorted(set(w for tokens in pre_processado.values() for w in tokens))

bow = {}
for i in range(1, 6):
    bow[i] = [pre_processado[i].count(w) for w in vocabulario]

palavras_alvo = ["matemática", "universidade", "física", "cachorro"]
N = 5
df = {}
idf = {}
for w in palavras_alvo:
    idx = vocabulario.index(w)
    d = sum(1 for i in range(1, 6) if bow[i][idx] > 0)
    df[w] = d
    idf[w] = math.log(N / d)

fig, axes = plt.subplots(4, 1, figsize=(13, 18), height_ratios=[1.1, 1.4, 1.4, 1.6])
fig.suptitle("Exercício (a mão) — Pré-processamento, BoW e TF-IDF", fontsize=16, fontweight="bold")

# --- Tabela 1: pre-processamento ---
ax0 = axes[0]
ax0.axis("off")
ax0.set_title("Tokens finais (após pré-processamento)", fontsize=12, fontweight="bold", loc="left")
rows1 = [[f"Texto {i}", ", ".join(pre_processado[i])] for i in range(1, 6)]
t1 = ax0.table(cellText=rows1, colLabels=["Documento", "Tokens finais"],
               cellLoc="left", loc="center", colWidths=[0.15, 0.8])
t1.auto_set_font_size(False)
t1.set_fontsize(10)
t1.scale(1, 1.6)
for (r, c), cell in t1.get_celld().items():
    if r == 0:
        cell.set_facecolor("#D9E1F2")
        cell.set_text_props(fontweight="bold")

# --- Tabela 2: BoW ---
ax1 = axes[1]
ax1.axis("off")
ax1.set_title("Matriz Bag of Words  |  Vocabulário: " + ", ".join(vocabulario),
               fontsize=12, fontweight="bold", loc="left")
header2 = ["Documento"] + vocabulario
rows2 = [[f"Texto {i}"] + bow[i] for i in range(1, 6)]
t2 = ax1.table(cellText=rows2, colLabels=header2, cellLoc="center", loc="center")
t2.auto_set_font_size(False)
t2.set_fontsize(9)
t2.scale(1, 1.6)
for (r, c), cell in t2.get_celld().items():
    if r == 0:
        cell.set_facecolor("#D9E1F2")
        cell.set_text_props(fontweight="bold")
    if c == 0:
        cell.set_text_props(fontweight="bold")

# --- Tabela 3: TF, DF, IDF, TF-IDF ---
ax2 = axes[2]
ax2.axis("off")
ax2.set_title("TF, DF, IDF e TF-IDF (matemática, universidade, física, cachorro)",
               fontsize=12, fontweight="bold", loc="left")
header3 = ["Documento"]
for w in palavras_alvo:
    header3 += [f"TF({w})", f"TF-IDF({w})"]
rows3 = []
for i in range(1, 6):
    line = [f"Texto {i}"]
    for w in palavras_alvo:
        idx = vocabulario.index(w)
        tf = bow[i][idx]
        tfidf = round(tf * idf[w], 4)
        line += [tf, tfidf]
    rows3.append(line)
# linha de DF/IDF no topo
rows3_full = [["DF / IDF"] + [v for w in palavras_alvo for v in (df[w], round(idf[w], 4))]] + rows3
t3 = ax2.table(cellText=rows3_full, colLabels=header3, cellLoc="center", loc="center")
t3.auto_set_font_size(False)
t3.set_fontsize(9)
t3.scale(1, 1.6)
for (r, c), cell in t3.get_celld().items():
    if r == 0:
        cell.set_facecolor("#D9E1F2")
        cell.set_text_props(fontweight="bold")
    if r == 1:
        cell.set_facecolor("#FCE4D6")
        cell.set_text_props(fontweight="bold")
    if c == 0:
        cell.set_text_props(fontweight="bold")

# --- Memória de cálculo ---
ax3 = axes[3]
ax3.axis("off")
ax3.set_title(f"Memória de cálculo (N = {N} documentos, IDF = ln(N/DF))",
              fontsize=12, fontweight="bold", loc="left")

linhas_memoria = []
for w in palavras_alvo:
    docs_com_palavra = [i for i in range(1, 6) if bow[i][vocabulario.index(w)] > 0]
    docs_str = ", ".join(f"T{i}" for i in docs_com_palavra)
    linhas_memoria.append(f"{w.upper()}")
    linhas_memoria.append(f"  DF({w}) = {df[w]}  (aparece em: {docs_str})")
    linhas_memoria.append(f"  IDF({w}) = ln({N}/{df[w]}) = ln({N/df[w]:.4f}) = {idf[w]:.4f}")
    partes_tfidf = []
    for i in range(1, 6):
        idx = vocabulario.index(w)
        tf = bow[i][idx]
        tfidf = tf * idf[w]
        partes_tfidf.append(f"T{i}: {tf}×{idf[w]:.4f}={tfidf:.4f}")
    linhas_memoria.append("  TF-IDF = TF × IDF  ->  " + " | ".join(partes_tfidf))
    linhas_memoria.append("")

texto_memoria = "\n".join(linhas_memoria)
ax3.text(0, 1, texto_memoria, ha="left", va="top", fontsize=10, family="monospace",
         transform=ax3.transAxes)

plt.tight_layout(rect=[0, 0, 1, 0.97])
out_path = os.path.join(os.path.dirname(__file__), "..", "exercicio_tfidf.png")
plt.savefig(out_path, dpi=160, bbox_inches="tight")
print("saved", out_path)
