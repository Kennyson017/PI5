import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

caminho_imagem = os.path.join(os.path.dirname(__file__), "..", "exercicio_tfidf.png")
imagem = mpimg.imread(caminho_imagem)

plt.imshow(imagem)
plt.axis("off")
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

mng = plt.get_current_fig_manager()
try:
    mng.window.state("zoomed")  # TkAgg (Windows)
except AttributeError:
    try:
        mng.window.showMaximized()  # Qt
    except AttributeError:
        mng.full_screen_toggle()  # fallback

plt.show()
