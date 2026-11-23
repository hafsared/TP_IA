import os

os.environ["TCL_LIBRARY"] = r"C:\Users\HP\AppData\Local\Programs\Python\Python313\tcl\tcl8.6"
os.environ["TK_LIBRARY"] = r"C:\Users\HP\AppData\Local\Programs\Python\Python313\tcl\tk8.6"

import tkinter as tk


import tkinter as tk
import numpy as np
from PIL import Image, ImageDraw
from tensorflow.keras.models import load_model

# =====================================
# 1. CHARGEMENT DU MODELE
# =====================================

model = load_model("modele_emnist.keras")

print("Modèle chargé avec succès !")

# =====================================
# 2. FENETRE PRINCIPALE
# =====================================

root = tk.Tk()
root.title("Reconnaissance des lettres A-Z")
root.geometry("600x550")
root.resizable(False, False)

# Image utilisée pour préparer le dessin
image = Image.new("L", (280, 280), 0)
draw = ImageDraw.Draw(image)

# =====================================
# 3. TITRE
# =====================================

titre = tk.Label(
    root,
    text="Reconnaissance des lettres",
    font=("Arial", 20, "bold")
)
titre.pack(pady=15)

# =====================================
# 4. ZONE DE DESSIN
# =====================================

canvas = tk.Canvas(
    root,
    width=280,
    height=280,
    bg="black",
    cursor="cross"
)
canvas.pack(pady=10)

def dessiner(event):
    x, y = event.x, event.y

    rayon = 8

    canvas.create_oval(
        x-rayon, y-rayon,
        x+rayon, y+rayon,
        fill="white",
        outline="white"
    )

    draw.ellipse(
        (x-rayon, y-rayon, x+rayon, y+rayon),
        fill=255
    )

canvas.bind("<B1-Motion>", dessiner)

# =====================================
# 5. PREDICTION
# =====================================

resultat = tk.Label(
    root,
    text="Dessine une lettre !",
    font=("Arial", 18, "bold")
)
resultat.pack(pady=15)

def predire():

    # Redimensionner en 28x28
    img = image.resize((28, 28))

    # Convertir en tableau
    pixels = np.array(img).astype("float32")

    # Normalisation
    pixels = pixels / 255.0

    # Aplatir en 784 valeurs
    pixels = pixels.reshape(1, 784)

    # Prédiction
    predictions = model.predict(pixels, verbose=0)

    indice = np.argmax(predictions[0])

    lettre = chr(65 + indice)

    confiance = predictions[0][indice] * 100

    resultat.config(
        text=f"Lettre : {lettre}\nConfiance : {confiance:.2f}%"
    )

# =====================================
# 6. EFFACER
# =====================================

def effacer():

    canvas.delete("all")

    draw.rectangle(
        (0, 0, 280, 280),
        fill=0
    )

    resultat.config(text="Dessine une lettre !")

# =====================================
# 7. BOUTONS
# =====================================

boutons = tk.Frame(root)
boutons.pack(pady=10)

btn_predire = tk.Button(
    boutons,
    text="Prédire",
    font=("Arial", 13, "bold"),
    bg="#4CAF50",
    fg="white",
    width=12,
    command=predire
)
btn_predire.pack(side="left", padx=10)

btn_effacer = tk.Button(
    boutons,
    text="Effacer",
    font=("Arial", 13, "bold"),
    bg="#f44336",
    fg="white",
    width=12,
    command=effacer
)
btn_effacer.pack(side="left", padx=10)

# =====================================
# 8. LANCEMENT
# =====================================

root.mainloop()
