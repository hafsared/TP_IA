
import numpy as np
from PIL import Image, ImageDraw

# Charger les données
data = np.load("uppercase-letters_train.npz")

images = data["x_train"]
labels = data["y_train"]

# Créer une image blanche pour afficher les 10 lettres
largeur = 500
hauteur = 220

resultat = Image.new("RGB", (largeur, hauteur), "white")
dessin = ImageDraw.Draw(resultat)

for i in range(10):
    # Récupérer l'image
    lettre = images[i]

    # Convertir en image Pillow
    image = Image.fromarray(lettre.astype("uint8")).convert("L")

    # Agrandir l'image
    image = image.resize((100, 100))

    # Convertir en RGB
    image = image.convert("RGB")

    # Positionner l'image
    x = (i % 5) * 100
    y = (i // 5) * 110

    resultat.paste(image, (x, y))

    # Ajouter le nom de la lettre
    nom = chr(65 + int(labels[i]))
    dessin.text((x + 40, y + 100), nom, fill="black")

# Enregistrer
resultat.save("apercu_lettres.png")

print("Image enregistrée avec succès !")
print("Valeur minimale :", images.min())
print("Valeur maximale :", images.max())
