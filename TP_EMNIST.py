
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam, SGD, RMSprop


# ==========================================
# 1. CHARGEMENT ET PREPARATION DES DONNEES
# ==========================================

data_train = np.load("uppercase-letters_train.npz")
data_test = np.load("uppercase-letters_test.npz")

x_train = data_train["x_train"]
y_train = data_train["y_train"]

x_test = data_test["x_test"]
y_test = data_test["y_test"]

print("x_train :", x_train.shape)
print("y_train :", y_train.shape)
print("x_test :", x_test.shape)
print("y_test :", y_test.shape)

print("Valeurs des labels :", np.unique(y_train))
print("Minimum des labels :", y_train.min())
print("Maximum des labels :", y_train.max())


# Aplatir les images
x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
x_test = x_test.reshape(-1, 784).astype("float32") / 255.0

# Transformer les labels en catégories
y_train = to_categorical(y_train, num_classes=26)
y_test = to_categorical(y_test, num_classes=26)


# ==========================================
# 2. MODELE AVEC 50 NEURONES
# ==========================================

print("\n===== TEST 1 : 50 NEURONES =====")

model_50 = Sequential([
    Input(shape=(784,)),
    Dense(50, activation="relu"),
    Dense(26, activation="softmax")
])

model_50.summary()

model_50.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history_50 = model_50.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.1
)

loss_50, accuracy_50 = model_50.evaluate(
    x_test, y_test, verbose=0
)

print("Précision avec 50 neurones :",
      round(accuracy_50 * 100, 2), "%")


# ==========================================
# 3. MODELE AVEC 100 NEURONES
# ==========================================

print("\n===== TEST 2 : 100 NEURONES =====")

model_100 = Sequential([
    Input(shape=(784,)),
    Dense(100, activation="relu"),
    Dense(26, activation="softmax")
])

model_100.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history_100 = model_100.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2
)

loss_100, accuracy_100 = model_100.evaluate(
    x_test, y_test, verbose=0
)

print("Précision avec 100 neurones :",
      round(accuracy_100 * 100, 2), "%")


# ==========================================
# 4. MODELE AVEC DEUX COUCHES CACHEES
# ==========================================

print("\n===== TEST 3 : DEUX COUCHES CACHEES =====")

model_2c = Sequential([
    Input(shape=(784,)),
    Dense(100, activation="relu"),
    Dense(50, activation="relu"),
    Dense(26, activation="softmax")
])

model_2c.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history_2c = model_2c.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2
)

loss_2c, accuracy_2c = model_2c.evaluate(
    x_test, y_test, verbose=0
)

print("Précision avec deux couches cachées :",
      round(accuracy_2c * 100, 2), "%")


# ==========================================
# 5. COMPARAISON DES OPTIMISEURS
# ==========================================

print("\n===== TEST 4 : OPTIMISEURS =====")

optimiseurs = {
    "Adam": Adam(),
    "SGD": SGD(),
    "RMSprop": RMSprop()
}

resultats_optimiseurs = {}

for nom, optimiseur in optimiseurs.items():

    print("\nOptimiseur testé :", nom)

    model_opt = Sequential([
        Input(shape=(784,)),
        Dense(100, activation="relu"),
        Dense(50, activation="relu"),
        Dense(26, activation="softmax")
    ])

    model_opt.compile(
        optimizer=optimiseur,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model_opt.fit(
        x_train,
        y_train,
        epochs=10,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )

    loss_opt, accuracy_opt = model_opt.evaluate(
        x_test, y_test, verbose=0
    )

    resultats_optimiseurs[nom] = accuracy_opt * 100

    print("Précision avec", nom, ":",
          round(accuracy_opt * 100, 2), "%")


# ==========================================
# 6. COMPARAISON DES FONCTIONS D'ACTIVATION
# ==========================================

print("\n===== TEST 5 : FONCTIONS D'ACTIVATION =====")

fonctions_activation = ["relu", "sigmoid", "tanh"]

resultats_activation = {}

for activation in fonctions_activation:

    print("\nFonction d'activation :", activation)

    model_activation = Sequential([
        Input(shape=(784,)),
        Dense(100, activation=activation),
        Dense(50, activation=activation),
        Dense(26, activation="softmax")
    ])

    model_activation.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model_activation.fit(
        x_train,
        y_train,
        epochs=10,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )

    loss_activation, accuracy_activation = model_activation.evaluate(
        x_test, y_test, verbose=0
    )

    resultats_activation[activation] = accuracy_activation * 100

    print("Précision avec", activation, ":",
          round(accuracy_activation * 100, 2), "%")


print("\n===== RESULTATS DES ACTIVATIONS =====")

for activation, precision in resultats_activation.items():
    print(activation, ":", round(precision, 2), "%")

meilleure_activation = max(
    resultats_activation,
    key=resultats_activation.get
)

print("Meilleure activation :", meilleure_activation)


# ==========================================
# 7. QUESTION 4 : DATA AUGMENTATION
# ==========================================

print("\n===== QUESTION 4 : DATA AUGMENTATION =====")

# Recharger les données originales en images 28x28
data_train_aug = np.load("uppercase-letters_train.npz")
data_test_aug = np.load("uppercase-letters_test.npz")

images = data_train_aug["x_train"]
labels = data_train_aug["y_train"]

images_test = data_test_aug["x_test"]
labels_test = data_test_aug["y_test"]


# ------------------------------------------
# CREATION DES IMAGES DECALEES
# ------------------------------------------

# Décalage vers le haut
haut = np.zeros_like(images)
haut[:, :-1, :] = images[:, 1:, :]

# Décalage vers le bas
bas = np.zeros_like(images)
bas[:, 1:, :] = images[:, :-1, :]

# Décalage vers la gauche
gauche = np.zeros_like(images)
gauche[:, :, :-1] = images[:, :, 1:]

# Décalage vers la droite
droite = np.zeros_like(images)
droite[:, :, 1:] = images[:, :, :-1]


# ------------------------------------------
# ENRICHISSEMENT DU DATASET
# ------------------------------------------

x_train_aug = np.concatenate(
    [images, haut, bas, gauche, droite],
    axis=0
)

y_train_aug = np.concatenate(
    [labels, labels, labels, labels, labels],
    axis=0
)

print("Images avant augmentation :", len(images))
print("Images après augmentation :", len(x_train_aug))


# Mélanger les images
indices = np.random.permutation(len(x_train_aug))

x_train_aug = x_train_aug[indices]
y_train_aug = y_train_aug[indices]


# ------------------------------------------
# PREPARATION DES DONNEES AUGMENTEES
# ------------------------------------------

x_train_aug = (
    x_train_aug.reshape(-1, 784).astype("float32") / 255.0
)

x_test_aug = (
    images_test.reshape(-1, 784).astype("float32") / 255.0
)

y_train_aug = to_categorical(
    y_train_aug, num_classes=26
)

y_test_aug = to_categorical(
    labels_test, num_classes=26
)


# ------------------------------------------
# CREATION DU MODELE AUGMENTE
# ------------------------------------------

model_aug = Sequential([
    Input(shape=(784,)),
    Dense(100, activation="relu"),
    Dense(50, activation="relu"),
    Dense(26, activation="softmax")
])

model_aug.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


# Entraînement
history_aug = model_aug.fit(
    x_train_aug,
    y_train_aug,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)


# ------------------------------------------
# EVALUATION DU MODELE AUGMENTE
# ------------------------------------------

loss_aug, accuracy_aug = model_aug.evaluate(
    x_test_aug,
    y_test_aug,
    verbose=1
)

print("\n===== RESULTAT QUESTION 4 =====")

print("Nombre d'images utilisées :", len(x_train_aug))

print("Précision après augmentation :",
      round(accuracy_aug * 100, 2), "%")


# ==========================================
# 8. VISUALISATION DE 10 PREDICTIONS
# ==========================================

print("\n===== VISUALISATION DES PREDICTIONS =====")

# Sélectionner 10 images du jeu de test
images_a_tester = x_test_aug[:10]

# Prédictions avec le modèle augmenté
predictions = model_aug.predict(images_a_tester)

plt.figure(figsize=(12, 5))

for i in range(10):

    vraie_lettre = chr(65 + np.argmax(y_test_aug[i]))
    lettre_predite = chr(65 + np.argmax(predictions[i]))

    plt.subplot(2, 5, i + 1)

    plt.imshow(
        images_a_tester[i].reshape(28, 28),
        cmap="gray"
    )

    plt.title(
        f"Prédiction : {lettre_predite}\nRéelle : {vraie_lettre}"
    )

    plt.axis("off")

plt.tight_layout()
plt.show()


# ==========================================
# 9. SAUVEGARDE DU MEILLEUR MODELE
# ==========================================

model_aug.save("modele_emnist.keras")

print("\n================================")
print("Modèle augmenté sauvegardé !")
print("Fichier : modele_emnist.keras")
print("================================")
