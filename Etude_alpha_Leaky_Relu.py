import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, LeakyReLU
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# --- 1. Chargement des données MNIST ---
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype("float32") / 255.
x_test = x_test.astype("float32") / 255.

# --- 2. Liste des valeurs de alpha à tester pour LeakyReLU ---
alpha_values = [0.0, 0.01, 0.1, 0.2, 0.5, 0.8, 1.0]  # 0.0 équivaut à ReLU, 1.0 à identité linéaire
num_epochs = 20
batch_size = 128

# --- 3. Fonction pour créer un modèle avec LeakyReLU(alpha) ---
def create_leakyrelu_model(alpha):
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28)))

    model.add(Dense(256))
    model.add(LeakyReLU(alpha=alpha))

    model.add(Dropout(0.3))

    model.add(Dense(128))
    model.add(LeakyReLU(alpha=alpha))
    model.add(Dropout(0.3))

    model.add(Dense(10, activation='softmax'))
    return model

# --- 4. Entraînement et évaluation ---
results = {}

for alpha in alpha_values:
    print(f"\n>>> Entraînement avec LeakyReLU alpha = {alpha}")
    model = create_leakyrelu_model(alpha)
    model.compile(optimizer=Adam(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                        validation_split=0.2,
                        epochs=num_epochs,
                        batch_size=batch_size,
                        verbose=0)

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    results[alpha] = {
        "accuracy": test_acc,
        "history": history.history
    }
    print(f"Test accuracy : {test_acc:.4f}")

# --- 5. Affichage des résultats ---
plt.figure()
for alpha in alpha_values:
    plt.plot(results[alpha]["history"]["val_accuracy"], label=f"α={alpha}")
plt.title("Validation Accuracy - LeakyReLU selon α")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()
