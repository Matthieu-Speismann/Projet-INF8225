import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation # type: ignore
from tensorflow.keras.layers import LeakyReLU, PReLU # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
import matplotlib.pyplot as plt

# --- 1. Charger les données MNIST ---
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype("float32") / 255.
x_test = x_test.astype("float32") / 255.

# --- 2. Paramètres généraux ---
activation_options = ["relu" , "leaky_relu" , "prelu", "elu", "swish"]
num_epochs = 50
batch_size = 128

# --- 3. Fonction pour créer le modèle avec activation choisie ---
def create_model(activation_choice):
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28)))

    if activation_choice == "leaky_relu":
        model.add(Dense(256))
        model.add(LeakyReLU(negative_slope=0.1))
    elif activation_choice == "prelu":
        model.add(Dense(256))
        model.add(PReLU())
    else:
        model.add(Dense(256, activation=activation_choice))

    model.add(Dropout(0.3))

    if activation_choice == "leaky_relu":
        model.add(Dense(128))
        model.add(LeakyReLU(negative_slope=0.1))
    elif activation_choice == "prelu":
        model.add(Dense(128))
        model.add(PReLU())
    else:
        model.add(Dense(128, activation=activation_choice))

    model.add(Dropout(0.3))
    model.add(Dense(10, activation='softmax'))
    return model

# --- 4. Entraîner et évaluer chaque modèle ---
results = {}

for act in activation_options:
    print(f"\n>>> Entraînement avec activation : {act}")
    model = create_model(act)
    model.compile(optimizer=Adam(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                        validation_split=0.2,
                        epochs=num_epochs,
                        batch_size=batch_size,
                        verbose=0)

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    results[act] = {
        "accuracy": test_acc,
        "history": history.history
    }
    print(f"Test accuracy : {test_acc:.4f}")

# --- 5. Affichage des résultats ---
plt.figure()
for act in activation_options:
    plt.plot(results[act]["history"]["accuracy"], label=act)
plt.title("Validation Accuracy selon la fonction d'activation")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()
