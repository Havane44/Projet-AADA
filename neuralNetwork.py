# ---------------------------------------------------------
# Ce fichier sert à construire et entraîner un réseau de
# neurones avec la bibliothèque TensorFlow.
# ---------------------------------------------------------

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time
from progress.bar import Bar

model_is_trained = True

# ---------------------------------------------------------
# Importation des données
# ---------------------------------------------------------

print("Importation des données en cours...")
start = time()

training_dataset = pd.read_csv('training_dataset.csv').to_numpy()[:, 1:7]
training_labels = pd.read_csv('training_labels.csv').to_numpy()[:, 1]
testing_dataset = pd.read_csv('testing_dataset.csv').to_numpy()[:, 1:7]
testing_labels = pd.read_csv('testing_labels.csv').to_numpy()[:, 1]

end = time()
print("Importation des données terminée en", end-start, "secondes")

# ---------------------------------------------------------
# Conditionnement des labels sous la forme de one-hots
# ex : [3] devient [0, 0, 1, 0, ..., 0]
# ---------------------------------------------------------

training_labels_onehot = []
testing_labels_onehot = []

for label in training_labels:
  temp = [0]*27
  temp[label-1] = 1
  training_labels_onehot.append(temp)

for label in testing_labels:
  temp = [0]*27
  temp[label-1] = 1
  testing_labels_onehot.append(temp)

training_labels_onehot = np.array(training_labels_onehot)
testing_labels_onehot = np.array(testing_labels_onehot)

if not model_is_trained:
  # ---------------------------------------------------------
  # Entraînement et validation du modèle
  # ---------------------------------------------------------

  hiddenLayer1 = 512
  hiddenLayer2 = 1024

  model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(hiddenLayer1, input_shape=(6,), activation='relu'),
    tf.keras.layers.Dense(hiddenLayer2, activation='relu'),
    tf.keras.layers.Dense(27, activation='softmax')
  ])

  model.summary()

  model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

  history = model.fit(training_dataset, training_labels_onehot, epochs=50, batch_size=100)

  model.save("./models/neuralNetwork_{0}_{1}.keras".format(hiddenLayer1, hiddenLayer2))

else:
  size1 = int(input("Nombre de neurones dans la couche 1 : "))
  size2 = int(input("Nombre de neurones dans la couche 2 : "))

  model = tf.keras.models.load_model("./models/neuralNetwork_{0}_{1}.keras".format(size1, size2))
  predictions = model.predict(testing_dataset)

  # result = {
  #   "Array de la prédiction": [],
  #   "Prédiction": [],
  #   "Classe réelle": []
  # }
  # with Bar('Enregistrement des prédictions...', max=len(predictions)) as bar:
  #   for i in range(len(predictions)):
  #     result["Array de la prédiction"].append(predictions[i])
  #     result["Prédiction"].append(np.argmax(predictions[i]) + 1)
  #     result["Classe réelle"].append(testing_labels[i])
  #     bar.next()
  # result = pd.DataFrame(result)
  # result.to_csv("./prédictions/predictions_{0}_{1}.csv".format(size1, size2))

  errors = 0
  with Bar('Comptage des erreurs de prédiction...', max=len(predictions)) as bar:
    for i in range(len(predictions)):
      if np.argmax(predictions[i]) != np.argmax(testing_labels_onehot[i]):
        errors += 1
      bar.next()

  print("Nombre d'erreurs : " + str(errors) + " sur " + str(len(predictions)) + " prédictions")
  print("Pourcentage d'erreurs : " + str(int(errors/len(predictions) * 100)) + "%")

  errors = 0
  with Bar("Comptage des erreurs de prédiction dans l'ensemble d'apprentissage...", max=len(predictions)) as bar:
    for i in range(len(training_labels_onehot)):
      if np.argmax(predictions[i]) != np.argmax(training_labels_onehot[i]):
        errors += 1
      bar.next()

  print("Nombre d'erreurs : " + str(errors) + " sur " + str(len(predictions)) + " prédictions")
  print("Pourcentage d'erreurs : " + str(int(errors/len(predictions) * 100)) + "%")