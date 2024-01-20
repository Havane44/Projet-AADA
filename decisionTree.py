# ---------------------------------------------------------
# Ce fichier sert à construire et entraîner un arbre de 
# décision.
# ---------------------------------------------------------

from sklearn import tree
from sklearn.metrics import accuracy_score
from time import time
from progress.bar import Bar
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

print(training_dataset[0])
print(training_labels[0])
print(testing_dataset[0])
print(testing_labels[0])

# ---------------------------------------------------------
# Construction d'un arbre de décision
# ---------------------------------------------------------

print("Entraînement de l'arbre de décision en cours...")
start = time()

# Construction et entraînement d'un arbre de décision
decisionTree = tree.DecisionTreeClassifier()
decisionTree = decisionTree.fit(training_dataset,training_labels)

end = time()
print("Entraînement de l'arbre de décision terminé en", end-start, "secondes")

print("Profondeur de l'arbre : ", decisionTree.get_depth())

# tree.plot_tree(decisionTree, 
#                feature_names=['Accel X', 'Accel Y', 'Accel Z', 'Gyro X', 'Gyro Y', 'Gyro Z'], 
#                class_names=[str(i) for i in list(range(1, 28))], 
#                fontsize=12,
#                filled=True)
# tree.plot_tree(decisionTree)
# plt.show()

predictions = []
bar = Bar("Prédictions sur l'ensemble de validation en cours...", max=len(testing_dataset))
for entry in testing_dataset:
    predictions.append(decisionTree.predict([entry]))
    bar.next()
bar.finish()

accuracy = accuracy_score(testing_labels, predictions)
print(f"Précision de l'arbre de décision : {accuracy * 100:.2f}%")