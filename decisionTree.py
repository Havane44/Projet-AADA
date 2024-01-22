# ---------------------------------------------------------
# Ce fichier sert à construire et entraîner un arbre de 
# décision.
# ---------------------------------------------------------

from sklearn import tree
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import export_graphviz
from time import time
from progress.bar import Bar
from functions import evaluation, confusion_matrix_csv, confusion_matrix_png

import pandas as pd
# import numpy as np
import matplotlib.pyplot as plt
import graphviz
import seaborn as sns

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
# Construction d'un arbre de décision
# ---------------------------------------------------------

print("Entraînement de l'arbre de décision en cours...")
start = time()

# Construction et entraînement d'un arbre de décision
# decisionTree = tree.DecisionTreeClassifier(max_depth=20, min_samples_split=2, min_samples_leaf=1)
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
# plt.savefig("tree.png")

# dot_data = tree.export_graphviz(decisionTree, out_file=None,
#                            feature_names=['Accel X', 'Accel Y', 'Accel Z', 'Gyro X', 'Gyro Y', 'Gyro Z'],
#                            class_names=[str(i) for i in range(1, 28)],
#                            filled=True, rounded=True, special_characters=True)

# # Convertir le fichier DOT en image
# graph = graphviz.Source(dot_data)
# graph.render("decision_tree", format="png", cleanup=True)

predictions = []
bar = Bar("Prédictions sur l'ensemble de validation en cours...", max=len(testing_dataset))
for entry in testing_dataset:
    predictions.append(decisionTree.predict([entry]))
    bar.next()
bar.finish()

print("")
print("Evaluation du modèle : ")
print("")
evaluation(testing_labels, predictions)

confusion_matrix_csv(testing_labels, predictions)
confusion_matrix_png(testing_labels, predictions)