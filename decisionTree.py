# ---------------------------------------------------------
# Ce fichier sert à construire et entraîner un arbre de 
# décision.
# ---------------------------------------------------------

from sklearn import tree
from sklearn.tree import export_graphviz
from time import time
from dataProcessing import evaluation, confusion_matrix_csv, confusion_matrix_png

import pandas as pd
import graphviz

# ---------------------------------------------------------
# Importation des données
# ---------------------------------------------------------
print("Importation des données en cours...")
start = time()

# On importe uniquement les moyennes et écart-types, pas les numéros de sujet etc...

training_dataset = pd.read_csv('./processed_data/training_dataset.csv').to_numpy()[:, 4:16]
training_labels = pd.read_csv('./processed_data/training_labels.csv').to_numpy()[:, 1]
testing_dataset = pd.read_csv('./processed_data/testing_dataset.csv').to_numpy()[:, 4:16]
testing_labels = pd.read_csv('./processed_data/testing_labels.csv').to_numpy()[:, 1]

end = time()
print("Importation des données terminée en", end-start, "secondes")

# ---------------------------------------------------------
# Construction d'un arbre de décision
# ---------------------------------------------------------

print("Entraînement de l'arbre de décision en cours...")
start = time()

decisionTree = tree.DecisionTreeClassifier()
decisionTree = decisionTree.fit(training_dataset,training_labels)

end = time()
print("Entraînement de l'arbre de décision terminé en", end-start, "secondes")

print("Profondeur de l'arbre : ", decisionTree.get_depth())

dot_data = tree.export_graphviz(decisionTree, out_file=None,
                           feature_names=['Moyenne A_X', 'Ecart-type A_X', 'Moyenne A_Y','Ecart-type A_Y', 'Moyenne A_Z','Ecart-type A_Y', 'Moyenne G_X','Ecart-type G_X', 'Moyenne G_Y','Ecart-type G_Y', 'Moyenne G_Z', 'Ecart-type G_Z'],
                           class_names=[str(i) for i in range(1, 28)],
                           filled=True, rounded=True, special_characters=True)

# Convertir le fichier DOT en image
graph = graphviz.Source(dot_data)
graph.render("decision_tree", format="png", cleanup=True)

predictions = []
for entry in testing_dataset:
    predictions.append(decisionTree.predict([entry]))

# ---------------------------------------------------------
# Evaluation du modèle et affichage de la matrice de confusion
# ---------------------------------------------------------

print("")
print("Evaluation du modèle : ")
print("")
evaluation(testing_labels, predictions)

# confusion_matrix_csv(testing_labels, predictions)
confusion_matrix_png(testing_labels, predictions)