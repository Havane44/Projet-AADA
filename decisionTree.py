# ---------------------------------------------------------
# Ce fichier sert à construire et entraîner un arbre de 
# décision.
# ---------------------------------------------------------

from sklearn import tree
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
from time import time

# ---------------------------------------------------------
# Importation des données
# ---------------------------------------------------------
print("Importation des données en cours...")
start = time()

training_dataset = pd.read_csv('training_dataset.csv')
training_labels = pd.read_csv('training_labels.csv')
testing_dataset = pd.read_csv('testing_dataset.csv')
testing_labels = pd.read_csv('testing_labels.csv')

end = time()
print("Importation des données terminée en", end-start, "secondes")
# ---------------------------------------------------------
# Partie 9
# Construction de modèles prédictifs avec plusieurs méthodes
# ---------------------------------------------------------

print("Entraînement de l'arbre de décision en cours...")
start = time()

# Construction et entraînement d'un arbre de décision
decisionTree = tree.DecisionTreeClassifier(max_depth=15)
decisionTree = decisionTree.fit(training_dataset,training_labels)

end = time()
print("Entraînement de l'arbre de décision terminé en", end-start, "secondes")

# tree.plot_tree(decisionTree, feature_names=['Accel X', 'Accel Y', 'Accel Z', 'Gyro X', 'Gyro Y', 'Gyro Z'], class_names=[str(i) for i in list(range(1, 28))], filled=True)
tree.plot_tree(decisionTree, 
               feature_names=['Accéléromètre X', 'Accéléromètre Y', 'Accéléromètre Z', 'Gyroscope X', 'Gyroscope Y', 'Gyroscope Z'],
               fontsize=12)
plt.show()

predictions = decisionTree.predict(training_dataset)
accuracy = accuracy_score(testing_labels, predictions)
print(f"Précision de l'arbre de décision : {accuracy * 100:.2f}%")