# ---------------------------------------------------------
# Ce fichier sert à construire et entraîner des modèles 
# prédictifs.
# ---------------------------------------------------------

from sklearn import tree
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Importation des données
# ---------------------------------------------------------
training_dataset = pd.read_csv('training_dataset.csv')
training_labels = pd.read_csv('training_labels.csv')
testing_dataset = pd.read_csv('testing_dataset.csv')
testing_labels = pd.read_csv('testing_labels.csv')

# ---------------------------------------------------------
# Partie 9
# Construction de modèles prédictifs avec plusieurs méthodes
# ---------------------------------------------------------

# Construction et entraînement d'un arbre de décision
data = pd.read_csv("training_dataset.csv")
labels = pd.read_csv("training_labels.csv")
clf = tree.DecisionTreeClassifier(max_depth=8)
clf = clf.fit(data,labels)

# tree.plot_tree(clf, feature_names=['Accel X', 'Accel Y', 'Accel Z', 'Gyro X', 'Gyro Y', 'Gyro Z'], class_names=[str(i) for i in list(range(1, 28))], filled=True)
tree.plot_tree(clf)
plt.show()

predictions = clf.predict(training_dataset)
accuracy = accuracy_score(testing_labels, predictions)
print(f"Précision de l'arbre de décision : {accuracy * 100:.2f}%")