# ---------------------------------------------------------
# Ce fichier sert à construire les jeux de données à partir
# des fichiers .mat fournis et les enregistre
# sous la forme de fichiers .csv
# ---------------------------------------------------------

import numpy as np 
import pandas as pd
from scipy.io import loadmat

from fonctions import *

# ---------------------------------------------------------
# Construction du dataset à partir des fichiers .mat fournis
# ---------------------------------------------------------

print("Construction du jeu de données en cours...")

data = {
    "accéléromètre_X": [],
    "accéléromètre_Y": [],
    "accéléromètre_Z": [],
    "gyroscope_X": [],
    "gyroscope_Y": [],
    "gyroscope_Z": [],
    "id_sujet": [],
    "id_essai": [],
    "id_action": []
}

for action in range(1, 28):
    for sujet in range(1, 9):
        for essai in range(1, 5):
            # lire le contenu du fichier et l'ajouter au dataframe
            # appeler la fonction qui lit le fichier souhaité
            load_data(action, sujet, essai, data)

data = pd.DataFrame(data)
data.to_csv('dataset_imu.csv')

print("Construction du jeu de données terminé.")

# ---------------------------------------------------------
# Division du dataset en ensembles d'apprentissage et de validation
# ---------------------------------------------------------

print("Division du jeu de données en cours...")

training_dataset, training_labels, testing_dataset, testing_labels = normalize_data(data)

# training_dataset.drop('id_action', axis=1)
# testing_dataset.drop('id_action', axis=1)

training_dataset.to_csv('training_dataset.csv')
training_labels.to_csv('training_labels.csv')
testing_dataset.to_csv('testing_dataset.csv')
testing_labels.to_csv('testing_labels.csv')

print("Division du jeu de données terminé.")