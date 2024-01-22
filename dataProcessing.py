# ---------------------------------------------------------
# Ce fichier sert à construire les jeux de données à partir
# des fichiers .mat fournis et les enregistre
# sous la forme de fichiers .csv
# ---------------------------------------------------------

# import numpy as np 
# import pandas as pd
# from scipy.io import loadmat

from functions import *

# ---------------------------------------------------------
# Construction du dataset à partir des fichiers .mat fournis
# ---------------------------------------------------------

print("Construction du jeu de données en cours...")

data = read_all_mat_files()
# Enregistrement du DataFrame
data.to_csv('dataset_imu.csv')

print("Construction du jeu de données terminé.")

# ---------------------------------------------------------
# Division du dataset en ensembles d'apprentissage et de validation
# ---------------------------------------------------------

print("Division du jeu de données en cours...")

training_dataset, training_labels, testing_dataset, testing_labels = normalize_data(data)

# On supprime les 3 dernières colonnes pour ne garder que les données de l'accéléromètre et du gyroscope
training_dataset.drop(columns=[training_dataset.columns[-1]], inplace=True)
training_dataset.drop(columns=[training_dataset.columns[-1]], inplace=True)
training_dataset.drop(columns=[training_dataset.columns[-1]], inplace=True)

testing_dataset.drop(columns=[testing_dataset.columns[-1]], inplace=True)
testing_dataset.drop(columns=[testing_dataset.columns[-1]], inplace=True)
testing_dataset.drop(columns=[testing_dataset.columns[-1]], inplace=True)

training_dataset.to_csv('training_dataset.csv')
training_labels.to_csv('training_labels.csv')
testing_dataset.to_csv('testing_dataset.csv')
testing_labels.to_csv('testing_labels.csv')

print("Division du jeu de données terminé.")