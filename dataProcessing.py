# ---------------------------------------------------------
# Ce fichier sert à construire les jeux de données à partir
# des fichiers .mat fournis et les enregistre
# sous la forme de fichiers .csv
# ---------------------------------------------------------

from functions import *

# ---------------------------------------------------------
# Construction du dataset à partir des fichiers .mat fournis
# ---------------------------------------------------------

print("Construction du jeu de données en cours...")

data = read_all_mat_files()
# Enregistrement du DataFrame
data.to_csv('./processed_data/dataset_imu.csv')

print("Construction du jeu de données terminé.")

# ---------------------------------------------------------
# Extraction des features et division du dataset en 
# ensembles d'apprentissage et de validation
# ---------------------------------------------------------

print("Division du jeu de données en cours...")

# Chargement des données depuis le fichier CSV
data = pd.read_csv("./processed_data/dataset_imu.csv")

# Appel de la fonction pour obtenir le tableau combiné
data = feature_extraction_moyenne_et_ecart_type(data)

training_dataset = data[(data['id_sujet'] == 1) | 
                        (data['id_sujet'] == 3) | 
                        (data['id_sujet'] == 5) | 
                        (data['id_sujet'] == 7)]
training_labels = training_dataset['id_action']
    
testing_dataset = data[(data['id_sujet'] == 2) |
                       (data['id_sujet'] == 4) | 
                       (data['id_sujet'] == 6) | 
                       (data['id_sujet'] == 8)]
testing_labels = testing_dataset['id_action']

training_dataset.to_csv('./processed_data/training_dataset.csv')
training_labels.to_csv('./processed_data/training_labels.csv')
testing_dataset.to_csv('./processed_data/testing_dataset.csv')
testing_labels.to_csv('./processed_data/testing_labels.csv')

print("Division du jeu de données terminé.")