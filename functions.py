# ---------------------------------------------------------
# Ce fichier contient des fonctions utiles pour charger et 
# traiter les données de la base de donnée fournie.
# ---------------------------------------------------------

from scipy.io import loadmat
from sklearn.metrics import classification_report, confusion_matrix
from seaborn import heatmap

import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------------------
# Partie 6 : Chargement des données
# ---------------------------------------------------------

def load_data(nb_action: int, nb_sujet: int, nb_essai: int, data):
    """
    Fonction pour charger et lire un fichier .mat
    """
    try:
        file = loadmat("./imu_data/a{0}_s{1}_t{2}_inertial.mat".format(nb_action, nb_sujet, nb_essai))
        file_data = file['d_iner']
        for line in file_data:
            data["accéléromètre_X"].append(line[0])
            data["accéléromètre_Y"].append(line[1])
            data["accéléromètre_Z"].append(line[2])
            data["gyroscope_X"].append(line[3])
            data["gyroscope_Y"].append(line[4])
            data["gyroscope_Z"].append(line[5])
            data["id_sujet"].append(nb_sujet)
            data["id_essai"].append(nb_essai)
            data["id_action"].append(nb_action)
    except:
        print("fichier a{0}_s{1}_t{2}_inertial.mat manquant".format(nb_action, nb_sujet, nb_essai))

def read_all_mat_files() -> pd.DataFrame: 
    """
    Fonction qui lit tous les fichiers .mat et construit un dataframe contenant toutes les données.
    """

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
                load_data(action, sujet, essai, data)

    return pd.DataFrame(data)

def tracer_signal(dataframe: pd.DataFrame, capteur: int, num_action: int, num_sujet: int, num_essai: int):
    """
    Fonction pour tracer les trois signaux d'un capteur pour une action, un sujet et un essai particulier.
    
    Exemple d'utilisation :
    tracer_signal(data, capteur=2, num_action=2, num_sujet=1, num_essai=1)
    """

    subset = dataframe[(dataframe['id_action'] == num_action) & 
                       (dataframe['id_sujet'] == num_sujet) & 
                       (dataframe['id_essai'] == num_essai)]

    if capteur == 1:  # Accéléromètre
        signal_x = subset['accéléromètre_X']
        signal_y = subset['accéléromètre_Y']
        signal_z = subset['accéléromètre_Z']
        capteur_name = 'Accéléromètre'
    elif capteur == 2:  # Gyroscope
        signal_x = subset['gyroscope_X']
        signal_y = subset['gyroscope_Y']
        signal_z = subset['gyroscope_Z']
        capteur_name = 'Gyroscope'
    else:
        print("Capteur non valide.")
        return

    # Créer un graphique pour chaque signal
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(signal_x)
    plt.title(f'Signal {capteur_name} - Axe X')
    plt.xlabel('Échantillons')
    plt.ylabel('Amplitude')

    plt.subplot(3, 1, 2)
    plt.plot(signal_y)
    plt.title(f'Signal {capteur_name} - Axe Y')
    plt.xlabel('Échantillons')
    plt.ylabel('Amplitude')

    plt.subplot(3, 1, 3)
    plt.plot(signal_z)
    plt.title(f'Signal {capteur_name} - Axe Z')
    plt.xlabel('Échantillons')
    plt.ylabel('Amplitude')

    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------
# Partie 7 : Calcul des attributs (moyenne et écart-type)
# ---------------------------------------------------------

def feature_extraction_moyenne(dataframe: pd.DataFrame):
    """
    Fonction qui calcule la moyenne des échantillons pour chaque type d'action
    """
    
    res = []
    # On cycle sur les 27 actions possibles (de 1 à 27)
    for num_action in range(1, 28):
        # Dataframe des lignes de l'action i
        subset = dataframe[(dataframe['id_action'] == 1)]
        res.append([subset['accéléromètre_X'].mean(), 
                    subset['accéléromètre_Y'].mean(), 
                    subset['accéléromètre_Z'].mean(),
                    subset['gyroscope_X'].mean(),
                    subset['gyroscope_Y'].mean(),
                    subset['gyroscope_Z'].mean()
                    ])
    return res

def feature_extraction_ecart_type(dataframe: pd.DataFrame):
    """
    Fonction qui calcule l'écart-type des échantillons pour chaque type d'action
    """

    res = []
    # On cycle sur les 27 actions possibles (de 1 à 27)
    for num_action in range(1, 28):
        # Dataframe des lignes de l'action i
        subset = dataframe[(dataframe['id_action'] == num_action)]
        res.append([subset['accéléromètre_X'].std(),
                    subset['accéléromètre_Y'].std(), 
                    subset['accéléromètre_Z'].std(),
                    subset['gyroscope_X'].std(),
                    subset['gyroscope_Y'].std(),
                    subset['gyroscope_Z'].std()
                    ])
    return res

# ---------------------------------------------------------
# Partie 8 : Préparation des données
# ---------------------------------------------------------

def normalize_data(data: pd.DataFrame):
    """
    Fonction pour séparer et normaliser les données grâce à la moyenne et l'écart-type
    """
    
    # Division du dataset original en 4 datasets (apprentissage & test)
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
    
    moyennes = feature_extraction_moyenne(data)
    ecarts_types = feature_extraction_ecart_type(data)

    # Normalisation des données d'apprentissage
    for index, row in training_dataset.iterrows():
        training_dataset.at[index,'accéléromètre_X'] = (row['accéléromètre_X'] - moyennes[int(row['id_action']) - 1][0])/(ecarts_types[int(row['id_action']) - 1][0])
        training_dataset.at[index,'accéléromètre_Y'] = (row['accéléromètre_Y'] - moyennes[int(row['id_action']) - 1][1])/(ecarts_types[int(row['id_action']) - 1][1])
        training_dataset.at[index,'accéléromètre_Z'] = (row['accéléromètre_Z'] - moyennes[int(row['id_action']) - 1][2])/(ecarts_types[int(row['id_action']) - 1][2])
        training_dataset.at[index,'gyroscope_X'] = (row['gyroscope_X'] - moyennes[int(row['id_action']) - 1][3])/(ecarts_types[int(row['id_action']) - 1][3])
        training_dataset.at[index,'gyroscope_Y'] = (row['gyroscope_Y'] - moyennes[int(row['id_action']) - 1][4])/(ecarts_types[int(row['id_action']) - 1][4])
        training_dataset.at[index,'gyroscope_Z'] = (row['gyroscope_Z'] - moyennes[int(row['id_action']) - 1][5])/(ecarts_types[int(row['id_action']) - 1][5])

    for index, row in testing_dataset.iterrows():
        testing_dataset.at[index,'accéléromètre_X'] = (row['accéléromètre_X'] - moyennes[int(row['id_action']) - 1][0])/(ecarts_types[int(row['id_action']) - 1][0])
        testing_dataset.at[index,'accéléromètre_Y'] = (row['accéléromètre_Y'] - moyennes[int(row['id_action']) - 1][1])/(ecarts_types[int(row['id_action']) - 1][1])
        testing_dataset.at[index,'accéléromètre_Z'] = (row['accéléromètre_Z'] - moyennes[int(row['id_action']) - 1][2])/(ecarts_types[int(row['id_action']) - 1][2])
        testing_dataset.at[index,'gyroscope_X'] = (row['gyroscope_X'] - moyennes[int(row['id_action']) - 1][3])/(ecarts_types[int(row['id_action']) - 1][3])
        testing_dataset.at[index,'gyroscope_Y'] = (row['gyroscope_Y'] - moyennes[int(row['id_action']) - 1][4])/(ecarts_types[int(row['id_action']) - 1][4])
        testing_dataset.at[index,'gyroscope_Z'] = (row['gyroscope_Z'] - moyennes[int(row['id_action']) - 1][5])/(ecarts_types[int(row['id_action']) - 1][5])
    
    return training_dataset, training_labels, testing_dataset, testing_labels

# ---------------------------------------------------------
# Partie 11 : calcul des métriques utiles pour 
# l'évaluation du modèle
# ---------------------------------------------------------

def evaluation(true_labels, predicted_labels):
    """
    Affiche les principales métriques du modèle : 
    - précision
    - rappel
    - F-score
    - exactitude
    """

    target_names = ['Action {0}'.format(i) for i in range(1, 28)]
    print(classification_report(true_labels, predicted_labels, target_names=target_names))

def confusion_matrix_csv(true_labels, predicted_labels):
    """
    Calcule la matrice de confusion du modèle 
    et l'enregistre au format CSV.
    """

    actions = ['Action {0}'.format(i) for i in range(1, 28)]
    pd.DataFrame(confusion_matrix(true_labels, predicted_labels), columns=actions, index=actions).to_csv("matrice_de_confusion.csv")

def confusion_matrix_png(true_labels, predicted_labels):
    """
    Calcule la matrice de confusion du modèle et l'affiche à l'écran.
    """
    
    heatmap(confusion_matrix(true_labels, predicted_labels), annot=True, fmt='d', cmap='Blues',
                xticklabels=[str(i) for i in range(1, 28)],
                yticklabels=[str(i) for i in range(1, 28)])
    plt.xlabel('Prédictions')
    plt.ylabel('Vraies étiquettes')
    plt.title('Matrice de Confusion')
    plt.show()
