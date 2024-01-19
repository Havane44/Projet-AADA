import numpy as np 
import pandas as pd
from scipy.io import loadmat

def load_data(nb_action, nb_sujet, nb_essai, data):
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
load_data(1, 1, 1, data)

for action in range(1, 28):
    for sujet in range(1, 9):
        for essai in range(1, 5):
            # lire le contenu du fichier et l'ajouter au dataframe
            # appeler la fonction qui lit le fichier souhaité
            load_data(action, sujet, essai, data)

data = pd.DataFrame(data)
print(data)