# Projet de classification - AADA

**A propos des données :**
- 27 actions différentes
- 8 sujets
- Les sujets ont réalisé chaque action 4 fois => 8x27x4 = 864 enregistrements
- 6 données par enregistrement (les 3 axes de l'accéléromètre et du gyroscope)

**A propos des fichiers .mat :**
- Dictionnaire python avec les champs suivants : 
    - `__header__` : infos sur le fichier
    - `__version_` : version du fichier
    - `__globals__` : c'est vide 
    - `d_iner` : un array qui contient toutes les données du fichier

## To-Do : 
- [ ] Ecrire une fonction “load_data” qui permet de charger les données sur python (vous pouvez utilisez la fonction loadmat de la bibliothèque scipy). La fonction doit renvoyer un “dataframe” organisé comme suite :
    - Colonne 0-2 : contiennent les données de l’accéléromètre suivant les trois axes.
    - Colonne 3-5 : contiennent les données du gyroscope suivant les trois axes.
    - Colonne 6 : contient l’identifiant du sujet (1 à 8).
    - Colonne 7 : contient l’identifiant de l’essai (1 à 4).
    - Colonne 8 : contient l’identifiant (l’étiquette) de l’action (1 à 27).

- [ ] Ecrire une fonction “tracer_signal” qui permet de tracer les trois signaux (x, y, z) d’un capteur correspondant à une action. Cette fonction prend 5 entrées :
    - Le dataframe (renvoyer par la fonction “load_data”).
    - Le capteur (1 : accéléromètre, 2 : gyroscope).
    - Le numéro de l’action.
    - Le numéro du sujet.
    - Le numéro de l’essai.

- [ ] Ecrire une fonction ”feature_extraction” qui prend en entrée le “dataframe” crée et calcule pour chaque action un vecteur d’attributs.

*Exemple d'attributs : moyenne, moyenne quadratique, écart-type, médiane, quartiles, coefficient d’asymétrie, Kurtosis, entropie, énergie, maxima et minima, coefficient de
corrélation, histogramme, passages par zéro, nombre d’occurrences de pics.*

- [ ] Créez une fonction qui permet de calculer le vecteur des moyennes et des l’écart-types sur les attributs d’apprentissage (la dimensions des vecteurs doit être égale aux nombre d’attributs calculés), puis qui soustrait ensuite le vecteur moyenne des données d’apprentissage et de test et qui enfin divise le tout par l’écart-type.

- [ ] 