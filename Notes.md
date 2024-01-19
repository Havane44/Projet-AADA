**A propos des fichiers .mat :**
- Dictionnaire python avec les champs suivants : 
    - `__header__` : infos sur le fichier
    - `__version_` : version du fichier
    - `__globals__` : c'est vide 
    - `d_iner` : un array qui contient toutes les données du fichier

## To-Do : 
- [x] Ecrire une fonction “load_data” qui permet de charger les données sur Python.

- [x] Ecrire une fonction “tracer_signal” qui permet de tracer les trois signaux (x, y, z) d’un capteur correspondant à une action. Cette fonction prend 5 entrées :
    - Le dataframe (renvoyer par la fonction “load_data”).
    - Le capteur (1 : accéléromètre, 2 : gyroscope).
    - Le numéro de l’action.
    - Le numéro du sujet.
    - Le numéro de l’essai.

- [x] Ecrire une fonction ”feature_extraction” qui prend en entrée le “dataframe” crée et calcule pour chaque action un vecteur d’attributs.

*Exemple d'attributs : moyenne, moyenne quadratique, écart-type, médiane, quartiles, coefficient d’asymétrie, Kurtosis, entropie, énergie, maxima et minima, coefficient de
corrélation, histogramme, passages par zéro, nombre d’occurrences de pics.*

- [x] Créez une fonction qui permet de calculer le vecteur des moyennes et des l’écart-types sur les attributs d’apprentissage (la dimensions des vecteurs doit être égale aux nombre d’attributs calculés), puis qui soustrait ensuite le vecteur moyenne des données d’apprentissage et de test et qui enfin divise le tout par l’écart-type.

- [ ] 