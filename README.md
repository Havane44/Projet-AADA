# Projet de classification - AADA

Auteurs : 
- Yahya NASSIF
- Antoine JONCHERAY

Fichiers du projet de Data Mining & Machine Learning.

## A propos de l'organisation des fichiers :

Les principaux fichiers sont :

- `main.ipynb` : "Brouillon" utilisé pour coder les différentes parties du projet
- `functions.py` : Les fonctions à coder tout au long du projet
- `dataProcessing.py` : script pour construire le dataset à partir des fichiers .mat et les ensembles d'apprentissage et de validation
- `decisionTree.py` :  construction et entraînement d'un arbre de décision
- `neuralNetwork.py` : construction et entraînement d'un réseau de neurones avec TensorFlow

## Packages à installer (via pip)

- pandas
- scikit-learn
- numpy
- scipy
- tensorflow
- progress
- matplotlib

Pour mettre en place l'environnement afin de lancer les scripts : 
```bash
git clone https://github.com/Havane44/Projet-AADA.git
cd Projet-AADA

python3 -m venv env
source ./env/bin/activate
pip install pandas scikit-learn numpy scipy tensorflow progress matplotlib

# Pour construire les datasets à partir des fichiers .mat
python3 dataProcessing.py
```

## Performance des classifieurs

### Arbres de décision

Version 1 : 
- Profondeur : 30
- Précision : 30.92%

Version 2 : 
- Profondeur : 
- Précision : 

### Réseaux de neurones

Version 1 : 
- Deux couches cachées à 128 et 64 neurones (fonction d'activation : relu)
- Couche de sortie avec 27 neurones (fonction d'activation : softmax)
- 50 epochs, batchs de 100
- Précision : 38%

### Machines à vecteur de support

### K-plus proches voisins

## To-Do list : 
- [x] Fonction `load_data()`
- [x] Fonction `tracer_signal()`
- Fonctions `feature_extraction()`
    - [x] Moyenne
    - [x] Ecart-type
- Arbre de décision
    - [x] Construction
    - [x] Test
    - [ ] Amélioration
- Réseau de neurones
    - [x] Construction
    - [x] Test
    - [ ] Amélioration
- K-plus proches voisins
    - [ ] Construction
    - [ ] Test
    - [ ] Amélioration
- Machine à vecteur de support
    - [ ] Construction
    - [ ] Test
    - [ ] Amélioration
- Fonctions pour calculer les métriques des performances
    - [ ] Précision
    - [ ] Rappel
    - [ ] F-Score
    - [ ] Exactitude (accuracy)
- Fonction pour tracer la matrice de confusion