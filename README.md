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

Pour mettre en place l'environnement pour lancer les scripts : 
```bash
git clone https://github.com/Havane44/Projet-AADA.git
cd Projet-AADA

python3 -m venv env
source ./env/bin/activate
pip install pandas scikit-learn numpy scipy tensorflow progress matplotlib

# Pour construire les datasets à partir des fichiers .mat
python3 dataProcessing.py
```