# Projet de classification - AADA

Auteurs : 
- Yahya NASSIF
- Antoine JONCHERAY

Fichiers du projet de Data Mining & Machine Learning (2024)

## Fichiers et dossiers

Les principaux fichiers sont :

- `main.ipynb` : "Brouillon" utilisé pour coder les différentes parties du projet.
- `dataProcessing.py` : script contenant les fonctions à coder tout au long du projet, ainsi que la fonction pour construire le dataset à partir des fichiers .mat et les ensembles d'apprentissage et de validation.
- `decisionTree.py` :  construction et entraînement d'un arbre de décision.
- `neuralNetwork.py` : construction et entraînement d'un réseau de neurones avec TensorFlow.

`processed_data` contient les fichiers .csv issus du traitement des fichiers .mat.
`models` contient les paramètres des réseaux de neurones qu'on a construit tout au long du projet.
`img` contient les matrices de confusion des modèles.

## Mise en place

### Sous Linux : 

Exécuter les commandes suivantes : 
```bash
git clone https://github.com/Havane44/Projet-AADA.git
cd Projet-AADA

# Environnement virtuel
python3 -m venv env
source ./env/bin/activate

# Packages à installer
pip install pandas scikit-learn numpy scipy tensorflow progress matplotlib seaborn graphviz

# Pour construire les datasets à partir des fichiers .mat
python3 dataProcessing.py
```