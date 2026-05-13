# Projet MIND

Ce code permet, à partir d'une vidéo ou d'une caméra en temps réel, de transcrire automatiquement la langue des signes en [Typannot](https://www.typannot.com/) au format `.eaf` utilisable par [ELAN](https://archive.mpi.nl/tla/elan).

## Lancer le programme
### Utiliser la police d'écriture Typannot

Ce projet nécessite d'avoir la police d'écriture `TYPANNOT Beta Generics-Postural_Release_v3`. Sous Windows, mettez cette police d'écriture dans le dossier `C:\Windows\Fonts`. Pour Linux, le dossier varie selon les distributions.

### Créer l'environnement virtuel

Sous Linux :
```
python3.12 -m venv my-venv
source my-venv/bin/activate
pip install -r requirements.txt
```

Sous Windows :
```
python3.12 -m venv my-venv
my-venv/Scripts/pip install -r requirements.txt
```

### Télécharger le modèle MediaPipe Hand Landmarker.

Le modèle [MediaPipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) est un modèle créé par Google qui prend en entrée une photo et qui y place des points de repère de la main. Il est nécessaire de l'installer pour le projet. Vous pouvez le télécharger [ici](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task), puis le mettre à côté du fichier `main.py`.

### Lancer le code

Exécutez les commandes suivantes pour Linux :
```
source my-venv/bin/activate
python main.py
```
Ou alors pour Windows :
```
my-venv/Scripts/python main.py
```

## Utilisation du programme

Le programme a deux fonctionnalités : nous pouvons soit importer une vidéo, soit prendre une vidéo en temps réel à l'aide d'une caméra. Dans les deux cas, la transcription Typannot s'affichera en temps réel tout en étant stocké dans un fichier `.eaf`.

Ce fichier `.eaf` peut ensuite être ouvert à l'aide d'ELAN. Pour utiliser la police d'écriture Typannot dans ELAN, allez dans `Acteur > Modifier les attributs d'acteur > Plus d'options` puis sélectionner la police de l'acteur et cliquez sur `Appliquer`.

<!--
## Créer le requirements.txt
```
my-venv/Scripts/pip install mediapipe opencv-python matplotlib pandas torch tensorflow scikit-learn seaborn python-Levenshtein
my-venv/Scripts/pip freeze > requirements.txt
```
-->

## Autres fonctionnalités

### Utiliser un modèle basé sur MNIST ASL

Au lieu d'utiliser le modèle MediaPipe Hand Landmarker, il est aussi possible d'utiliser le modèle `model.keras`. Ce modèle est basé sur le modèle proposé dans [cet article](https://www.nature.com/articles/s41598-025-09643-2.pdf) sur la base de données [MNIST ASL](https://www.kaggle.com/datasets/datamunge/sign-language-mnist). Si vous voulez utiliser ce second modèle, modifier le fichier `main.py` à la ligne 310 pour mettre la variable `uses_mnist` à `True` au lieu de `False`.

### Entrainer le modèle Keras

Si vous voulez entrainer le modèle Keras par vous-mêmes, il faudra installer la base de données MNIST ASL [ici](https://github.com/namas191297/sign_language_mnist_cnn). Plus précisément, il faudra installer `sign_mnist_train.csv` et `sign_mnist_test.csv` puis les mettre à côté de `model_trainer.py` avant d'exécuter cette commande :
```
python model_trainer.py
```