# Projet MIND
## Sources

La base de données [MNIST ASL](https://www.kaggle.com/datasets/datamunge/sign-language-mnist).

L'article [A novel model for expanding horizons in sign Language recognition](https://www.nature.com/articles/s41598-025-09643-2.pdf).

## Créer l'environnement virtuel

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
my-venv/Scripts/python interface.py
```

## Lancer l'interface

```
source my-venv/bin/activate
python interface.py
```
Ou alors :
```
my-venv/Scripts/python interface.py
```

## Créer le requirements.txt
```
python3.12 -m venv my-venv
my-venv/Scripts/pip install mediapipe opencv-python matplotlib pandas torch tensorflow scikit-learn seaborn python-Levenshtein
my-venv/Scripts/pip freeze > requirements.txt
```