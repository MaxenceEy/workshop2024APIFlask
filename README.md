# Hate Speech Detection API

## Description

Cette API Flask détecte le discours haineux à partir de commentaires en utilisant un modèle de machine learning basé sur un Support Vector Machine (SVM). Les commentaires sont prétraités pour éliminer les caractères spéciaux, les chiffres et pour les normaliser avant d'être analysés par le modèle.

## Fonctionnalités

- Nettoyage et prétraitement des commentaires.
- Prédiction de la nature du commentaire (discours haineux ou non).
- Réponse au format JSON.

## Technologies Utilisées

- Python
- Flask
- Scikit-learn
- Pandas
- Regex

## Installation

### Prérequis

- Python 3.x
- pip


## Structure du Code

- **app.py** : Fichier principal contenant la logique de l'application Flask, le nettoyage du texte, la prédiction et le lancement du serveur.

### Exemple de réponse

```json
{
    "result": "Hate Speech"
}
```
