from flask import Flask, request, jsonify
import pandas as pd
import re
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

# Charger votre modèle entraîné (ou recréez-le si nécessaire)
vectorizer = CountVectorizer()  # Utilisez le même vectorizer que pour l'entraînement
model = MultinomialNB()  # Utilisez le même modèle

# Fonction de nettoyage de texte
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Endpoint pour tester si un commentaire est un discours de haine
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    comments = data.get('phrases', [])  # S'assurer que l'on attend 'phrases' comme clé
    
    # Nettoyage et vectorisation des commentaires
    cleaned_comments = [clean_text(comment) for comment in comments]
    X = vectorizer.transform(cleaned_comments)
    
    # Prédictions
    predictions = model.predict(X)
    
    # Formater les résultats pour l'extension
    results = [{'result': 'Hate Speech' if pred == 1 else 'Non Hate Speech'} for pred in predictions]
    
    # Retourner le résultat en JSON
    return jsonify(results)

if __name__ == '__main__':
    app.run(port=5000)
