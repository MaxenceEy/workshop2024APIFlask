from flask import Flask, request, jsonify
import pandas as pd
import re
import string
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

vectorizer = TfidfVectorizer()
model = SVC(kernel='linear')

# clean le commentaire
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# endpoint pour tester si un commentaire est un discours de haine
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    comment = data.get('comment', '')
    
    # celan et vectorisation des commentaires
    cleaned_comments = [clean_text(comment)]
    X = vectorizer.transform(cleaned_comments)
    
    # prediction
    prediction = model.predict(X)
    
    # formatage du résultat
    result = 'Hate Speech' if prediction[0] == 1 else 'Non Hate Speech'
    
    # return au format json
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(port=5000) # attention port 5000
