from flask import Flask, render_template, request, jsonify
from collections import Counter
import json
import os
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load symptom column names from the dataset
df = pd.read_csv('Final_Cleaned_Disease_Data.csv', nrows=1)
symptoms_list = list(df.drop(columns=['diseases']).columns)

# Load the three trained models
dt_model = joblib.load('dt_model.pkl')
nb_model = joblib.load('nb_model.pkl')
knn_model = joblib.load('knn_model.pkl')

# Load offline Arabic translations
TRANSLATIONS_FILE = 'diseases_ar.json'
if os.path.exists(TRANSLATIONS_FILE):
    with open(TRANSLATIONS_FILE, 'r', encoding='utf-8') as f:
        disease_translations = json.load(f)
else:
    disease_translations = {}

def format_disease_name(disease_en):
    disease_clean = str(disease_en).strip()
    return disease_translations.get(disease_clean.lower(), disease_clean)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/how-it-works')
def how_it_works():
    return render_template('how_it_works.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    selected_symptoms = data.get('symptoms', [])

    # Build binary input vector (0s and 1s)
    input_vector = np.zeros(len(symptoms_list))
    for sym in selected_symptoms:
        if sym in symptoms_list:
            idx = symptoms_list.index(sym)
            input_vector[idx] = 1

    input_df = pd.DataFrame([input_vector], columns=symptoms_list)

    # Get predictions from all three models
    dt_pred = str(dt_model.predict(input_df)[0])
    nb_pred = str(nb_model.predict(input_df)[0])
    knn_pred = str(knn_model.predict([input_vector])[0])

    # Majority voting (fallback to Naive Bayes if all 3 differ)
    votes = Counter([dt_pred, nb_pred, knn_pred])
    top_disease, top_count = votes.most_common(1)[0]

    if top_count >= 2:
        final_pred = top_disease
        agreement_note = f"بناءً على اتفاق {top_count} من أصل 3 خوارزميات"
    else:
        final_pred = nb_pred
        agreement_note = "بناءً على النموذج الأعلى دقة (Naive Bayes - 83.94%)"

    return jsonify({
        'Final_Diagnosis': format_disease_name(final_pred),
        'Agreement_Note': agreement_note,
        'Decision_Tree': format_disease_name(dt_pred),
        'Naive_Bayes': format_disease_name(nb_pred),
        'KNN': format_disease_name(knn_pred)
    })

if __name__ == '__main__':
    app.run(debug=True)