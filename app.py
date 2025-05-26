from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib

# Inicializar o app Flask
app = Flask(__name__)
CORS(app)  # Liberar CORS para requisições externas (frontend separado)

# Carregar modelos e pré-processador
rf_model = joblib.load("modelo_colorectal_rf.pkl")
nb_model = joblib.load("modelo_colorectal_nb.pkl")
xgb_model = joblib.load("modelo_colorectal_xgb.pkl")
preprocessor = joblib.load("preprocessor.pkl")

# Rota principal (formulário)
@app.route('/')
def home():
    return render_template('index.html')

# Rota de resultados (renderiza resultados.html)
@app.route('/resultados')
def resultados():
    # pegar os dados do query string
    params = request.args.to_dict()
    return render_template("resultados.html", resultados=params)

# Rota de predição
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Receber dados JSON do frontend
    df = pd.DataFrame([data])

    # Garantir que todas as colunas esperadas estejam presentes
    expected_columns = [
        'Age', 'Gender', 'Socioeconomic_Status', 'Red_Meat_Consumption', 'Screening_Regularity',
        'Alcohol_Consumption', 'BMI', 'Tumor_Aggressiveness', 'Colonoscopy_Access', 'Region',
        'Chemotherapy_Received', 'Urban_or_Rural', 'Follow_Up_Adherence', 'Surgery_Received',
        'Physical_Activity_Level', 'Insurance_Coverage', 'Race', 'Fiber_Consumption',
        'Time_to_Recurrence', 'Diet_Type', 'Radiotherapy_Received', 'Previous_Cancer_History',
        'Family_History', 'Treatment_Access', 'Smoking_Status', 'Recurrence', 'Time_to_Diagnosis'
    ]

    for col in expected_columns:
        if col not in df.columns:
            df[col] = "No"

    # Converter colunas numéricas
    numeric_cols = ['Age', 'BMI', 'Time_to_Recurrence']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    print("Colunas corrigidas no DataFrame:", df.dtypes)

    # Pré-processamento
    processed = preprocessor.transform(df)
    processed_df = pd.DataFrame(processed, columns=preprocessor.get_feature_names_out())

    # Previsão
    predictions = {
        "RandomForest": f"{rf_model.predict_proba(processed_df)[0][1]:.4f}",
        "XGBoost": f"{xgb_model.predict_proba(processed_df)[0][1]:.4f}",
        "NaiveBayes": f"{nb_model.predict_proba(processed_df)[0][1]:.4f}"
    }

    print("JSON enviado ao frontend:", predictions)
    return jsonify(predictions)

# Rodar app
if __name__ == '__main__':
    app.run(debug=True)
