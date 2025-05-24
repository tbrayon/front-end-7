import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os
import joblib

# ----------------------
# 1. DATA PREPARATION
# ----------------------

# Load and clean data
df = pd.read_csv("c:/Users/gina_/OneDrive/Área de Trabalho/PUC/AULAS 7 SEMESTRE/Projeto/Projeto/Etapa 3/colorectal_cancer_prediction_csv.csv")
df = df.drop(columns=['Patient_ID'])

# Separate features and target
target = 'Survival_Status'
features = df.drop(columns=[target])
# y = df[target]
y = df['Survival_Status'].replace({'Survived': 1, 'Deceased': 0}).infer_objects(copy=False)

# Split before any transformations
X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, stratify=y, random_state=42)

print(f"Número de linhas em X_train: {X_train.shape[0]}")
print(f"Número de colunas em X_train: {X_train.shape[1]}")

print(f"Número de linhas em X_test: {X_test.shape[0]}")
print(f"Número de colunas em X_test: {X_test.shape[1]}")

# Detect feature types
numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()
categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

# Pipelines
numeric_transformer = Pipeline(steps=[
	('imputer', SimpleImputer(strategy='median')),
	('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
	('imputer', SimpleImputer(strategy='most_frequent')),
	('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
	transformers=[
    	('num', numeric_transformer, numeric_features),
    	('cat', categorical_transformer, categorical_features)
	]
)

# Fit preprocessor and transform training and test features
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)
feature_names = preprocessor.get_feature_names_out()

X_train_df = pd.DataFrame(X_train_processed, columns=feature_names)
X_test_df = pd.DataFrame(X_test_processed, columns=feature_names)

# ----------------------
# 2. HANDLE IMBALANCE
# ----------------------

# Option A: Oversample minority with SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train_df, y_train)

# ----------------------
# 3. MODEL TRAINING
# ----------------------

# Sampling-based models
rf_model_sm = RandomForestClassifier(random_state=42)
nb_model_sm = GaussianNB()
xgb_model_sm = XGBClassifier(eval_metric='logloss', random_state=42)

rf_model_sm.fit(X_resampled, y_resampled)
nb_model_sm.fit(X_resampled, y_resampled)
xgb_model_sm.fit(X_resampled, y_resampled)

# Class-weighted models
rf_model_w = RandomForestClassifier(class_weight='balanced', random_state=42)
nb_model_w = GaussianNB()  # GaussianNB does not support class_weight directly
xgb_model_w = XGBClassifier(eval_metric='logloss', scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(), random_state=42)

rf_model_w.fit(X_train_df, y_train)
# For NB, manual sample weighting in fit
weights = np.where(y_train == 1, len(y_train) / (2 * (y_train == 1).sum()), len(y_train) / (2 * (y_train == 0).sum()))
nb_model_w.fit(X_train_df, y_train, sample_weight=weights)
xgb_model_w.fit(X_train_df, y_train)

# ----------------------
# 4. CONFUSION MATRICES
# ----------------------

def evaluate_models(models, strategy):
    print(f"\n--- Confusion Matrices ({strategy}) ---")
    for name, model in models.items():
        y_pred = model.predict(X_test_df)
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n{name}:")
        print(cm)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
        disp.plot(cmap='Blues')
        plt.title(name)
        plt.show()

# Collect models
models_smote = {
	'RandomForest_SMOTE': rf_model_sm,
	'NaiveBayes_SMOTE': nb_model_sm,
	'XGBoost_SMOTE': xgb_model_sm
}

models_weighted = {
	'RandomForest_Weighted': rf_model_w,
	'NaiveBayes_Weighted': nb_model_w,
	'XGBoost_Weighted': xgb_model_w
}

# Evaluate
evaluate_models(models_smote, 'SMOTE')
evaluate_models(models_weighted, 'Weighted')

# ----------------------
# 5. INFERENCE AND SAVE
# ----------------------

def recommend_treatment(patient_features, model, preprocessor, feature_names, base_features):
    combos = [
        {'Chemotherapy_Received': 'Yes', 'Radiotherapy_Received': 'Yes', 'Surgery_Received': 'Yes'},
        {'Chemotherapy_Received': 'Yes', 'Radiotherapy_Received': 'Yes', 'Surgery_Received': 'No'},
        {'Chemotherapy_Received': 'Yes', 'Radiotherapy_Received': 'No', 'Surgery_Received': 'Yes'},
        {'Chemotherapy_Received': 'Yes', 'Radiotherapy_Received': 'No', 'Surgery_Received': 'No'},
        {'Chemotherapy_Received': 'No', 'Radiotherapy_Received': 'Yes', 'Surgery_Received': 'Yes'},
        {'Chemotherapy_Received': 'No', 'Radiotherapy_Received': 'Yes', 'Surgery_Received': 'No'},
        {'Chemotherapy_Received': 'No', 'Radiotherapy_Received': 'No', 'Surgery_Received': 'Yes'}
    ]

    best_prob, best_combo = -1, None

    for combo in combos:
        patient = patient_features.copy()
        for k, v in combo.items():
            patient[k] = v
        
        # Alinha as colunas antes da transformação
        patient = patient[base_features]
        
        # Transforma os dados corretamente
        processed = preprocessor.transform(patient)
        processed_df = pd.DataFrame(processed, columns=feature_names)

        # Garante que `predict_proba` funciona corretamente
        prob = model.predict_proba(processed_df)[0][1]
        
        if prob > best_prob:
            best_prob, best_combo = prob, combo

    return best_combo, best_prob

def save_result(patient, model_name, combo, prob):
    ts = datetime.now().strftime("%Y%m%d%H%M%S%f")
    row = {
        'PatientID': ts,
        'Model': model_name,
        'RecommendedTreatment': combo,
        'SurvivalProb': f"{prob:.4f}",
        **patient.iloc[0].to_dict()
    }
    pd.DataFrame([row]).to_csv(
        "recommendation_results.csv", mode='a', index=False,
        header=not os.path.exists("recommendation_results.csv")
    )

# ----------------------
# 6. TEST FUNCTION
# ----------------------

# def generate_random_patient():
# 	return features.sample(n=1).reset_index(drop=True)

def generate_random_patient():
	"""Generate a synthetic colorectal cancer patient with realistic attributes"""
	patient = {
    	# Demographics
    	'Age': np.random.randint(20, 89),
    	'Gender': np.random.choice(['Male', 'Female']),
    	'Race': np.random.choice(['White', 'Black', 'Asian', 'Other']),
    	'Region': np.random.choice(['North America', 'Europe', 'Asia', 'Other']),
    	'Urban_or_Rural': np.random.choice(['Urban', 'Rural']),
    	'Socioeconomic_Status': np.random.choice(['Low', 'Middle', 'High']),
   	 
    	# Medical History
    	'Family_History': np.random.choice(['Yes', 'No'], p=[0.3, 0.7]),
    	'Previous_Cancer_History': np.random.choice(['Yes', 'No'], p=[0.1, 0.9]),
   	 
    	# Cancer Characteristics
    	'Stage_at_Diagnosis': np.random.choice(['I', 'II', 'III', 'IV']),
    	'Tumor_Aggressiveness': np.random.choice(['Low', 'Medium', 'High']),
   	 
    	# Screening/Prevention
    	'Colonoscopy_Access': np.random.choice(['Yes', 'No']),
    	'Screening_Regularity': np.random.choice(['Regular', 'Irregular', 'Never']),
   	 
    	# Lifestyle Factors
    	'Diet_Type': np.random.choice(['Western', 'Balanced', 'Traditional']),
    	'BMI': np.round(np.random.uniform(18.5, 40.0), 1),
    	'Physical_Activity_Level': np.random.choice(['Low', 'Medium', 'High']),
    	'Smoking_Status': np.random.choice(['Never', 'Former', 'Current'], p=[0.5, 0.3, 0.2]),
    	'Alcohol_Consumption': np.random.choice(['Low', 'Medium', 'High']),
    	'Red_Meat_Consumption': np.random.choice(['Low', 'Medium', 'High']),
    	'Fiber_Consumption': np.random.choice(['Low', 'Medium', 'High']),
   	 
    	# Healthcare Access
    	'Insurance_Coverage': np.random.choice(['Yes', 'No']),
    	'Time_to_Diagnosis': np.random.choice(['Timely', 'Delayed']),
    	'Treatment_Access': np.random.choice(['Good', 'Limited']),
   	 
    	# Treatments (initialized to No for simulation)
    	'Chemotherapy_Received': 'No',
    	'Radiotherapy_Received': 'No',
    	'Surgery_Received': 'No',
   	 
    	# Follow-up (not needed for prediction)
    	'Follow_Up_Adherence': np.random.choice(['Good', 'Poor']),
    	'Recurrence': np.random.choice(['Yes', 'No']),
    	'Time_to_Recurrence': np.random.randint(0, 59),
	}
    
	return pd.DataFrame([patient])

def test_recommendation():
	test_patient = generate_random_patient()
	print("Patient:")
	print(test_patient.transpose().to_string(header=False))
	for strategy, model_set in [('SMOTE', models_smote), ('Weighted', models_weighted)]:
		for name, model in model_set.items():
			combo, prob = recommend_treatment(test_patient, model, preprocessor, feature_names, features.columns)
			print(f"{strategy}-{name}: {combo} -> {prob:.2%}")
			save_result(test_patient, f"{strategy}_{name}", combo, prob)
               
# Salvar o modelo treinado
joblib.dump(rf_model_w, "modelo_colorectal_rf.pkl")  
joblib.dump(nb_model_w, "modelo_colorectal_nb.pkl")  
joblib.dump(xgb_model_w, "modelo_colorectal_xgb.pkl")  

# Salvar o pré-processador
joblib.dump(preprocessor, "preprocessor.pkl")

# ----------------------
# 7. RUN TEST
# ----------------------

test_recommendation()

