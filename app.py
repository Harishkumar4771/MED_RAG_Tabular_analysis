import pandas as pd
import joblib
from flask import Flask, render_template, request
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Global variables to hold our artifacts
model = None
scaler = None

def train_and_save():
    df = pd.read_csv('df_encoded.csv')
    X = df.drop("readmitted", axis=1)
    y = df["readmitted"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    lr = LogisticRegression()
    rf = RandomForestClassifier(n_estimators=100)
    xgb = XGBClassifier()
    ensemble = VotingClassifier(
        estimators=[("lr", lr), ("rf", rf), ("xgb", xgb)], 
        voting="hard"
    )
    ensemble.fit(X_scaled, y)
    joblib.dump(ensemble, 'model.pkl', compress=3)
    joblib.dump(scaler, 'scaler.pkl', compress=3)
    return ensemble, scaler

# Initialize model on startup
try:
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
except:
    model, scaler = train_and_save()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/form")
def prediction_form():
    return render_template("form.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get form data as dictionary
    input_data = request.form.to_dict()
    
    # 1. Age Mapping
    age_map = {
        '[40-50)': 45,
        '[50-60)': 55,
        '[60-70)': 65,
        '[70-80)': 75,
        '[80-90)': 85,
        '[90-100)': 95
    }
    age_val = age_map.get(input_data.get('age'), 55)
    
    # 2. Extract numeric values from form
    try:
        time_in_hospital = float(input_data.get('time_in_hospital', 0))
        n_lab_procedures = float(input_data.get('n_lab_procedures', 0))
        n_procedures = float(input_data.get('n_procedures', 0))
        n_medications = float(input_data.get('n_medications', 0))
        n_outpatient = float(input_data.get('n_outpatient', 0))
        n_inpatient = float(input_data.get('n_inpatient', 0))
        n_emergency = float(input_data.get('n_emergency', 0))
    except ValueError:
        return "Invalid numeric input provided."

    # 3. Calculate Derived Features
    total_visits = n_inpatient + n_outpatient + n_emergency
    visit_severity = 3 * n_inpatient + 2 * n_emergency + 1 * n_outpatient
    polypharmacy = 1 if n_medications >= 5 else 0
    meds_per_day = n_medications / time_in_hospital if time_in_hospital > 0 else 0
    long_stay = 1 if time_in_hospital > 7 else 0

    # 4. Prepare the base DataFrame with numeric/derived features
    processed_data = {
        'age': age_val,
        'time_in_hospital': time_in_hospital,
        'n_lab_procedures': n_lab_procedures,
        'n_procedures': n_procedures,
        'n_medications': n_medications,
        'n_outpatient': n_outpatient,
        'n_inpatient': n_inpatient,
        'n_emergency': n_emergency,
        'total_visits': total_visits,
        'visit_severity': visit_severity,
        'polypharmacy': polypharmacy,
        'meds_per_day': meds_per_day,
        'long_stay': long_stay
    }
    
    input_df = pd.DataFrame([processed_data])

    # 5. One-Hot Encoding for Categorical Features
    # Categories and their expected values from df_encoded.csv
    categorical_fields = {
        'medical_specialty': ['Cardiology', 'Emergency/Trauma', 'Family/GeneralPractice', 'InternalMedicine', 'Missing', 'Other', 'Surgery'],
        'diag_1': ['Circulatory', 'Diabetes', 'Digestive', 'Injury', 'Missing', 'Musculoskeletal', 'Other', 'Respiratory'],
        'diag_2': ['Circulatory', 'Diabetes', 'Digestive', 'Injury', 'Missing', 'Musculoskeletal', 'Other', 'Respiratory'],
        'diag_3': ['Circulatory', 'Diabetes', 'Digestive', 'Injury', 'Missing', 'Musculoskeletal', 'Other', 'Respiratory'],
        'glucose_test': ['high', 'no', 'normal'],
        'A1Ctest': ['high', 'no', 'normal'],
        'change': ['no', 'yes'],
        'diabetes_med': ['no', 'yes']
    }

    for field, categories in categorical_fields.items():
        user_choice = input_data.get(field)
        for cat in categories:
            col_name = f"{field}_{cat}"
            input_df[col_name] = 1.0 if user_choice == cat else 0.0

    # 6. Ensure all columns match df_encoded.csv exactly
    df_train = pd.read_csv("df_encoded.csv")
    feature_cols = df_train.drop("readmitted", axis=1).columns.tolist()
    
    # Reorder and fill any missing columns (shouldn't be any now, but for safety)
    for col in feature_cols:
        if col not in input_df.columns:
            input_df[col] = 0.0
            
    input_df = input_df[feature_cols]

    # 7. Scale and Predict
    try:
        scaled_data = scaler.transform(input_df)
        prediction = model.predict(scaled_data)[0]
        
        proba = None
        if hasattr(model, "predict_proba"):
            proba_arr = model.predict_proba(scaled_data)[0]
            # Use probability of class 1 (readmitted)
            proba = proba_arr[1] if len(proba_arr) > 1 else proba_arr[0]
        
        print(f"Prediction: {prediction}, Proba: {proba}")
        print("Input Features Sample:")
        print(input_df.iloc[0].to_dict())
            
        return render_template("result.html", 
                               prediction=int(prediction), 
                               probability=proba,
                               glucose_test=input_data.get('glucose_test'),
                               A1Ctest=input_data.get('A1Ctest'))
    except Exception as e:
        print(f"Error: {e}")
        return f"Error during prediction: {str(e)}"
if __name__ == "__main__":
    app.run(debug=True)