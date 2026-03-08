import joblib
import pandas as pd
import numpy as np

# Load the saved assets
model = joblib.load('icu_risk_model.pkl')
scaler = joblib.load('scaler.pkl')

def get_patient_risk(vitals_dict):
    """
    vitals_dict should look like:
    {
        'Age': 65, 'Gender': 1, 'Heart Rate (bpm)': 110, 'Oxygen Saturation (%)': 92,
        'Systolic BP (mmHg)': 145, 'Diastolic BP (mmHg)': 88, 'Body Temperature (°C)': 38.5,
        'Fall Detection': 0, 'Derived_HRV': 0.08, 'Derived_MAP': 107, 'Derived_Pulse_Pressure': 57
    }
    """
    # 1. Convert dict to DataFrame
    input_data = pd.DataFrame([vitals_dict])
    
    # 2. Scale the data using the SAME scaler from training
    scaled_data = scaler.transform(input_data)
    
    # 3. Predict probability and class
    risk_prob = model.predict_proba(scaled_data)[0][1] # Probability of "High Risk"
    risk_class = model.predict(scaled_data)[0]
    
    # 4. Map back to labels (assuming High Risk was encoded as 0 and Low Risk as 1)
    status = "High Risk" if risk_class == 0 else "Low Risk"
    
    return {
        "status": status,
        "probability": round(float(risk_prob), 2)
    }

# --- TEST IT ---
test_patient = {
    'Age': 70, 'Gender': 1, 'Heart Rate (bpm)': 120, 'Oxygen Saturation (%)': 88,
    'Systolic BP (mmHg)': 160, 'Diastolic BP (mmHg)': 95, 'Body Temperature (°C)': 39.0,
    'Fall Detection': 1, 'Derived_HRV': 0.04, 'Derived_MAP': 116, 'Derived_Pulse_Pressure': 65
}
print(get_patient_risk(test_patient))