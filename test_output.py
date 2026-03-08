import joblib
import numpy as np

# 1. Load the files you just generated
model = joblib.load('icu_risk_model.pkl')
scaler = joblib.load('scaler.pkl')

# 2. Create a "Dummy" High-Risk Patient
# Order: [Age, Gender, HR, SpO2, SystolicBP, DiastolicBP, Temp, Fall, HRV, MAP, PulsePressure]
test_patient = np.array([[75, 1, 120, 88, 160, 95, 38.5, 1, 0.04, 116, 65]])

# 3. Scale and Predict
scaled_patient = scaler.transform(test_patient)
prediction = model.predict(scaled_patient)

# 4. Show result
# (Note: In your dataset, High Risk is usually 0 and Low Risk is 1)
result = "High Risk" if prediction[0] == 0 else "Low Risk"
print(f"Prediction for Test Patient: {result}")