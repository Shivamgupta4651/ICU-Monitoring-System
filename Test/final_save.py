import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

# --- STEP 1: DEFINING THE DATA ---
df = pd.read_csv('ICU_Preprocessed_Data.csv')
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])
df['Fall Detection'] = le.fit_transform(df['Fall Detection'])
df['Risk_Target'] = le.fit_transform(df['Risk Category'])

features = ['Age', 'Gender', 'Heart Rate (bpm)', 'Oxygen Saturation (%)', 
            'Systolic BP (mmHg)', 'Diastolic BP (mmHg)', 'Body Temperature (°C)', 
            'Fall Detection', 'Derived_HRV', 'Derived_MAP', 'Derived_Pulse_Pressure']
X = df[features]
y = df['Risk_Target']

# --- STEP 2: DEFINING THE SCALER ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# --- STEP 3: DEFINING THE MODEL ---
model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5)
model.fit(X_train, y_train)

# --- STEP 4: SAVING (This will now work because model/scaler are defined above) ---
joblib.dump(model, 'icu_risk_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("!!! HANDOVER READY !!!")
print("Files 'icu_risk_model.pkl' and 'scaler.pkl' are now officially saved.")