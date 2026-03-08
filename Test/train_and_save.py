import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

# 1. Load your dataset
df = pd.read_csv('ICU_Preprocessed_Data.csv')

# 2. Preprocess
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])
df['Fall Detection'] = le.fit_transform(df['Fall Detection'])
df['Risk_Target'] = le.fit_transform(df['Risk Category'])

# 3. Features & Target (Matching your capstone architecture)
features = ['Age', 'Gender', 'Heart Rate (bpm)', 'Oxygen Saturation (%)', 
            'Systolic BP (mmHg)', 'Diastolic BP (mmHg)', 'Body Temperature (°C)', 
            'Fall Detection', 'Derived_HRV', 'Derived_MAP', 'Derived_Pulse_Pressure']
X = df[features]
y = df['Risk_Target']

# 4. Scale and Train
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5)
model.fit(X_train, y_train)

# 5. GENERATE THE FILES
joblib.dump(model, 'icu_risk_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Success! 'icu_risk_model.pkl' and 'scaler.pkl' are now in your folder.")