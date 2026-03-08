import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 1. Load the Super Polished Numeric Dataset (v3)
try:
    df = pd.read_csv('ICU_Preprocessed_Data_v3.csv')
    print(f"Success: Loaded {len(df)} rows of balanced data.")
except FileNotFoundError:
    print("Error: ICU_Preprocessed_Data_v3.csv not found!")
    exit()

# 2. Define Features and Target
# Note: Fall Detection agar v3 mein nahi hai toh feature list se hata dena
features = ['Age', 'Gender', 'Heart Rate (bpm)', 'Oxygen Saturation (%)', 
            'Systolic BP (mmHg)', 'Diastolic BP (mmHg)', 'Body Temperature (°C)', 
            'Derived_HRV', 'Derived_MAP', 'Derived_Pulse_Pressure']

X = df[features]
y = df['Risk Category']

# 3. Scaling (Very important for consistency)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Train-Test Split (80% Training, 20% Testing)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 5. XGBoost Training
model = XGBClassifier(
    n_estimators=100, 
    learning_rate=0.1, 
    max_depth=5, 
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)
model.fit(X_train, y_train)

# 6. Save Assets
joblib.dump(model, 'icu_risk_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# 7. Evaluation & Visuals
y_pred = model.predict(X_test)
print("\n--- FINAL CLASSIFICATION REPORT ---")
print(classification_report(y_test, y_pred))

# Correlation Heatmap for Report
plt.figure(figsize=(12, 10))
sns.heatmap(df[features + ['Risk Category']].corr(), annot=True, cmap='RdYlGn', fmt='.2f')
plt.title("Final Vitals Correlation (Balanced Data)")
plt.savefig('correlation_matrix.png')

# Feature Importance for Report
plt.figure(figsize=(10, 6))
feat_importances = pd.Series(model.feature_importances_, index=features)
feat_importances.nlargest(10).sort_values().plot(kind='barh', color='teal')
plt.title("Top Vitals Influencing ICU Risk")
plt.tight_layout()
plt.savefig('feature_importance.png')

print("\nMISSION ACCOMPLISHED: Final Model & Graphics Saved!")