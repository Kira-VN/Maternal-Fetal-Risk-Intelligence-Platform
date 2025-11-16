# maternal_training.py

import warnings
warnings.simplefilter(action="ignore")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error
import pickle

data_path = r"E:\Major\Maternal Health Risk Data Set.csv"  
m_df = pd.read_csv(data_path)

risk_mapping = {"low risk": 0, "mid risk": 1, "high risk": 2}
m_df["RiskLevel"] = m_df["RiskLevel"].map(risk_mapping)

if "SystolicBP" in m_df.columns:
    m_df = m_df.drop(["SystolicBP"], axis=1)

m_df = m_df[m_df.HeartRate != 7]

feature_columns = ["Age", "DiastolicBP", "BS", "BodyTemp", "HeartRate"]
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(m_df.drop(["RiskLevel"], axis=1)), columns=feature_columns)
y = m_df["RiskLevel"]

pickle.dump(scaler, open("maternal_scaler.sav", "wb"))
print("Scaler saved as 'maternal_scaler.sav'")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

gbc = GradientBoostingClassifier(
    learning_rate=0.5,
    loss="deviance",
    max_depth=10,
    n_estimators=100,
    subsample=1,
    random_state=42
)
gbc_mod = gbc.fit(X_train, y_train)

y_pred = gbc_mod.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
train_score = gbc_mod.score(X_train, y_train)
test_score = gbc_mod.score(X_test, y_test)

print("\nGradient Boosting Classifier Metrics:")
print(f"Train Accuracy: {train_score:.3f}")
print(f"Test Accuracy: {test_score:.3f}")
print(f"MSE: {mse:.3f}, RMSE: {rmse:.3f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

pickle.dump(gbc_mod, open("finalized_maternal_model.sav", "wb"))
print("Model saved as 'finalized_maternal_model.sav'")
