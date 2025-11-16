# Maternal–Fetal Risk System

A Streamlit-based machine learning application that predicts **maternal pregnancy risk** and **fetal health status** using clinical data and **CTG (Cardiotocography) features**.  
The platform combines dual ML models, preprocessing pipelines, and interactive dashboards powered by government health data.

## 🚀 Features

- 🤰 **Maternal Risk Prediction**
  - Predicts Low / Medium / High pregnancy risk
  - Inputs: Age, Diastolic BP, Blood Sugar, Body Temperature, Heart Rate
  - Gradient Boosting model with **84% accuracy**

- 👶 **Fetal Health Prediction (CTG-Based)**
  - Classifies Normal / Suspect / Pathological
  - Uses 21 CTG features:
    - Accelerations
    - Variability
    - Decelerations
    - Uterine contractions
    - Histogram metrics
  - Gradient Boosting model with **93.4% accuracy**

- 📊 **Interactive Dashboard**
  - Live health data fetched from data.gov.in
  - Bubble and Pie charts using Plotly
  - State-wise maternal health insights

- 🧪 **Model Evaluation**
  - Accuracy, Precision, Recall, F1-Score
  - AUC (OvR)
  - Confusion Matrix Visualization

## 🛠️ Tech Stack

- Python 🐍  
- Streamlit 🌐  
- scikit-learn 🤖  
- Plotly 📈  
- Pandas & NumPy 🔢  
- Requests 🌍  
- Pickle (Model Serialization) 💾  

## 📂 Input Features

### Maternal Model
```
Age, DiastolicBP, BS, BodyTemp, HeartRate
```

### Fetal Model (CTG)
```
21 clinical CTG parameters including accelerations, variability, decelerations, contractions, histogram features
```

## 🧪 Setup Instructions

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/Maternal-Fetal-Risk-Intelligence-Platform.git
cd Maternal-Fetal-Risk-Intelligence-Platform

# 2. Install dependencies
pip install -r requirements.txt

# 3. Change paths of Model.sav in the Main.py file

# 4. Run the Streamlit application
streamlit run app.py
```

## 🔮 Future Improvements

- Add deep learning models (CNN/LSTM) for CTG waveform analysis  
- Add authentication and data history  
- Enhanced UI components  
- Cloud deployment (Azure/GCP)  
- Additional clinical risk modules  

---

> “Data-driven healthcare can enhance diagnosis, awareness, and timely intervention. This system is intended only for preliminary screening and decision support; it cannot diagnose conditions or replace professional medical evaluation. \"
