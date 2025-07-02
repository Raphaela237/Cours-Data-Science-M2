import streamlit as st
import joblib
import pandas as pd

# Charger le modèle entraîné
model = joblib.load("titanic_pipeline.pkl")

# Interface utilisateur
st.title("🛳️ Titanic Survival Prediction")

Pclass = st.selectbox("Classe (Pclass)", [1, 2, 3])
Sex = st.selectbox("Sexe", ["male", "female"])
Age = st.slider("Âge", 0, 100, 30)
Fare = st.slider("Tarif du billet", 0.0, 500.0, 32.0)
Embarked = st.selectbox("Port d’embarquement", ["S", "C", "Q"])

if st.button("Prédire la survie"):
    X_input = pd.DataFrame([[Pclass, Sex, Age, Fare, Embarked]],
                           columns=["Pclass", "Sex", "Age", "Fare", "Embarked"])
    prediction = model.predict(X_input)
    result = "🟢 Survécu" if prediction[0] == 1 else "🔴 N’a pas survécu"
    st.success(f"Prédiction : {result}")
