import openpyxl
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Sidebar navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", [
    "Preview Data", "EDA & Visualization", "Model Building", "Model Evaluation", "Confusion Matrix", "Prediction App"
])
st.title("Upload an Excel File")
uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx", "xls"])
if uploaded_file is not None:
    data = pd.read_excel(uploaded_file, engine="openpyxl")
    data.columns = data.columns.str.strip() 
if section == "Preview Data":
        st.write("### Preview of Uploaded Dataset")
        st.dataframe(data)
      

if section == "EDA & Visualization":
        st.title("Exploratory Data Analysis")
        st.write("### Data Info")
        st.write(data.info())
        st.write("### First 5 rows")
        st.write(data.head())
        st.write("### Checking for missing values")
        st.write(data.isnull().sum())
        
        # Visualize class distribution (Bankruptcy vs Non-Bankruptcy)
        st.write("### Class Distribution (Bankruptcy vs Non-Bankruptcy)")
        plt.figure(figsize=(6, 4))
        sns.histplot(x='class', data=data)
        plt.title('Class Distribution (Bankruptcy vs Non-Bankruptcy)')
        st.pyplot()

# Visualizing the distribution of individual features
        st.write("### Feature Distributions")
        plt.figure(figsize=(10, 6))
        sns.histplot(data['industrial_risk'], kde=True, bins=10, color='green', label='Industrial Risk')
        sns.histplot(data['management_risk'], kde=True, bins=10, color='green', label='Management Risk')
        sns.histplot(data['financial_flexibility'], kde=True, bins=10, color='red', label='Financial Flexibility')
        sns.histplot(data['credibility'], kde=True, bins=10, color='purple', label='Credibility')
        sns.histplot(data['competitiveness'], kde=True, bins=10, color='orange', label='Competitiveness')
        sns.histplot(data['operating_risk'], kde=True, bins=10, color='yellow', label='Operating Risk')
        plt.title('Feature Distributions')
        st.pyplot()

# Boxplots for feature distribution
        st.write("### Boxplot of Features")
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=data[['industrial_risk', 'management_risk', 'financial_flexibility',
                        'credibility', 'competitiveness', 'operating_risk']])
        plt.title('Boxplot of Features')
        st.pyplot() 
        # Pairplot visualization
        st.write("### Pairplot of Features")
        plt.figure()
        sns.pairplot(data)
        st.pyplot()
        
        # Heatmap
        st.write("### Correlation Heatmap")
        plt.figure(figsize=(10, 8))
        sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
        st.pyplot()
    
if section == "Model Building":
        st.title("Model Training")
        
        X = data.drop(columns=['class'])
        y = data['class']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        models = {
            "Random Forest": RandomForestClassifier(random_state=42),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Logistic Regression": LogisticRegression(random_state=42),
            "SVM": SVC(random_state=42)
        }
        
        trained_models = {}
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            trained_models[name] = model
        
        joblib.dump(trained_models["SVM"], 'best_bankruptcy_model.pkl')
        joblib.dump(scaler, 'scaler.pkl')
        st.write("Model training complete!")
    
if section == "Model Evaluation":
        st.title("Model Evaluation")
        
        def evaluate_model(model_name, model):
            y_pred = model.predict(X_test_scaled)
            st.write(f"### {model_name} Model")
            st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
            st.write("Classification Report:")
            st.text(classification_report(y_test, y_pred))
        
        for name, model in trained_models.items():
            evaluate_model(name, model)
    
if section == "Confusion Matrix":
        st.title("Confusion Matrix")
        
        def plot_confusion_matrix(y_true, y_pred, title):
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(6, 4))
            sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=['Non-Bankruptcy', 'Bankruptcy'], yticklabels=['Non-Bankruptcy', 'Bankruptcy'])
            plt.title(f"{title} Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            st.pyplot()
        
        for name, model in trained_models.items():
            y_pred = model.predict(X_test_scaled)
            plot_confusion_matrix(y_test, y_pred, name)
    
if section == "Prediction App":
        st.title("Bankruptcy Prediction App")
        
        model = joblib.load('best_bankruptcy_model.pkl')
        scaler = joblib.load('scaler.pkl')
        
        industrial_risk = st.selectbox("Industrial Risk", [0, 0.5, 1])
        management_risk = st.selectbox("Management Risk", [0, 0.5, 1])
        financial_flexibility = st.selectbox("Financial Flexibility", [0, 0.5, 1])
        credibility = st.selectbox("Credibility", [0, 0.5, 1])
        competitiveness = st.selectbox("Competitiveness", [0, 0.5, 1])
        operating_risk = st.selectbox("Operating Risk", [0, 0.5, 1])
        
        input_features = np.array([industrial_risk, management_risk, financial_flexibility, credibility, competitiveness, operating_risk]).reshape(1, -1)
        input_features_scaled = scaler.transform(input_features)
        
        prediction = model.predict(input_features_scaled)
        result = "likely to go bankrupt" if prediction == 0 else "not likely to go bankrupt"
        st.write(f"The company is **{result}**.")
