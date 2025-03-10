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
def evaluate_model(model_name, y_true, y_pred):
    st.write(y_test)
    st.write(f"### {model_name} Model Evaluation")
    st.write(f"**Accuracy:** {accuracy_score(y_true, y_pred):.4f}")
    st.write("**Classification Report:**")
    st.text(classification_report(y_true, y_pred))
    st.write("**Confusion Matrix:**")
    st.text(confusion_matrix(y_true, y_pred))

st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", [
    "Preview Data", "EDA & Visualization", "Model Building", "Model Evaluation", "Confusion Matrix", "Prediction App"
])

st.title("Upload an Excel File")
uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx", "xls"])


if uploaded_file is not None:
    data = pd.read_excel(uploaded_file, engine="openpyxl")
    data.columns = data.columns.str.strip()  # Clean column names by stripping extra spaces
    X = data.drop(columns=['class'])  # Features
    y = data['class']  # Target variable
    # Split the data into training and testing sets (80% training, 20% testing)
    global X_train, X_test;
    X_train, X_test,y_train,y_test = train_test_split(X, y, test_size=0.2,random_state=42)

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
    data['class'] = data['class'].replace({'bankruptcy': 1, 'non-bankruptcy': 0})
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    st.pyplot()

    
if section == "Model Building":
    st.title("Model Building")
    

    # Scaling the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize models
    rf_model = RandomForestClassifier(random_state=42)
    dt_model = DecisionTreeClassifier(random_state=42)
    lr_model = LogisticRegression(random_state=42)
    svm_model = SVC(random_state=42)

    # Training the models
    rf_model.fit(X_train_scaled, y_train)
    dt_model.fit(X_train_scaled, y_train)
    lr_model.fit(X_train_scaled, y_train)
    svm_model.fit(X_train_scaled, y_train)
   
    st.write("Model training complete!")
   
    
if section == "Model Evaluation":
    st.title("Model Evaluation")
     # Predictions
    rf_pred = rf_model.predict(X_test_scaled)
    dt_pred = dt_model.predict(X_test_scaled)
    lr_pred = lr_model.predict(X_test_scaled)
    svm_pred = svm_model.predict(X_test_scaled)
    evaluate_model("Random Forest", y_test, rf_pred)
    evaluate_model("Decision Tree", y_test, dt_pred)
    evaluate_model("Logistic Regression", y_test, lr_pred)
    evaluate_model("SVM", y_test, svm_pred)
    # Save the best model (Random Forest in this case)
    joblib.dump(rf_model, 'best_bankruptcy_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    st.success("Best model and scaler saved successfully!")



if section == "Confusion Matrix":
    
    # Confusion Matrix
    st.write("**Confusion Matrix:**")
    cm = confusion_matrix(y_test, y_pred)
    st.text(cm)   
     # Visualize confusion matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Bankruptcy', 'Bankruptcy'], yticklabels=['Non-Bankruptcy', 'Bankruptcy'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix for {model_name}')
    st.pyplot(fig)
    

if section == "Prediction App":
    st.title("Bankruptcy Prediction App")

    # Load the best model and scaler
    model = joblib.load('best_bankruptcy_model.pkl')
    scaler = joblib.load('scaler.pkl')

    # User inputs for prediction
    industrial_risk = st.selectbox("Industrial Risk", [0, 0.5, 1])
    management_risk = st.selectbox("Management Risk", [0, 0.5, 1])
    financial_flexibility = st.selectbox("Financial Flexibility", [0, 0.5, 1])
    credibility = st.selectbox("Credibility", [0, 0.5, 1])
    competitiveness = st.selectbox("Competitiveness", [0, 0.5, 1])
    operating_risk = st.selectbox("Operating Risk", [0, 0.5, 1])

    # Prepare input features for prediction
    input_features = np.array([industrial_risk, management_risk, financial_flexibility, credibility, competitiveness, operating_risk]).reshape(1, -1)
    input_features_scaled = scaler.transform(input_features)

    # Make prediction
    prediction = model.predict(input_features_scaled)
    result = "likely to go bankrupt" if prediction == 0 else "not likely to go bankrupt"
    st.write(f"The company is **{result}**.")
