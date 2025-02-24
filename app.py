import openpyxl


# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import streamlit as st

import requests
from io import BytesIO

# GitHub Raw URL of the Excel file
GITHUB_URL = "https://github.com/shanthini1002/bankruptcy_project.git"

@st.cache_data
def load_excel(url):
    response = requests.get(url)
    file = BytesIO(response.content)
    return pd.read_excel(file, engine="openpyxl")

st.title("Streamlit App with Excel File from GitHub")

df = load_excel(GITHUB_URL)
    
    # Display the dataset
    st.write("### Preview of Uploaded Dataset")
    st.dataframe(data)

    
    
# EDA and Visualizations
st.write("Exploratory Data Analysis and Model Evaluation")

# Display the data info and first few rows
st.write("### Data Info")
st.write(data.info())
st.write("### First 5 rows of the data:")
st.write(data.head())
st.write("### Data Types")
st.write(data.dtypes)
st.write("### Number of rows & column")
st.write(data.shape)
st.write("### Column names")
st.write(data.columns)
st.write("### summary statistics")
st.write(data.describe())
st.write("### Checking for missing values")
st.write(data.isnull().sum())

# Pairplot to visualize relationships between features
st.write("### Pairplot of Features")
sns.pairplot(data)
plt.title('Pairplot of Features')
st.pyplot()

# Visualizing the distribution of individual features
st.write("### Feature Distributions")
plt.figure(figsize=(10, 6))
sns.histplot(data['industrial_risk'], kde=True, bins=10, color='blue', label='Industrial Risk')
sns.histplot(data['management_risk'], kde=True, bins=10, color='green', label='Management Risk')
sns.histplot(data['financial_flexibility'], kde=True, bins=10, color='red', label='Financial Flexibility')
sns.histplot(data['credibility'], kde=True, bins=10, color='purple', label='Credibility')
sns.histplot(data['competitiveness'], kde=True, bins=10, color='orange', label='Competitiveness')
sns.histplot(data['operating_risk'], kde=True, bins=10, color='yellow', label='Operating Risk')
plt.legend()
plt.title('Feature Distributions')
st.pyplot()

# Visualize class distribution (Bankruptcy vs Non-Bankruptcy)
st.write("### Class Distribution (Bankruptcy vs Non-Bankruptcy)")
plt.figure(figsize=(6, 4))
sns.countplot(x='class', data=data)
plt.title('Class Distribution (Bankruptcy vs Non-Bankruptcy)')
st.pyplot()

# Boxplots for feature distribution
st.write("### Boxplot of Features")
plt.figure(figsize=(12, 6))
sns.boxplot(data=data[['industrial_risk', 'management_risk', 'financial_flexibility',
                        'credibility', 'competitiveness', 'operating_risk']])
plt.title('Boxplot of Features')
st.pyplot()

# Heatmap of correlation between features
st.write("### Correlation Heatmap")
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=1)
plt.title('Correlation Heatmap')
st.pyplot()

# Data Preprocessing (Split Data and Scale Features)
X = data.drop(columns=['class'])  # Features
y = data['class']  # Target variable

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

# Predictions
rf_pred = rf_model.predict(X_test_scaled)
dt_pred = dt_model.predict(X_test_scaled)
lr_pred = lr_model.predict(X_test_scaled)
svm_pred = svm_model.predict(X_test_scaled)

# Evaluating models
def evaluate_model(model_name, y_true, y_pred):
    st.write(f"### {model_name} Model Evaluation")
    st.write(f"**Accuracy:** {accuracy_score(y_true, y_pred):.4f}")
    st.write("**Classification Report:**")
    st.text(classification_report(y_true, y_pred))
    st.write("**Confusion Matrix:**")
    st.text(confusion_matrix(y_true, y_pred))

# Evaluate each model
evaluate_model("Random Forest", y_test, rf_pred)
evaluate_model("Decision Tree", y_test, dt_pred)
evaluate_model("Logistic Regression", y_test, lr_pred)
evaluate_model("SVM", y_test, svm_pred)

# Save the best model (SVM in this case)
joblib.dump(svm_model, 'best_bankruptcy_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Streamlit UI to get inputs from the user
st.title("Bankruptcy Prediction App")
st.write("Enter the company attributes to predict bankruptcy likelihood.")

# Input features (drop-down for categorical values)
industrial_risk = st.selectbox("Industrial Risk", [0, 0.5, 1])
management_risk = st.selectbox("Management Risk", [0, 0.5, 1])
financial_flexibility = st.selectbox("Financial Flexibility", [0, 0.5, 1])
credibility = st.selectbox("Credibility", [0, 0.5, 1])
competitiveness = st.selectbox("Competitiveness", [0, 0.5, 1])
operating_risk = st.selectbox("Operating Risk", [0, 0.5, 1])

# Combine the input values into a feature vector
features = np.array([industrial_risk, management_risk, financial_flexibility,
                     credibility, competitiveness, operating_risk]).reshape(1, -1)

# Scale the features using the scaler
features_scaled = scaler.transform(features)

# Load the saved model
model = joblib.load('best_bankruptcy_model.pkl')

# Predict using the trained model
prediction = model.predict(features_scaled)

# Show the prediction result
if prediction == 0:
    st.write("The company is **not likely to go bankrupt**.")
else:
    st.write("The company is **likely to go bankrupt**.")
