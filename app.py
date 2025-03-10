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

st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", [
    "Preview Data", "EDA & Visualization", "Model Building", "Model Evaluation", "Confusion Matrix", "Model Comparison","Prediction App"
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
def evaluate_model(model_name, y_true, y_pred):
    st.write(f"### {model_name} Model Evaluation")
    st.write(f"**Accuracy:** {accuracy_score(y_true, y_pred):.4f}")
    st.write("**Classification Report:**")
    st.text(classification_report(y_true, y_pred))
    

   

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
   # Scaling the features
    
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
    st.title("Confusion Matrix")
    # Assuming your models are already trained and named model1, model2, model3, and model4

    y_pred1 = rf_model.predict(X_test)
    y_pred2 = dt_model.predict(X_test)
    y_pred3 = lr_model.predict(X_test)
    y_pred4 = svm_model.predict(X_test)
    cm1 = confusion_matrix(y_test, y_pred1)
    cm2 = confusion_matrix(y_test, y_pred2)
    cm3 = confusion_matrix(y_test, y_pred3)
    cm4 = confusion_matrix(y_test, y_pred4)
    st.write("Confusion Matrix for Random_forest_Model:")
    st.write(cm1)

    st.write("Confusion Matrix for Decision_tree_Model:")
    st.write(cm2)

    st.write("Confusion Matrix for Logistic_regression_Model:")
    st.write(cm3)

    st.write("Confusion Matrix for SVM_Model:")
    st.write(cm4)
    # Create confusion matrices
    
    # Set up the figure for subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))  # 2x2 grid, adjust size as needed

    # Plot Model 1 confusion matrix
    sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0], cbar=False)
    axes[0, 0].set_title("Random_forest_Model: Confusion Matrix")
    axes[0, 0].set_xlabel('Predicted Labels')
    axes[0, 0].set_ylabel('True Labels')

    # Plot Model 2 confusion matrix
    sns.heatmap(cm2, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1], cbar=False)
    axes[0, 1].set_title("Decision_tree_Model: Confusion Matrix")
    axes[0, 1].set_xlabel('Predicted Labels')
    axes[0, 1].set_ylabel('True Labels')

    # Plot Model 3 confusion matrix
    sns.heatmap(cm3, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0], cbar=False)
    axes[1, 0].set_title("Logistic_regression_Model: Confusion Matrix")
    axes[1, 0].set_xlabel('Predicted Labels')
    axes[1, 0].set_ylabel('True Labels')

    # Plot Model 4 confusion matrix
    sns.heatmap(cm4, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1], cbar=False)
    axes[1, 1].set_title("SVM_Model: Confusion Matrix")
    axes[1, 1].set_xlabel('Predicted Labels')
    axes[1, 1].set_ylabel('True Labels')

    # Adjust layout to avoid overlapping elements
    plt.tight_layout()

    # Show the plot
    st.pyplot()

if section == "Model Comparison":
    # Calculate accuracy for each model
    accuracy1 = accuracy_score(y_test, y_pred1)
    accuracy2 = accuracy_score(y_test, y_pred2)
    accuracy3 = accuracy_score(y_test, y_pred3)
    accuracy4 = accuracy_score(y_test, y_pred4)

# Store accuracy values in a list
    accuracies = [accuracy1, accuracy2, accuracy3, accuracy4]

# Model names (optional, for labels in the bar plot)
    model_names = ['Random_forest_Model', 'Decision_tree_Model', 'Logistic_regression_Model', 'SVM_Model']

# Create a bar plot
    plt.figure(figsize=(8, 6))
    plt.bar(model_names, accuracies, color=['blue', 'green', 'orange', 'red'])

# Add labels and title
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title('Comparison of Model Accuracies')

# Show the plot
    st.pyplot()


     

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
