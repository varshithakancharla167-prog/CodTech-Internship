# Pima Indians Diabetes Analysis (Task 4) 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# 1. Load Real Dataset
try:
    # Official Pima Dataset
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    columns = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigree','Age','Outcome']
    df = pd.read_csv(url, header=None, names=columns)
    print(f"Real Pima Dataset Loaded: {df.shape[0]} rows")
    
    # Data Cleaning - Replace 0 with median (very important for Pima)
    for col in ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']:
        df[col] = df[col].replace(0, df[col].median())

    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

except:
    # Fallback to your data
    data = {
        'Glucose': [148, 85, 183, 89, 137, 116, 78, 125, 110, 168, 100, 95],
        'BMI': [33.6, 26.6, 23.3, 28.1, 43.1, 25.6, 31.0, 35.2, 32.0, 38.0, 27.0, 29.0],
        'Age': [50, 31, 32, 21, 33, 30, 26, 54, 30, 45, 25, 40],
        'Outcome': [1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0]
    }
    df = pd.DataFrame(data)
    X = df[['Glucose', 'BMI', 'Age']]
    y = df['Outcome']

print("Dataset Preview:")
print(df.head())

# 2. Scaling - compulsory for Logistic Regression
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)

# 4. Train
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# 5. Evaluate - THIS IS THE MAIN OUTPUT
y_pred = model.predict(X_test)
print(f"\nAccuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 6. Prediction for new patient
# For real dataset we need 8 values
import numpy as np
if X.shape[1] == 8:
    # Example: 2 pregnancies, Glucose 150, BP 80, etc
    new_patient = np.array([[2, 150, 80, 35, 150, 32.5, 0.5, 45]])
    new_patient_scaled = scaler.transform(new_patient)
else:
    new_patient_scaled = scaler.transform([[150, 32.5, 45]])

prediction = model.predict(new_patient_scaled)
result = "Has Diabetes" if prediction[0] == 1 else "Does Not Have Diabetes"
print(f"\nPrediction for new patient (Glucose 150, BMI 32.5, Age 45): {result}")
