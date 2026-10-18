# Pima Indians Diabetes Analysis (Task 4)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Create a simple sample dataset simulating diabetes diagnostic data
# Features: Glucose, BMI, Age
# Target: Outcome (0 = No Diabetes, 1 = Has Diabetes)
data = {
    'Glucose': [148, 85, 183, 89, 137, 116, 78, 125],
    'BMI': [33.6, 26.6, 23.3, 28.1, 43.1, 25.6, 31.0, 35.2],
    'Age': [50, 31, 32, 21, 33, 30, 26, 54],
    'Outcome': [1, 0, 1, 0, 1, 0, 1, 1]
}

df = pd.DataFrame(data)
print("Diabetes Dataset Preview:")
print(df.head())

# 2. Define Features (X) and Target (y)
X = df[['Glucose', 'BMI', 'Age']]
y = df['Outcome']

# 3. Split the data into Training and Testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 4. Train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# 5. Make a prediction for a new patient 
# (e.g., Glucose=150, BMI=32.5, Age=45)
new_patient = pd.DataFrame([[150, 32.5, 45]], columns=['Glucose', 'BMI', 'Age'])
prediction = model.predict(new_patient)

result = "Has Diabetes" if prediction[0] == 1 else "Does Not Have Diabetes"
print(f"\nPrediction for the new patient: {result}.")
