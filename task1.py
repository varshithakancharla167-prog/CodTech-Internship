# Student Performance Tracker (Task 1)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Sample data: Study hours and attendance vs exam score
data = {
    'Study_Hours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Attendance_Percentage': [50, 60, 65, 70, 80, 85, 90, 95, 98, 100],
    'Exam_Score': [51, 58, 63, 72, 78, 85, 88, 92, 96, 99]
}

df = pd.DataFrame(data)
print("Dataset Preview:")
print(df.head())

# Features and Target
X = df[['Study_Hours', 'Attendance_Percentage']]
y = df['Exam_Score']

# Train and Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction for a new student
new_student = pd.DataFrame([[7.5, 92]], columns=['Study_Hours', 'Attendance_Percentage'])
predicted_score = model.predict(new_student)

print(f"\nPredicted Exam Score: {predicted_score[0]:.2f}")
