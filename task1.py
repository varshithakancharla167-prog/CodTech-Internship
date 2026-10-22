# Student Performance Tracker (Task 1) - Final Version
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Sample data
data = {
    'Study_Hours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Attendance_Percentage': [50, 60, 65, 70, 80, 85, 90, 95, 98, 100],
    'Exam_Score': [51, 58, 63, 72, 78, 85, 88, 92, 96, 99]
}

df = pd.DataFrame(data)
print("Dataset Preview:")
print(df)

X = df[['Study_Hours', 'Attendance_Percentage']]
y = df['Exam_Score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Evaluation - IMPORTANT FOR REPORT
y_pred = model.predict(X_test)
print(f"\nR2 Score: {r2_score(y_test, y_pred):.2f}")
print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")

# Prediction
new_student = pd.DataFrame([[7.5, 92]], columns=['Study_Hours', 'Attendance_Percentage'])
predicted_score = model.predict(new_student)
print(f"\nPredicted Exam Score for 7.5 hrs, 92% attendance: {predicted_score[0]:.2f}")

# Graph - IMPORTANT FOR LINKEDIN POST
plt.scatter(df['Study_Hours'], df['Exam_Score'])
plt.xlabel('Study Hours')
plt.ylabel('Exam Score')
plt.title('Study Hours vs Exam Score')
plt.show()
