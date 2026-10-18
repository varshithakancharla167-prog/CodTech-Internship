# Mushroom Classification (Task 2)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 1. Create a simple sample dataset simulating mushroom features
# Features: Cap_Shape (0=Bell, 1=Convex), Gill_Color (0=Brown, 1=White), Odor (0=Almond, 1=Foul)
# Target: Class (0 = Edible, 1 = Poisonous)
data = {
    'Cap_Shape': [0, 1, 0, 1, 0, 1, 0, 1],
    'Gill_Color': [0, 1, 0, 1, 0, 0, 1, 1],
    'Odor': [0, 1, 0, 1, 0, 1, 1, 1],
    'Class': [0, 1, 0, 1, 0, 1, 1, 1]  # 0: Edible, 1: Poisonous
}

df = pd.DataFrame(data)
print("Mushroom Dataset Preview:")
print(df.head())

# 2. Define Features (X) and Target (y)
X = df[['Cap_Shape', 'Gill_Color', 'Odor']]
y = df['Class']

# 3. Split the data into Training and Testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 4. Train a Decision Tree Classifier model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# 5. Make a prediction for a new mushroom
# Let's test a mushroom with Cap_Shape=1, Gill_Color=1, Odor=1
sample_mushroom = pd.DataFrame([[1, 1, 1]], columns=['Cap_Shape', 'Gill_Color', 'Odor'])
prediction = model.predict(sample_mushroom)

result = "Poisonous" if prediction[0] == 1 else "Edible"
print(f"\nPrediction for the test mushroom: The mushroom is {result}.")
