# Mushroom Classification (Task 2) 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load Real Dataset (from UCI)
# If internet not working, your 8-row data will still work as fallback
try:
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
    columns = ['class','cap-shape','cap-surface','cap-color','bruises','odor','gill-attachment','gill-spacing','gill-size','gill-color','stalk-shape','stalk-root','stalk-surface-above-ring','stalk-surface-below-ring','stalk-color-above-ring','stalk-color-below-ring','veil-type','veil-color','ring-number','ring-type','spore-print-color','population','habitat']
    df = pd.read_csv(url, header=None, names=columns)
    print(f"Real Dataset Loaded: {df.shape[0]} rows")
    # Encode all columns to numbers
    df_encoded = df.apply(lambda x: pd.factorize(x)[0])
    X = df_encoded.drop('class', axis=1)
    y = df_encoded['class']
except:
    # Fallback to your sample data
    data = {
        'Cap_Shape': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        'Gill_Color': [0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1],
        'Odor': [0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1],
        'Class': [0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1]
    }
    df = pd.DataFrame(data)
    X = df[['Cap_Shape', 'Gill_Color', 'Odor']]
    y = df['Class']

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 3. Train
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# 4. Evaluate - VERY IMPORTANT
y_pred = model.predict(X_test)
print(f"\nAccuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 5. Prediction for new mushroom
sample_mushroom = X_test.iloc[[0]]
prediction = model.predict(sample_mushroom)
result = "Poisonous" if prediction[0] == 1 else "Edible"
print(f"\nPrediction for sample mushroom: {result}")
