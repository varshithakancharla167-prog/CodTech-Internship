# Mall Customer Segmentation (Task 3) 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 1. Load Dataset - Try real file, fallback to your data
try:
    # If you downloaded Mall_Customers.csv from Kaggle
    df = pd.read_csv('Mall_Customers.csv')
    # Use the two columns
    X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
    # Rename for easy use
    X.columns = ['Annual_Income', 'Spending_Score']
except:
    # Your sample data
    data = {
        'Annual_Income': [15, 16, 17, 70, 78, 80, 120, 126, 130, 15, 18, 19, 70, 72, 75, 122, 125, 128],
        'Spending_Score': [39, 81, 6, 77, 75, 79, 15, 6, 12, 40, 80, 8, 76, 74, 78, 14, 8, 10]
    }
    X = pd.DataFrame(data)
    df = X.copy()

print("Customer Dataset Preview:")
print(df.head())

# 2. ELBOW METHOD - THIS IS COMPULSORY FOR TASK 3
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8,4))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow
