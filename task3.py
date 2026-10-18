# Mall Customer Segmentation (Task 3)
import pandas as pd
from sklearn.cluster import KMeans

# 1. Create a simple sample dataset simulating customer features
# Features: Annual_Income (k$), Spending_Score (1-100)
data = {
    'Annual_Income': [15, 16, 17, 70, 78, 80, 120, 126, 130],
    'Spending_Score': [39, 81, 6, 77, 75, 79, 15, 6, 12]
}

df = pd.DataFrame(data)
print("Customer Dataset Preview:")
print(df.head())

# 2. Apply K-Means Clustering to group customers into 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(df[['Annual_Income', 'Spending_Score']])

print("\nCustomer Data with Assigned Clusters:")
print(df)
