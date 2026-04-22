# -------- IMPORT LIBRARIES --------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score

# -------- LOAD DATA --------
df = pd.read_csv("E:\Customer Segmentation\Mall_Customers.csv")

print("Dataset Preview:\n", df.head())

# -------- DATA PREPROCESSING --------
df = df.drop(['CustomerID'], axis=1)
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})

# -------- FEATURE SELECTION --------
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# -------- SCALING --------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================================
# K-MEANS CLUSTERING
# =========================================
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df['KMeans'] = kmeans.fit_predict(X_scaled)

kmeans_score = silhouette_score(X_scaled, df['KMeans'])
print("\nKMeans Silhouette Score:", kmeans_score)

# =========================================
# DBSCAN CLUSTERING
# =========================================
dbscan = DBSCAN(eps=0.5, min_samples=5)
df['DBSCAN'] = dbscan.fit_predict(X_scaled)

# DBSCAN me -1 noise hota hai → remove for score
if len(set(df['DBSCAN'])) > 1:
    dbscan_score = silhouette_score(X_scaled, df['DBSCAN'])
    print("DBSCAN Silhouette Score:", dbscan_score)
else:
    print("DBSCAN Silhouette Score: Not valid (single cluster)")

# =========================================
# HIERARCHICAL CLUSTERING
# =========================================
hc = AgglomerativeClustering(n_clusters=5)
df['Hierarchical'] = hc.fit_predict(X_scaled)

hc_score = silhouette_score(X_scaled, df['Hierarchical'])
print("Hierarchical Silhouette Score:", hc_score)

# =========================================
# VISUALIZATION (FIXED ✅)
# =========================================

plt.figure(figsize=(18,5))

# -------- KMEANS --------
plt.subplot(1,3,1)
sns.scatterplot(
    x=X['Annual Income (k$)'],
    y=X['Spending Score (1-100)'],
    hue=df['KMeans'],
    palette='Set2'
)
plt.title("K-Means Clustering")

# -------- DBSCAN --------
plt.subplot(1,3,2)
sns.scatterplot(
    x=X['Annual Income (k$)'],
    y=X['Spending Score (1-100)'],
    hue=df['DBSCAN'],
    palette='Set1'
)
plt.title("DBSCAN Clustering")

# -------- HIERARCHICAL --------
plt.subplot(1,3,3)
sns.scatterplot(
    x=X['Annual Income (k$)'],
    y=X['Spending Score (1-100)'],
    hue=df['Hierarchical'],
    palette='coolwarm'
)
plt.title("Hierarchical Clustering")

plt.tight_layout()
plt.show()

# -------- SAVE OUTPUT --------
df.to_csv("final_segmented_customers.csv", index=False)

print("\n✅ Project Completed Successfully!")

