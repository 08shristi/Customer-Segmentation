

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Customer Segmentation", layout="wide")

# ---------- CLEAN CSS ----------
st.markdown("""
<style>
.card {

            
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)
# ---------- TITLE ----------
st.title("💖 Customer Segmentation Dashboard")

# ---------- SIDEBAR ----------
st.sidebar.header("⚙️ Controls")

algorithm = st.sidebar.selectbox(
    "Choose Algorithm",
    ["KMeans", "DBSCAN", "Hierarchical"]
)

k = st.sidebar.slider("Number of Clusters", 2, 10, 5)

# ---------- LOAD DATA ----------
df = pd.read_csv("Mall_Customers.csv")

# ---------- DATA PREVIEW ----------
st.markdown('<div class="card">', unsafe_allow_html=True)# html use karke card design kiya hai
st.subheader("📊 Dataset Preview")
st.dataframe(df.head())
st.markdown('</div>', unsafe_allow_html=True)

# ---------- PREPROCESSING ----------
df = df.drop(['CustomerID'], axis=1)
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})

X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------- MODEL ----------
if algorithm == "KMeans":
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
elif algorithm == "DBSCAN":
    model = DBSCAN(eps=0.5, min_samples=5)
else:
    model = AgglomerativeClustering(n_clusters=k)

df['Cluster'] = model.fit_predict(X_scaled)

# ---------- MAIN GRAPH ----------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader(f"🎨 {algorithm} Clustering")

fig, ax = plt.subplots()

sns.scatterplot(
    x=X['Annual Income (k$)'],
    y=X['Spending Score (1-100)'],
    hue=df['Cluster'],
    palette='Set2',
    ax=ax
)
#..........ELBOW METHOD..........#
st.pyplot(fig)
st.markdown('</div>', unsafe_allow_html=True)
st.subheader("📉 Elbow Method")

wcss = []
for i in range(1, 11):
    km = KMeans(n_clusters=i, random_state=42, n_init=10)
    km.fit(X_scaled)
    wcss.append(km.inertia_)

fig2, ax2 = plt.subplots()
ax2.plot(range(1,11), wcss, marker='o')
ax2.set_xlabel("Number of Clusters")
ax2.set_ylabel("WCSS")# Within Cluster Sum of Squares

st.pyplot(fig2)   # 🔥 MOST IMPORTANT

# ---------- ALL ALGORITHMS COMPARISON ----------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📊 Compare All Algorithms")

col1, col2, col3 = st.columns(3)

# KMeans
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df['KMeans'] = kmeans.fit_predict(X_scaled)

with col1:
    st.write("KMeans")
    fig1, ax1 = plt.subplots()
    sns.scatterplot(x=X['Annual Income (k$)'],
                    y=X['Spending Score (1-100)'],
                    hue=df['KMeans'], ax=ax1)
    st.pyplot(fig1)

# DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
df['DBSCAN'] = dbscan.fit_predict(X_scaled)

with col2:
    st.write("DBSCAN")
    fig2, ax2 = plt.subplots()
    sns.scatterplot(x=X['Annual Income (k$)'],
                    y=X['Spending Score (1-100)'],
                    hue=df['DBSCAN'], ax=ax2)
    st.pyplot(fig2)

# Hierarchical
hc = AgglomerativeClustering(n_clusters=5)
df['Hierarchical'] = hc.fit_predict(X_scaled)

with col3:
    st.write("Hierarchical")
    fig3, ax3 = plt.subplots()
    sns.scatterplot(x=X['Annual Income (k$)'],
                    y=X['Spending Score (1-100)'],
                    hue=df['Hierarchical'], ax=ax3)
    st.pyplot(fig3)

st.markdown('</div>', unsafe_allow_html=True)

# ---------- SUMMARY ----------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📈 Cluster Summary")

summary = df.groupby('Cluster').mean(numeric_only=True)
st.dataframe(summary)

st.markdown('</div>', unsafe_allow_html=True)

# ---------- METRICS ----------
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 👥 Customers")
    st.write(f"### {len(df)}")

with col2:
    st.markdown("### 📊 Clusters")
    st.write(f"### {len(set(df['Cluster']))}")

with col3:
    st.markdown("### 💰 Avg Spending")
    st.write(f"### {round(df['Spending Score (1-100)'].mean(), 2)}")
# ---------DOWNLOAD BUTTON---------
csv = df.to_csv(index=False).encode('utf-8')

st.download_button(
    "📥 Download Result CSV",
    csv,
    "customers_segmented.csv",
    "text/csv"
)


# ---------- FOOTER ----------
st.success("✅ Project Running Successfully!")

#----------BUSINESS INSIGHTS----------
st.subheader("🧠 Business Insights")

st.write("""
- High income & high spending customers → Premium segment  
- Low income & low spending → Budget segment  
- Medium groups → Target marketing  
- Helps improve sales strategy  
""")