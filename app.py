import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Page setup
st.set_page_config(page_title="Urban Mobility Clustering Dashboard", layout="wide")

st.title("🚦 Urban Mobility Pattern Discovery using Clustering")

st.markdown("""
This dashboard analyzes **urban travel patterns** using Machine Learning clustering algorithms.

Algorithms used:
- K-Means Clustering
- DBSCAN
- Hierarchical Clustering
- PCA for dimensionality reduction
""")

# Load dataset
data = pd.read_csv("mobility_data.csv")

st.subheader("Dataset Preview")
st.dataframe(data)

# Convert to numeric
data = data.apply(pd.to_numeric, errors='coerce')

# Remove missing values
data = data.dropna()

# Feature selection
X = data[['travel_time','distance_km','trip_frequency','vehicle_type']]

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA reduction
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Sidebar settings
st.sidebar.header("Clustering Settings")
k_clusters = st.sidebar.slider("Select number of clusters (KMeans)", 2, 5, 3)

# ---------------- KMEANS ----------------

st.header("K-Means Clustering")

kmeans = KMeans(n_clusters=k_clusters, random_state=42)
data['KMeans_Cluster'] = kmeans.fit_predict(X_scaled)

st.subheader("Cluster Assignment Table")
st.dataframe(data)

# Cluster summary
st.subheader("Cluster Summary")

cluster_summary = data.groupby("KMeans_Cluster").mean()

st.dataframe(cluster_summary)

# PCA visualization
fig1, ax1 = plt.subplots()

scatter = ax1.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=data['KMeans_Cluster'],
    cmap="viridis"
)

ax1.set_xlabel("PCA Component 1")
ax1.set_ylabel("PCA Component 2")
ax1.set_title("K-Means Cluster Visualization")

st.pyplot(fig1)

# Bar chart for cluster distance
st.subheader("Average Travel Distance per Cluster")

fig2, ax2 = plt.subplots()

cluster_summary['distance_km'].plot(kind='bar', ax=ax2)

ax2.set_ylabel("Distance (km)")
ax2.set_title("Cluster Comparison")

st.pyplot(fig2)

# ---------------- DBSCAN ----------------

st.header("DBSCAN Clustering")

dbscan = DBSCAN(eps=1.5, min_samples=2)

data['DBSCAN_Cluster'] = dbscan.fit_predict(X_scaled)

st.write("Cluster labels (-1 represents outliers)")

st.dataframe(data[['travel_time','distance_km','trip_frequency','vehicle_type','DBSCAN_Cluster']])

fig3, ax3 = plt.subplots()

ax3.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=data['DBSCAN_Cluster'],
    cmap="plasma"
)

ax3.set_title("DBSCAN Cluster Visualization")

st.pyplot(fig3)

# ---------------- HIERARCHICAL ----------------

st.header("Hierarchical Clustering")

hier = AgglomerativeClustering(n_clusters=3)

data['Hierarchical_Cluster'] = hier.fit_predict(X_scaled)

st.dataframe(data[['travel_time','distance_km','trip_frequency','vehicle_type','Hierarchical_Cluster']])

fig4, ax4 = plt.subplots()

ax4.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=data['Hierarchical_Cluster'],
    cmap="rainbow"
)

ax4.set_title("Hierarchical Cluster Visualization")

st.pyplot(fig4)

# Footer
st.markdown("---")
st.markdown("Machine Learning Mini Project | Clustering & Dimensionality Reduction")
