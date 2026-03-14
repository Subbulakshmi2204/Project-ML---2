import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

st.set_page_config(page_title="Urban Mobility Clustering", layout="wide")

st.title("🚦 Urban Mobility Pattern Discovery")

# Load dataset
data = pd.read_csv("mobility_data.csv")

# Show dataset
st.subheader("Dataset")
st.dataframe(data)

# Convert all columns to numeric
data = data.apply(pd.to_numeric, errors='coerce')

# Remove rows with missing values
data = data.dropna()

# Select features
X = data[['travel_time','distance_km','trip_frequency','vehicle_type']]

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# KMeans
st.subheader("KMeans Clustering")

kmeans = KMeans(n_clusters=3, random_state=42)
k_labels = kmeans.fit_predict(X_scaled)

fig1, ax1 = plt.subplots()
ax1.scatter(X_pca[:,0], X_pca[:,1], c=k_labels, cmap="viridis")
ax1.set_title("KMeans Clusters")
st.pyplot(fig1)

# DBSCAN
st.subheader("DBSCAN Clustering")

dbscan = DBSCAN(eps=1.5, min_samples=2)
d_labels = dbscan.fit_predict(X_scaled)

fig2, ax2 = plt.subplots()
ax2.scatter(X_pca[:,0], X_pca[:,1], c=d_labels, cmap="plasma")
ax2.set_title("DBSCAN Clusters")
st.pyplot(fig2)

# Hierarchical
st.subheader("Hierarchical Clustering")

hier = AgglomerativeClustering(n_clusters=3)
h_labels = hier.fit_predict(X_scaled)

fig3, ax3 = plt.subplots()
ax3.scatter(X_pca[:,0], X_pca[:,1], c=h_labels, cmap="rainbow")
ax3.set_title("Hierarchical Clusters")
st.pyplot(fig3)
