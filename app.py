import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Page configuration
st.set_page_config(
    page_title="Urban Mobility Clustering Dashboard",
    page_icon="🚦",
    layout="wide"
)

st.title("🚦 Urban Mobility Pattern Discovery")
st.markdown("Analyze transportation behavior using clustering algorithms.")

# Load dataset
data = pd.read_csv("mobility_data.csv")

# Sidebar controls
st.sidebar.header("Clustering Settings")

k_clusters = st.sidebar.slider("Select number of clusters for KMeans", 2, 6, 3)

# Dataset preview
st.subheader("Dataset Preview")
st.dataframe(data)

# Feature selection
X = data[['travel_time','distance_km','trip_frequency','vehicle_type']]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Metrics
col1, col2, col3 = st.columns(3)

col1.metric("Total Trips", len(data))
col2.metric("Average Distance", round(data["distance_km"].mean(),2))
col3.metric("Average Travel Time", round(data["travel_time"].mean(),2))

# Tabs for algorithms
tab1, tab2, tab3 = st.tabs(["K-Means Clustering","DBSCAN Clustering","Hierarchical Clustering"])

# ---------------- KMeans ----------------

with tab1:

    st.subheader("K-Means Clustering")

    kmeans = KMeans(n_clusters=k_clusters, random_state=42)
    labels = kmeans.fit_predict(X_scaled)

    fig, ax = plt.subplots()

    scatter = ax.scatter(
        X_pca[:,0],
        X_pca[:,1],
        c=labels,
        cmap="viridis"
    )

    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.set_title("K-Means Clustering Visualization")

    st.pyplot(fig)

# ---------------- DBSCAN ----------------

with tab2:

    st.subheader("DBSCAN Clustering")

    dbscan = DBSCAN(eps=1.5, min_samples=2)
    labels = dbscan.fit_predict(X_scaled)

    fig, ax = plt.subplots()

    scatter = ax.scatter(
        X_pca[:,0],
        X_pca[:,1],
        c=labels,
        cmap="plasma"
    )

    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.set_title("DBSCAN Clustering Visualization")

    st.pyplot(fig)

# ---------------- Hierarchical ----------------

with tab3:

    st.subheader("Hierarchical Clustering")

    hierarchical = AgglomerativeClustering(n_clusters=3)
    labels = hierarchical.fit_predict(X_scaled)

    fig, ax = plt.subplots()

    scatter = ax.scatter(
        X_pca[:,0],
        X_pca[:,1],
        c=labels,
        cmap="rainbow"
    )

    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.set_title("Hierarchical Clustering Visualization")

    st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("Machine Learning Mini Project | Clustering & Dimensionality Reduction")
