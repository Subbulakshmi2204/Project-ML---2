import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Page configuration
st.set_page_config(page_title="Urban Mobility Insights", layout="wide")

st.title("🚦 Urban Mobility Pattern Analysis")

st.markdown("""
This app analyzes **urban travel behavior** using Machine Learning clustering.

It groups travelers based on:
- Travel Time
- Travel Distance
- Trip Frequency
- Vehicle Type
""")

# Load dataset
data = pd.read_csv("mobility_data.csv")

# Convert to numeric
data = data.apply(pd.to_numeric, errors='coerce')
data = data.dropna()

# Features
X = data[['travel_time','distance_km','trip_frequency','vehicle_type']]

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Sidebar control
st.sidebar.header("Settings")
k = st.sidebar.slider("Select number of mobility groups", 2, 5, 3)

# KMeans clustering
kmeans = KMeans(n_clusters=k, random_state=42)
data['Cluster'] = kmeans.fit_predict(X_scaled)

# PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# ------------------ DATA OVERVIEW ------------------

st.header("📊 Dataset Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Total Trips", len(data))
col2.metric("Average Distance", round(data['distance_km'].mean(),2))
col3.metric("Average Travel Time", round(data['travel_time'].mean(),2))

# ------------------ CLUSTER VISUALIZATION ------------------

st.header("📍 Travel Pattern Groups")

fig, ax = plt.subplots()

scatter = ax.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=data['Cluster'],
)

ax.set_xlabel("Mobility Pattern 1")
ax.set_ylabel("Mobility Pattern 2")
ax.set_title("Traveler Groups Based on Similar Behavior")

st.pyplot(fig)

# ------------------ CLUSTER INTERPRETATION ------------------

st.header("🧠 What These Groups Mean")

summary = data.groupby("Cluster").mean()

for i in summary.index:

    st.subheader(f"Group {i}")

    st.write(
        f"""
        • Average Travel Time: **{round(summary.loc[i,'travel_time'],2)} minutes**  
        • Average Distance: **{round(summary.loc[i,'distance_km'],2)} km**  
        • Trip Frequency: **{round(summary.loc[i,'trip_frequency'],2)} trips**
        """
    )

# ------------------ DATA VIEW ------------------

with st.expander("View Full Dataset"):
    st.dataframe(data)

st.markdown("---")
st.caption("Machine Learning Mini Project – Clustering & Dimensionality Reduction")
