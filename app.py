import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Page configuration
st.set_page_config(page_title="Urban Mobility Analytics", layout="wide")

st.title("🚦 Urban Mobility Pattern Analysis")
st.markdown(
"""
This application analyzes **urban travel behavior** using machine learning clustering.
Travelers are grouped based on:

• Travel Time  
• Distance Travelled  
• Trip Frequency  
• Vehicle Type
"""
)

# ---------------- DATASET UPLOAD ----------------

st.header("Upload Mobility Dataset")

uploaded_file = st.file_uploader("Upload CSV File")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
else:
    data = pd.read_csv("mobility_data.csv")

# Convert to numeric
data = data.apply(pd.to_numeric, errors='coerce')
data = data.dropna()

# ---------------- DATA OVERVIEW ----------------

st.header("Dataset Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Total Trips", len(data))
col2.metric("Average Distance (km)", round(data['distance_km'].mean(),2))
col3.metric("Average Travel Time (min)", round(data['travel_time'].mean(),2))

# ---------------- FEATURE SELECTION ----------------

X = data[['travel_time','distance_km','trip_frequency','vehicle_type']]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------- CLUSTER SETTINGS ----------------

st.sidebar.header("Clustering Settings")

k = st.sidebar.slider("Select Number of Clusters",2,5,3)

# KMeans clustering
kmeans = KMeans(n_clusters=k, random_state=42)
data['Cluster'] = kmeans.fit_predict(X_scaled)

# ---------------- PCA VISUALIZATION ----------------

st.header("Traveler Groups Visualization")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

fig, ax = plt.subplots()

scatter = ax.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=data['Cluster']
)

ax.set_xlabel("Mobility Pattern 1")
ax.set_ylabel("Mobility Pattern 2")
ax.set_title("Clusters of Travelers")

st.pyplot(fig)

# ---------------- CLUSTER DISTRIBUTION ----------------

st.header("Traveler Type Distribution")

cluster_counts = data['Cluster'].value_counts()

fig2, ax2 = plt.subplots()

ax2.pie(cluster_counts, labels=cluster_counts.index, autopct='%1.1f%%')

ax2.set_title("Distribution of Traveler Groups")

st.pyplot(fig2)

# ---------------- CLUSTER SUMMARY ----------------

st.header("Cluster Insights")

summary = data.groupby("Cluster").mean()

for i in summary.index:

    st.subheader(f"Traveler Group {i}")

    st.write(
        f"""
        • Average Travel Time: **{round(summary.loc[i,'travel_time'],2)} minutes**  
        • Average Distance: **{round(summary.loc[i,'distance_km'],2)} km**  
        • Trip Frequency: **{round(summary.loc[i,'trip_frequency'],2)} trips**
        """
    )

# ---------------- PREDICTION TOOL ----------------

st.sidebar.header("Traveler Group Prediction")

time = st.sidebar.number_input("Travel Time (minutes)",0,200)
distance = st.sidebar.number_input("Distance (km)",0,100)
frequency = st.sidebar.number_input("Trip Frequency",0,50)
vehicle = st.sidebar.number_input("Vehicle Type (encoded)",0,5)

if st.sidebar.button("Predict Traveler Group"):

    new_data = [[time,distance,frequency,vehicle]]

    new_scaled = scaler.transform(new_data)

    prediction = kmeans.predict(new_scaled)

    st.sidebar.success(f"This traveler belongs to Group {prediction[0]}")

# ---------------- DATA VIEW ----------------

with st.expander("View Full Dataset"):
    st.dataframe(data)

st.markdown("---")
st.caption("Machine Learning Mini Project | Clustering & Dimensionality Reduction")
