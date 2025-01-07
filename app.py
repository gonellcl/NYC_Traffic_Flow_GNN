import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Load the data
@st.cache_resource
def load_data():
    return pd.read_csv("data/processed/node_features_with_predictions.csv")

data = load_data()

# Title and introduction
st.title("Traffic Anomaly Detection Dashboard")
st.markdown("""
This dashboard visualizes traffic anomalies using node-level predictions.
Start by selecting a **Month**, then refine the view by **Day of the Week** and **Hour**.
Anomalies are detected based on the difference between predictions and actual values.
""")

# Sidebar Filters
st.sidebar.header("Filter Options")

# Step 1: Select Month
month = st.sidebar.selectbox("Select Month:", options=sorted(data["month"].dropna().unique()), index=0)

# Filter by month
filtered_data = data[data["month"] == month]

# Step 2: Conditional Dropdown for Day of the Week
if st.sidebar.checkbox("Filter by Day of the Week"):
    day_of_week = st.sidebar.selectbox("Select Day of the Week:", options=sorted(filtered_data["day_of_week"].dropna().unique()))
    filtered_data = filtered_data[filtered_data["day_of_week"] == day_of_week]
else:
    day_of_week = None

# Step 3: Conditional Dropdown for Hour
if st.sidebar.checkbox("Filter by Hour"):
    hour = st.sidebar.selectbox("Select Hour:", options=sorted(filtered_data["hour"].dropna().unique()))
    filtered_data = filtered_data[filtered_data["hour"] == hour]
else:
    hour = None

# Anomaly detection threshold
threshold = st.sidebar.slider("Set Anomaly Threshold:", min_value=0.0, max_value=10.0, value=2.0, step=0.1)
# Display anomaly table
st.subheader("Anomaly Details")
st.markdown("View details of nodes identified as anomalies.")
print(filtered_data.dtypes)

# Ensure numeric types for truth and predictions
filtered_data["truth"] = pd.to_numeric(filtered_data["truth"], errors="coerce")
filtered_data["predictions"] = pd.to_numeric(filtered_data["predictions"], errors="coerce")

# Handle NaN values
filtered_data["truth"].fillna(0, inplace=True)
filtered_data["predictions"].fillna(0, inplace=True)


filtered_data["anomaly"] = abs(filtered_data["truth"] - filtered_data["predictions"]) > threshold

# Map Visualization
st.subheader("Traffic Anomalies Map")
st.markdown("Anomalies are highlighted on the map in red.")

fig = px.scatter_mapbox(
    filtered_data,
    lat="latitude_x",
    lon="longitude_x",
    color="anomaly",
    color_discrete_map={True: "red", False: "blue"},
    size=np.abs(filtered_data["truth"] - filtered_data["predictions"]),
    hover_data=["node_id", "spatial_cluster_x", "grid_cluster_x", "truth", "predictions"],
    mapbox_style="carto-positron",
    title="Traffic Anomalies Map",
    zoom=10,
    height=600
)
st.plotly_chart(fig, use_container_width=True)

# Display anomaly table
st.subheader("Anomaly Details")
st.markdown("View details of nodes identified as anomalies.")
print(filtered_data.dtypes)

# Ensure numeric types for truth and predictions
filtered_data["truth"] = pd.to_numeric(filtered_data["truth"], errors="coerce")
filtered_data["predictions"] = pd.to_numeric(filtered_data["predictions"], errors="coerce")

# Handle NaN values
filtered_data["truth"].fillna(0, inplace=True)
filtered_data["predictions"].fillna(0, inplace=True)

# Detect anomalies
filtered_data["anomaly"] = abs(filtered_data["truth"] - filtered_data["predictions"]) > threshold

anomalies = filtered_data[filtered_data["anomaly"]]
st.dataframe(anomalies[["node_id", "spatial_cluster_x", "grid_cluster_x", "truth", "predictions", "hour", "month", "day_of_week"]])

# Trends or summary statistics
st.subheader("Anomaly Trends")
st.markdown("""
Explore trends based on the selected filters and threshold.
""")

# Anomaly summary by spatial cluster
anomaly_summary = anomalies.groupby("spatial_cluster_x").size().reset_index(name="anomaly_count")
fig_bar = px.bar(
    anomaly_summary,
    x="spatial_cluster_x",
    y="anomaly_count",
    title="Anomaly Count by Spatial Cluster",
    labels={"spatial_cluster": "Spatial Cluster", "anomaly_count": "Anomaly Count"}
)
st.plotly_chart(fig_bar, use_container_width=True)
st.subheader("Traffic Volume Over Time")
time_trends = filtered_data.groupby(["hour", "day_of_week"])["truth"].mean().reset_index()
fig_line = px.line(
    time_trends,
    x="hour",
    y="truth",
    color="day_of_week",
    title="Average Traffic Volume by Hour and Day of Week",
    labels={"truth": "Average Volume", "hour": "Hour of Day", "day_of_week": "Day of Week"}
)
st.plotly_chart(fig_line, use_container_width=True)

st.subheader("Anomaly Heatmap")
fig_heatmap = px.density_mapbox(
    anomalies,
    lat="latitude_x",
    lon="longitude_x",
    z="anomaly",
    radius=10,
    mapbox_style="stamen-terrain",
    title="Anomaly Density Heatmap"
)

# Calculate anomaly score
filtered_data["anomaly_score"] = abs(filtered_data["truth"] - filtered_data["predictions"])

# Detect anomalies
filtered_data["anomaly"] = filtered_data["anomaly_score"] > threshold



st.markdown("### Download Filtered Data")
st.download_button(
    label="Download CSV",
    data=filtered_data.to_csv(index=False),
    file_name="filtered_traffic_anomalies.csv",
    mime="text/csv"
)
