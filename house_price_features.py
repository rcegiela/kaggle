"""House Price Features"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.preprocessing import OrdinalEncoder
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# pylint: disable=line-too-long, invalid-name

PREFIX = 'clw_'

def cluster_locations(df, n=10, time_scale=1/36.5):
    """
    Cluster locations using KMeans based on latitude, longitude, and sale date.
    Creates approximately n-sized circular clusters.
    """
    df = df.copy()

    # Convert sale_date to datetime
    df[f'{PREFIX}sale_date'] = pd.to_datetime(df['sale_date'])

    # Convert to ordinal (days since a reference point)
    df[f'{PREFIX}sale_days'] = (df[f'{PREFIX}sale_date'] - pd.Timestamp("1970-01-01")) // pd.Timedelta("1D")

    # Convert lat/lon to km
    df[f'{PREFIX}lat_km'] = df['latitude'] * 111
    df[f'{PREFIX}lon_km'] = df['longitude'] * 111

    # Scale time
    df[f'{PREFIX}time_km'] = df[f'{PREFIX}sale_days'] * time_scale

    # Combine features
    X = df[[f'{PREFIX}lat_km', f'{PREFIX}lon_km', f'{PREFIX}time_km']].copy()

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Estimate number of clusters based on target ~n points per cluster
    num_clusters = max(1, len(df) // n)

    # Run KMeans
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    df[f'{PREFIX}cluster'] = kmeans.fit_predict(X_scaled)

    # Compute cluster centers (in degrees)
    centers = df.groupby(f'{PREFIX}cluster')[['latitude', 'longitude']].mean().rename(
        columns={'latitude': f'{PREFIX}center_lat', 'longitude': f'{PREFIX}center_lon'}
    )

    # Merge to compute distances
    df = df.merge(centers, on=f'{PREFIX}cluster')
    df[f'{PREFIX}center_lat_km'] = df[f'{PREFIX}center_lat'] * 111
    df[f'{PREFIX}center_lon_km'] = df[f'{PREFIX}center_lon'] * 111

    # Distance from center in km
    df[f'{PREFIX}distance_km'] = np.sqrt(
        (df[f'{PREFIX}lat_km'] - df[f'{PREFIX}center_lat_km'])**2 +
        (df[f'{PREFIX}lon_km'] - df[f'{PREFIX}center_lon_km'])**2
    )

    # Max radius per cluster
    radii = df.groupby(f'{PREFIX}cluster')[f'{PREFIX}distance_km'].max().rename(f'{PREFIX}radius_km')

    # Average price per cluster
    prices = df.groupby(f'{PREFIX}cluster')['sale_price'].mean().rename(f'{PREFIX}avg_sale_price')

    # Final cluster summary
    clusters = pd.concat([centers, radii, prices], axis=1).reset_index()

    # Merge cluster average price back into df
    df = df.merge(clusters[[f'{PREFIX}cluster', f'{PREFIX}avg_sale_price']], on=f'{PREFIX}cluster', how='left')

    return df, clusters, kmeans, scaler, time_scale


def assign_clusters(df, kmeans, scaler, clusters, time_scale):
    """Assign clusters and average prices to test dataset using trained KMeans and scaler."""
    df = df.copy()

    # Preprocess test data
    df[f'{PREFIX}sale_date'] = pd.to_datetime(df['sale_date'])
    df[f'{PREFIX}sale_days'] = (df[f'{PREFIX}sale_date'] - pd.Timestamp("1970-01-01")) // pd.Timedelta("1D")
    df[f'{PREFIX}lat_km'] = df['latitude'] * 111
    df[f'{PREFIX}lon_km'] = df['longitude'] * 111
    df[f'{PREFIX}time_km'] = df[f'{PREFIX}sale_days'] * time_scale

    # Feature matrix and scale it
    X_test = df[[f'{PREFIX}lat_km', f'{PREFIX}lon_km', f'{PREFIX}time_km']]
    X_test_scaled = scaler.transform(X_test)

    # Predict cluster labels
    df[f'{PREFIX}cluster'] = kmeans.predict(X_test_scaled)

    # Merge with cluster summary to get avg sale price
    df = df.merge(
        clusters[[f'{PREFIX}cluster', f'{PREFIX}avg_sale_price', f'{PREFIX}center_lat', f'{PREFIX}center_lon', f'{PREFIX}radius_km']],
        on=f'{PREFIX}cluster', how='left'
    )

    df.drop(columns=[f'{PREFIX}sale_date'], inplace=True)
    return df


def plot_clusters(clusters, figsize=(6, 3)):
    """Plot semi-transparent cluster circles based on cluster_summary only."""
    _, ax = plt.subplots(figsize=figsize)

    for _, row in clusters.iterrows():
        center_lat = row[f'{PREFIX}center_lat']
        center_lon = row[f'{PREFIX}center_lon']
        radius_km = row[f'{PREFIX}radius_km']

        # Convert radius to degrees (approx)
        radius_deg = radius_km / 111

        # Draw semi-transparent circle with border
        circle = Circle(
            (center_lon, center_lat), radius_deg,
            facecolor='blue', edgecolor='black', linewidth=1.0, alpha=0.3
        )
        ax.add_patch(circle)

    # Set axis limits based on bounds
    buffer = 0.1  # degrees
    ax.set_xlim(clusters[f'{PREFIX}center_lon'].min() - buffer,
                clusters[f'{PREFIX}center_lon'].max() + buffer)
    ax.set_ylim(clusters[f'{PREFIX}center_lat'].min() - buffer,
                clusters[f'{PREFIX}center_lat'].max() + buffer)

    ax.set_xlabel("Longitude", fontsize=10)
    ax.set_ylabel("Latitude", fontsize=10)
    ax.set_title("Cluster Areas (Radius in Degrees)", fontsize=12)
    ax.tick_params(axis='both', labelsize=8)
    ax.grid(True)
    plt.show()


def feature_engineering(df):
    """Perform feature engineering on the dataset."""
    df['sale_date'] = pd.to_datetime(df['sale_date'])
    df['sale_year'] = df['sale_date'].dt.year
    df['sale_month'] = df['sale_date'].dt.month
    df['sale_dayofweek'] = df['sale_date'].dt.dayofweek
    df = df.drop(columns=['sale_date'])

    obj_cols = df.select_dtypes(include='object').columns
    df[obj_cols] = df[obj_cols].fillna('missing')
    df[obj_cols] = OrdinalEncoder().fit_transform(df[obj_cols])

    return df
