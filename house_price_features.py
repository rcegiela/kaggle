"""House Price Features"""
import re
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import OrdinalEncoder
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# pylint: disable=line-too-long, invalid-name

PREFIX = 'clw_'

class ClusterLocations:
    """Class to handle clustering of locations based on latitude, longitude, and sale date."""

    def __init__(self, n=10, time_scale=1/36.5):
        self.n = n
        self.time_scale = time_scale
        self.df, self.clusters, self.kmeans = None, None, None

    def fit(self, df):
        """Fit the clustering model to the data."""
        self.df, self.clusters, self.kmeans = ClusterLocations.cluster_locations(df, n=self.n, time_scale=self.time_scale)
        return self

    def transform(self, df):
        """Transform the data by assigning clusters."""
        return ClusterLocations.assign_clusters(df, self.kmeans, self.clusters, self.time_scale)

    @staticmethod
    def cluster_locations(df, n=10, time_scale=1/36.5):
        """
        Cluster locations using KMeans based on latitude, longitude, and sale date.
        Creates approximately n-sized circular clusters.
        """
        df = df.copy()

        # Convert to ordinal (days since a reference point)
        df[f'{PREFIX}sale_days'] = (df['sale_date'] - pd.Timestamp("1970-01-01")) // pd.Timedelta("1D")

        # Convert lat/lon to km
        df[f'{PREFIX}lat_km'] = df['latitude'] * 111
        df[f'{PREFIX}lon_km'] = df['longitude'] * 111

        # Scale time
        df[f'{PREFIX}time_km'] = df[f'{PREFIX}sale_days'] * time_scale

        # Combine features
        X = df[[f'{PREFIX}lat_km', f'{PREFIX}lon_km', f'{PREFIX}time_km']].copy()

        # Removed StandardScaler
        X_scaled = X

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
        sale_price_columns =  ['sale_price', 'sale_price_per_sqft', 'sale_price_net', 'sale_price_net_per_sqft']
        prices = df.groupby(f'{PREFIX}cluster')[sale_price_columns].mean().rename(columns=lambda col: f'{PREFIX}avg_{col}')

        # Final cluster summary
        clusters = pd.concat([centers, radii, prices], axis=1).reset_index()

        # Merge cluster average price back into df
        df = df.merge(clusters[[f'{PREFIX}cluster'] + list(prices.columns)], on=f'{PREFIX}cluster', how='left')

        # Return without the scaler since we're not using it
        return df, clusters, kmeans

    @staticmethod
    def assign_clusters(df, kmeans, clusters, time_scale):
        """Assign clusters and average prices to test dataset using trained KMeans."""
        df = df.copy()

        # Preprocess test data
        df[f'{PREFIX}sale_date'] = pd.to_datetime(df['sale_date'])
        df[f'{PREFIX}sale_days'] = (df[f'{PREFIX}sale_date'] - pd.Timestamp("1970-01-01")) // pd.Timedelta("1D")
        df[f'{PREFIX}lat_km'] = df['latitude'] * 111
        df[f'{PREFIX}lon_km'] = df['longitude'] * 111
        df[f'{PREFIX}time_km'] = df[f'{PREFIX}sale_days'] * time_scale

        # Feature matrix without scaling
        X_test = df[[f'{PREFIX}lat_km', f'{PREFIX}lon_km', f'{PREFIX}time_km']]

        # Predict cluster labels (no scaling)
        df[f'{PREFIX}cluster'] = kmeans.predict(X_test)

        # Merge with cluster summary to get avg sale price
        df = df.merge(
            clusters,
            on=f'{PREFIX}cluster',
            how='left',
            suffixes=('', '_cluster')
        )

        df.drop(columns=[f'{PREFIX}sale_date'], inplace=True)
        return df


    def plot(self, figsize=(6, 3)):
        """Plot semi-transparent cluster circles based on cluster_summary only."""
        _, ax = plt.subplots(figsize=figsize)

        for _, row in self.clusters.iterrows():
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
        ax.set_xlim(self.clusters[f'{PREFIX}center_lon'].min() - buffer,
                    self.clusters[f'{PREFIX}center_lon'].max() + buffer)
        ax.set_ylim(self.clusters[f'{PREFIX}center_lat'].min() - buffer,
                    self.clusters[f'{PREFIX}center_lat'].max() + buffer)

        ax.set_xlabel("Longitude", fontsize=10)
        ax.set_ylabel("Latitude", fontsize=10)
        ax.set_title("Cluster Areas (Radius in Degrees)", fontsize=12)
        ax.tick_params(axis='both', labelsize=8)
        ax.grid(True)
        plt.show()




def feature_engineering(df):
    """Perform feature engineering on the dataset."""
    df = df.copy()

    df['sale_date'] = pd.to_datetime(df['sale_date'])
    df['sale_year'] = df['sale_date'].dt.year
    df['sale_month'] = df['sale_date'].dt.month
    df['sale_dayofweek'] = df['sale_date'].dt.dayofweek

    def split_letters_numbers(text):
        letters = ''.join(re.findall(r'[^\d.]', str(text)))
        numbers = ''.join(re.findall(r'[\d.]+', str(text)))
        return letters, numbers

    # Apply the function to create new columns
    df['sale_nbr'] = df['sale_nbr'].fillna(-1).astype('int')

    df['stories'] = (df['stories']*10).astype(int)
    df['stories_half'] = df['stories'] % 10 != 0

    df['sqft_living']=df['sqft']+df['sqft_1']+df['sqft_fbsmt']
    df['sqft_total']=df['sqft_living']+df['garb_sqft']+df['gara_sqft']

    df['land_imp_val'] = df['land_val'] + df['imp_val']
    df['land_val_per_sqft'] = df['land_val'] / df['sqft_lot']
    df['imp_val_per_sqft'] = df['imp_val'] / df['sqft_lot']

    if 'sale_price' in df.columns:
        df['sale_price_per_sqft'] = df['sale_price'] / df['sqft_living']
        df['sale_price_net'] = df['sale_price'] - df['imp_val'] - df['land_val']
        df['sale_price_net_per_sqft'] = df['sale_price_net'] / df['sqft_living']

    df[['zoning_code', 'zoning_number']] = df['zoning'].apply(lambda x: pd.Series(split_letters_numbers(x)))

    df['subdivision_ADD'] = df['subdivision'].str.contains(r'\bADD\b', case=True, na=False).astype(bool)
    df['subdivision_DIV'] = df['subdivision'].str.contains(r'\bDIV\b', case=True, na=False).astype(bool)

    obj_cols = df.select_dtypes(include='object').columns
    df[obj_cols] = df[obj_cols].fillna('missing')
    df[obj_cols] = OrdinalEncoder().fit_transform(df[obj_cols])

    return df

def feature_clean_up(df):
    """Clean up the features before training."""
    df = df.copy()
    drop_cols = ['sale_date', 'id']+[col for col in df.columns if col.startswith('sale_price_')]
    df.drop(columns=drop_cols, inplace=True)

    return df
