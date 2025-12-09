import numpy as np
import pandas as pd
from geopy.distance import geodesic


def geodesic_distance(df:pd.DataFrame, lat_key:str, lon_key:str, name_key:str) -> pd.DataFrame:
    """
    calculate geodesic distance using latitudes and longitudes
    """

    # Create a list of (latitude, longitude) tuples
    coords = list(zip(df[lat_key], df[lon_key]))

    # Initialize an empty distance matrix
    num_locations = len(coords)
    distance_matrix = pd.DataFrame(index=df[name_key], columns=df[name_key])

    # Calculate all-pairs geodesic distances
    for i in range(num_locations):
        for j in range(num_locations):
            if i == j:
                distance_matrix.iloc[i, j] = 0.0  # Distance to self is 0
            else:
                dist = geodesic(coords[i], coords[j]).km  # Distance in kilometers
                distance_matrix.iloc[i, j] = dist

    return distance_matrix


import numpy as np

# ---------------------------
# 1. Distance helper (km)
# ---------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    """
    Compute great-circle distance in km between two points specified in degrees.
    """
    R = 6371.0  # Earth radius (km)
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def distance_matrix_km(lats, lons):
    """
    Compute NxN geodesic distance matrix for nodes with given lat/lon.
    """
    lats = np.asarray(lats)
    lons = np.asarray(lons)
    N = len(lats)
    D = np.zeros((N, N))
    for i in range(N):
        D[i, :] = haversine_km(lats[i], lons[i], lats, lons)
    return D