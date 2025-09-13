import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import heapq

# -----------------------
# Load dataset
# -----------------------
df = pd.read_csv('https://raw.githubusercontent.com/SSenitha/CCS3052_Advance_DSA/refs/heads/main/Cities_of_SriLanka.csv')
df.rename(columns={"city id": "city_id"}, inplace=True)

# Drop duplicates & unnecessary columns
df = df.drop_duplicates(subset=['latitude', 'longitude'], keep='first').reset_index(drop=True)
df = df.drop(columns=["district_id","name_si","name_ta","sub_name_en","sub_name_si","sub_name_ta","postcode"])

# Reset city_id
df.drop(columns=["city_id"], inplace=True)
df.insert(0, "city_id", df.index)

# -----------------------
# Haversine Distance Function
# -----------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

# -----------------------
# Nearest Neighbors Graph with real km distances
# -----------------------
num_cities = len(df)
k = 6  # number of neighbors

coords = df[['latitude', 'longitude']].values
nn = NearestNeighbors(n_neighbors=k, algorithm='ball_tree', metric='haversine').fit(np.radians(coords))
distances_rad, indices = nn.kneighbors(np.radians(coords))

# Convert distances from radians to km
distances_km = distances_rad * 6371

rows, cols, data = [], [], []
for i in range(num_cities):
    for j in range(1, k):
        neighbor_index = indices[i, j]
        neighbor_distance = distances_km[i, j]
        rows.append(i)
        cols.append(neighbor_index)
        data.append(neighbor_distance)

sparse_matrix = csr_matrix((data, (rows, cols)), shape=(num_cities, num_cities))

# -----------------------
# Utility Functions
# -----------------------
city_name_to_index = {name.lower(): idx for idx, name in enumerate(df['name_en'])}

def dijkstra(graph, start):
    n = graph.shape[0]
    distances = [float('inf')] * n
    previous_nodes = [None] * n
    distances[start] = 0
    pq = [(0, start)]  # priority queue

    while pq:
        current_distance, current_node = heapq.heappop(pq)
        if current_distance > distances[current_node]:
            continue

        row = graph.getrow(current_node)
        for neighbor, weight in zip(row.indices, row.data):
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous_nodes[neighbor] = current_node
                heapq.heappush(pq, (distance, neighbor))

    return distances, previous_nodes

def get_shortest_path(previous_nodes, start, end):
    path = []
    current = end
    while current is not None:
        path.insert(0, current)
        current = previous_nodes[current]
    return path if path[0] == start else []

# -----------------------
# Streamlit UI
# -----------------------
st.title("🛣️ Sri Lanka City Shortest Path Finder")

# Sort city names alphabetically
city_list = sorted(df['name_en'].values)

start_city = st.selectbox("Select Start City", city_list)
end_city = st.selectbox("Select Destination City", city_list)

if st.button("Find Shortest Path"):
    start_idx = city_name_to_index.get(start_city.lower())
    end_idx = city_name_to_index.get(end_city.lower())

    if start_idx is None or end_idx is None:
        st.error("Invalid city selection.")
    else:
        start_time = time.time()
        distances, previous_nodes = dijkstra(sparse_matrix, start_idx)
        end_time = time.time()

        shortest_distance = distances[end_idx]
        shortest_path_indices = get_shortest_path(previous_nodes, start_idx, end_idx)

        st.write(f"⏱️ Execution time: {end_time - start_time:.6f} seconds")

        if shortest_distance != float('inf'):
            st.success(f"Shortest distance from **{start_city}** to **{end_city}**: **{shortest_distance:.2f} km**")
            shortest_path_names = [df.loc[i, 'name_en'] for i in shortest_path_indices]
            st.write("📍 Path (city names):", " ➝ ".join(shortest_path_names))
        else:
            st.error(f"No path found from {start_city} to {end_city}.")
