import streamlit as st
import pandas as pd
import numpy as np
import math
import time
import heapq
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import folium
from streamlit_folium import folium_static
import os

# Set page configuration
st.set_page_config(
    page_title="Sri Lanka Shortest Path Finder",
    page_icon="🗺️",
    layout="wide"
)

# Title and description
st.title("🗺️ Sri Lanka Shortest Path Finder")
st.markdown("""
Find the shortest distance and path between two cities in Sri Lanka using different algorithms.
The app uses geographical coordinates to calculate real road distances and visualizes the path on an interactive map.
""")

# Haversine distance function
def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the great-circle distance between two points on Earth in kilometers"""
    R = 6371.0088  # Earth's radius in km
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    
    a = math.sin(dphi/2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2.0)**2
    a = min(1.0, max(0.0, a))  # Clamp to avoid numerical issues
    return 2 * R * math.asin(math.sqrt(a))

# Cache the data loading and graph building
@st.cache_data
def load_data():
    """Load and preprocess the cities data"""
    try:
        # Try to load from local file first
        df = pd.read_csv('Cities_of_SriLanka.csv')
    except FileNotFoundError:
        st.error("Cities_of_SriLanka.csv not found. Please make sure the file is in the same directory as this script.")
        st.stop()
    
    df.rename(columns={"city id": "city_id"}, inplace=True)
    df = df.drop_duplicates(subset=['latitude', 'longitude'], keep='first').reset_index(drop=True)
    df = df.drop(columns=["district_id","name_si","name_ta","sub_name_en","sub_name_si","sub_name_ta","postcode"], errors='ignore')
    df.drop(columns=["city_id"], inplace=True, errors='ignore')
    df.insert(0, "city_id", df.index)
    return df

@st.cache_data
def build_sparse_matrix(_df, k=6):
    """Build the sparse adjacency matrix using k-NN with Haversine distance"""
    num_cities = len(_df)
    X = _df[['latitude', 'longitude']].values
    
    # Use ball tree with Haversine metric
    nn = NearestNeighbors(n_neighbors=k, algorithm='ball_tree', metric='haversine')
    nn.fit(np.radians(X))
    distances_rad, indices = nn.kneighbors(np.radians(X))

    rows, cols, data = [], [], []
    seen_edges = set()

    for i in range(num_cities):
        for j in range(1, k):  # Skip self (j=0)
            n_index = indices[i, j]
            edge = frozenset((i, n_index))
            if edge in seen_edges:
                continue
            seen_edges.add(edge)
            
            # Convert radian distance to kilometers
            n_distance = distances_rad[i, j] * 6371.0088
            
            # Add both directions for undirected graph
            rows.append(i)
            cols.append(n_index)
            data.append(n_distance)
            rows.append(n_index)
            cols.append(i)
            data.append(n_distance)

    return csr_matrix((data, (rows, cols)), shape=(num_cities, num_cities))

def create_folium_map(df, path_indices, start_city, end_city, algorithm_name):
    """Create an interactive Folium map showing the path"""
    if not path_indices:
        return None
    
    # Get coordinates for the path
    path_coords = []
    for idx in path_indices:
        lat = df.iloc[idx]['latitude']
        lon = df.iloc[idx]['longitude']
        path_coords.append([lat, lon])
    
    # Calculate map center
    center_lat = sum(coord[0] for coord in path_coords) / len(path_coords)
    center_lon = sum(coord[1] for coord in path_coords) / len(path_coords)
    
    # Create map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=8,
        tiles='OpenStreetMap'
    )
    
    # Add path line
    folium.PolyLine(
        path_coords,
        color='blue',
        weight=4,
        opacity=0.8,
        popup=f"{algorithm_name} Path",
        tooltip=f"Path found by {algorithm_name}"
    ).add_to(m)
    
    # Add markers for start and end
    start_lat, start_lon = path_coords[0]
    end_lat, end_lon = path_coords[-1]
    
    folium.Marker(
        [start_lat, start_lon],
        popup=f"Start: {start_city}",
        tooltip="Start City",
        icon=folium.Icon(color='green', icon='play', prefix='fa')
    ).add_to(m)
    
    folium.Marker(
        [end_lat, end_lon],
        popup=f"End: {end_city}",
        tooltip="End City",
        icon=folium.Icon(color='red', icon='stop', prefix='fa')
    ).add_to(m)
    
    # Add intermediate city markers
    for i, idx in enumerate(path_indices[1:-1], 1):
        lat, lon = path_coords[i]
        city_name = df.iloc[idx]['name_en']
        folium.Marker(
            [lat, lon],
            popup=f"Via: {city_name} (Step {i})",
            tooltip=f"Via: {city_name}",
            icon=folium.Icon(color='blue', icon='arrow-right', prefix='fa')
        ).add_to(m)
    
    # Add distance information
    total_distance = calculate_path_distance(df, path_indices)
    
    # Add control to show/hide markers
    folium.LayerControl().add_to(m)
    
    return m

def calculate_path_distance(df, path_indices):
    """Calculate total distance of the path"""
    total_distance = 0
    for i in range(len(path_indices) - 1):
        idx1, idx2 = path_indices[i], path_indices[i + 1]
        lat1, lon1 = df.iloc[idx1]['latitude'], df.iloc[idx1]['longitude']
        lat2, lon2 = df.iloc[idx2]['latitude'], df.iloc[idx2]['longitude']
        total_distance += haversine_distance(lat1, lon1, lat2, lon2)
    return total_distance

# Algorithm implementations (same as before)
def dijkstra_algorithm(sparse_matrix, start_idx, end_idx):
    """Dijkstra's algorithm implementation"""
    start_time = time.time()
    
    num_nodes = sparse_matrix.shape[0]
    distances = [float('inf')] * num_nodes
    distances[start_idx] = 0
    priority_queue = [(0, start_idx)]
    previous_nodes = [None] * num_nodes
    nodes_visited = 0

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)

        if current_distance > distances[current_node]:
            continue

        nodes_visited += 1

        if current_node == end_idx:
            break

        row_start = sparse_matrix.indptr[current_node]
        row_end = sparse_matrix.indptr[current_node + 1]
        
        for i in range(row_start, row_end):
            neighbor = sparse_matrix.indices[i]
            weight = sparse_matrix.data[i]
            distance = current_distance + weight

            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous_nodes[neighbor] = current_node
                heapq.heappush(priority_queue, (distance, neighbor))

    path = []
    current_node = end_idx
    while current_node is not None:
        path.append(current_node)
        current_node = previous_nodes[current_node]
    path.reverse()
    
    execution_time = time.time() - start_time
    
    if path and path[0] == start_idx:
        return path, distances[end_idx], execution_time, nodes_visited
    else:
        return None, float('inf'), execution_time, nodes_visited

def bellman_ford_algorithm(sparse_matrix, start_idx, end_idx):
    """Bellman-Ford algorithm implementation"""
    start_time = time.time()
    
    num_nodes = sparse_matrix.shape[0]
    dist = [float('inf')] * num_nodes
    pred = [None] * num_nodes
    dist[start_idx] = 0.0
    relaxations = 0

    edges = []
    coo = sparse_matrix.tocoo()
    for u, v, w in zip(coo.row, coo.col, coo.data):
        edges.append((int(u), int(v), float(w)))

    for i in range(num_nodes - 1):
        updated = False
        for u, v, w in edges:
            if dist[u] != float('inf') and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                pred[v] = u
                updated = True
                relaxations += 1
        if not updated:
            break

    negative_cycle = False
    for u, v, w in edges:
        if dist[u] != float('inf') and dist[u] + w < dist[v]:
            negative_cycle = True
            break

    path = []
    current = end_idx
    while current is not None:
        path.append(current)
        current = pred[current]
    path.reverse()
    
    execution_time = time.time() - start_time
    
    if path and path[0] == start_idx and not negative_cycle:
        return path, dist[end_idx], execution_time, relaxations
    else:
        return None, float('inf'), execution_time, relaxations

def astar_algorithm(sparse_matrix, df, start_idx, end_idx):
    """A* algorithm implementation"""
    start_time = time.time()
    
    def get_neighbors(city_index):
        row_start = sparse_matrix.indptr[city_index]
        row_end = sparse_matrix.indptr[city_index + 1]
        neighbors = []
        for i in range(row_start, row_end):
            neighbors.append(sparse_matrix.indices[i])
        return neighbors
    
    def get_edge_weight(u, v):
        row_start = sparse_matrix.indptr[u]
        row_end = sparse_matrix.indptr[u + 1]
        for i in range(row_start, row_end):
            if sparse_matrix.indices[i] == v:
                return sparse_matrix.data[i]
        return float('inf')

    open_set = []
    heapq.heappush(open_set, (0, 0, start_idx, [start_idx]))
    g_costs = {start_idx: 0}
    nodes_expanded = 0

    while open_set:
        f_cost, current_distance, current_index, path = heapq.heappop(open_set)
        nodes_expanded += 1

        if current_index == end_idx:
            execution_time = time.time() - start_time
            return path, current_distance, execution_time, nodes_expanded

        neighbors = get_neighbors(current_index)

        for neighbor_index in neighbors:
            edge_weight = get_edge_weight(current_index, neighbor_index)
            tentative_g_cost = current_distance + edge_weight

            if neighbor_index not in g_costs or tentative_g_cost < g_costs[neighbor_index]:
                g_costs[neighbor_index] = tentative_g_cost
                
                lat1, lon1 = df.iloc[neighbor_index]['latitude'], df.iloc[neighbor_index]['longitude']
                lat2, lon2 = df.iloc[end_idx]['latitude'], df.iloc[end_idx]['longitude']
                heuristic = haversine_distance(lat1, lon1, lat2, lon2)
                
                f_cost = tentative_g_cost + heuristic
                heapq.heappush(open_set, (f_cost, tentative_g_cost, neighbor_index, path + [neighbor_index]))

    execution_time = time.time() - start_time
    return None, float('inf'), execution_time, nodes_expanded

# Main application
def main():
    # Load data
    with st.spinner("Loading city data..."):
        df = load_data()
        sparse_matrix = build_sparse_matrix(df)
    
    st.success(f"✅ Loaded {len(df)} cities and built road network")
    
    # Sidebar for controls
    st.sidebar.header("Configuration")
    
    # City selection
    city_names = df['name_en'].tolist()
    start_city = st.sidebar.selectbox("Select Start City", city_names, index=0)
    end_city = st.sidebar.selectbox("Select End City", city_names, index=min(1, len(city_names)-1))
    
    # Algorithm selection
    algorithms = st.sidebar.multiselect(
        "Select Algorithms to Compare",
        ["Dijkstra", "Bellman-Ford", "A*"],
        default=["Dijkstra", "A*"]
    )
    
    # Map visualization options
    show_map = st.sidebar.checkbox("Show Interactive Map", value=True)
    
    # Get city indices
    start_idx = df[df['name_en'] == start_city].index[0]
    end_idx = df[df['name_en'] == end_city].index[0]
    
    # Display city information
    col1, col2 = st.columns(2)
    with col1:
        start_data = df.iloc[start_idx]
        st.subheader(f"📍 Start: {start_city}")
        st.write(f"**Coordinates:** {start_data['latitude']:.4f}, {start_data['longitude']:.4f}")
        st.write(f"**City ID:** {start_idx}")
    
    with col2:
        end_data = df.iloc[end_idx]
        st.subheader(f"🎯 Destination: {end_city}")
        st.write(f"**Coordinates:** {end_data['latitude']:.4f}, {end_data['longitude']:.4f}")
        st.write(f"**City ID:** {end_idx}")
    
    # Straight-line distance
    straight_distance = haversine_distance(
        start_data['latitude'], start_data['longitude'],
        end_data['latitude'], end_data['longitude']
    )
    st.info(f"📏 Straight-line distance: {straight_distance:.2f} km")
    
    # Calculate paths
    if st.button("🚀 Find Shortest Path", type="primary"):
        if not algorithms:
            st.warning("Please select at least one algorithm.")
            return
            
        results = []
        maps = []
        
        for algorithm in algorithms:
            with st.spinner(f"Running {algorithm} algorithm..."):
                if algorithm == "Dijkstra":
                    path, distance, exec_time, nodes_visited = dijkstra_algorithm(sparse_matrix, start_idx, end_idx)
                elif algorithm == "Bellman-Ford":
                    path, distance, exec_time, nodes_visited = bellman_ford_algorithm(sparse_matrix, start_idx, end_idx)
                elif algorithm == "A*":
                    path, distance, exec_time, nodes_visited = astar_algorithm(sparse_matrix, df, start_idx, end_idx)
                
                if path is not None and distance != float('inf'):
                    path_names = [df.iloc[idx]['name_en'] for idx in path]
                    
                    # Create map if requested
                    folium_map = None
                    if show_map:
                        folium_map = create_folium_map(df, path, start_city, end_city, algorithm)
                    
                    results.append({
                        'algorithm': algorithm,
                        'path': path,
                        'path_names': path_names,
                        'distance': distance,
                        'time': exec_time,
                        'nodes_visited': len(path),
                        'nodes_explored': nodes_visited,
                        'map': folium_map
                    })
                else:
                    results.append({
                        'algorithm': algorithm,
                        'path': None,
                        'path_names': [],
                        'distance': float('inf'),
                        'time': exec_time,
                        'nodes_visited': 0,
                        'nodes_explored': nodes_visited,
                        'map': None
                    })
        
        # Display results
        st.header("📊 Results")
        
        # Show maps in tabs if multiple algorithms
        if show_map and any(r['map'] is not None for r in results):
            st.subheader("🗺️ Interactive Maps")
            
            valid_maps = [(i, r) for i, r in enumerate(results) if r['map'] is not None]
            
            if len(valid_maps) == 1:
                # Single map - show directly
                idx, result = valid_maps[0]
                st.write(f"**{result['algorithm']} Algorithm Path**")
                folium_static(result['map'], width=800, height=500)
            else:
                # Multiple maps - show in tabs
                tabs = st.tabs([f"{result['algorithm']}" for _, result in valid_maps])
                
                for tab, (idx, result) in zip(tabs, valid_maps):
                    with tab:
                        folium_static(result['map'], width=800, height=500)
        
        # Algorithm comparison section
        st.subheader("📈 Algorithm Comparison")
        
        # Create columns for results
        cols = st.columns(len(results))
        
        for i, (result, col) in enumerate(zip(results, cols)):
            with col:
                st.subheader(result['algorithm'])
                
                if result['path'] is not None:
                    st.metric("Distance", f"{result['distance']:.2f} km")
                    st.metric("Execution Time", f"{result['time']:.4f} s")
                    st.metric("Path Length", f"{result['nodes_visited']} cities")
                    st.metric("Nodes Explored", result['nodes_explored'])
                    
                    with st.expander("View Full Path"):
                        for j, city in enumerate(result['path_names']):
                            st.write(f"{j+1}. {city}")
                            
                            # Show coordinates for each city in path
                            city_idx = result['path'][j]
                            city_data = df.iloc[city_idx]
                            st.caption(f"   📍 {city_data['latitude']:.4f}, {city_data['longitude']:.4f}")
                else:
                    st.error("❌ No path found")
                    st.metric("Execution Time", f"{result['time']:.4f} s")
                    st.metric("Nodes Explored", result['nodes_explored'])
        
        # Detailed comparison table
        valid_results = [r for r in results if r['path'] is not None]
        if len(valid_results) > 1:
            st.subheader("📋 Detailed Comparison")
            
            comparison_data = {
                'Algorithm': [r['algorithm'] for r in valid_results],
                'Distance (km)': [r['distance'] for r in valid_results],
                'Time (s)': [r['time'] for r in valid_results],
                'Path Length': [r['nodes_visited'] for r in valid_results],
                'Nodes Explored': [r['nodes_explored'] for r in valid_results]
            }
            
            comp_df = pd.DataFrame(comparison_data)
            st.dataframe(comp_df, use_container_width=True)
            
            # Create visual comparison
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
            ax1.bar(comp_df['Algorithm'], comp_df['Distance (km)'], color=colors[:len(comp_df)])
            ax1.set_title('Distance Comparison')
            ax1.set_ylabel('Distance (km)')
            ax1.tick_params(axis='x', rotation=45)
            
            ax2.bar(comp_df['Algorithm'], comp_df['Time (s)'], color=colors[:len(comp_df)])
            ax2.set_title('Execution Time Comparison')
            ax2.set_ylabel('Time (seconds)')
            ax2.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            st.pyplot(fig)
        
        # Best result
        valid_results = [r for r in results if r['path'] is not None]
        if valid_results:
            best_result = min(valid_results, key=lambda x: x['distance'])
            st.success(f"🎯 **Best path found by {best_result['algorithm']}**: {best_result['distance']:.2f} km in {best_result['time']:.4f} seconds")
    
    # Information section
    with st.expander("ℹ️ About the Algorithms"):
        st.markdown("""
        **Dijkstra's Algorithm**: 
        - Guaranteed to find the shortest path in graphs with non-negative weights
        - Time complexity: O((V + E) log V) with binary heap
        - Explores all possible paths uniformly
        
        **Bellman-Ford Algorithm**:
        - Can handle graphs with negative weight edges
        - Can detect negative cycles
        - Time complexity: O(V × E)
        - Slower but more versatile
        
        **A* Algorithm**:
        - Uses heuristics to guide the search toward the target
        - More efficient than Dijkstra for single-pair shortest path
        - Guaranteed to find optimal path with admissible heuristic
        - Uses straight-line distance as heuristic
        """)
    
    with st.expander("🗺️ About the Map Visualization"):
        st.markdown("""
        The interactive map shows:
        - 🟢 **Green marker**: Starting city
        - 🔴 **Red marker**: Destination city  
        - 🔵 **Blue markers**: Intermediate cities along the path
        - 📍 **Blue line**: The actual path connecting all cities
        
        You can:
        - Zoom in/out using mouse wheel or +/- buttons
        - Click on markers to see city information
        - Drag the map to explore different areas
        - Switch between different map layers (if available)
        """)
    
    with st.expander("📋 About the Data"):
        st.write(f"**Total Cities in Dataset:** {len(df)}")
        st.write("**Data Source:** Cities of Sri Lanka geographical data")
        st.write("**Distance Metric:** Haversine formula for great-circle distance")
        st.write("**Graph Construction:** k-NN graph with k=6 nearest neighbors")
        st.write("**Map Tiles:** OpenStreetMap")

if __name__ == "__main__":
    main()