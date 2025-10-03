# -----------------------
# If installed the dependencies on a virtual env, run: .\.venv\Scripts\activate to activate it and swithch to the venv interpreter
# To run: cd "Visualizing Path" --> streamlit run updated.py
# -----------------------
import streamlit as st
import pandas as pd
import numpy as np
import time
import math
import heapq
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import folium
from streamlit_folium import st_folium

# Set page config
st.set_page_config(
    page_title="Sri Lanka Path Finder",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded"
)
# -----------------------
# Load Data
# -----------------------
@st.cache_data(show_spinner=False)
def load_data():
    url = 'https://raw.githubusercontent.com/SSenitha/CCS3052_Advance_DSA/refs/heads/main/Cities_of_SriLanka.csv'
    df = pd.read_csv(url)
    df = df.dropna(subset=['latitude', 'longitude'])
    df = df.drop_duplicates(subset=['latitude', 'longitude'], keep='first').reset_index(drop=True)
    df['city_id'] = range(len(df))
    return df

@st.cache_data(show_spinner=False)
def build_sparse_matrix(df, k=6):
    X = df[['latitude', 'longitude']]
    nn = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(X)
    distances, indices = nn.kneighbors(X)

    rows, cols, data = [], [], []
    for i in range(len(df)):
        for j in range(1, k):
            neighbor_index = indices[i, j]
            neighbor_distance = haversine_km(
                df.loc[i, 'latitude'], df.loc[i, 'longitude'],
                df.loc[neighbor_index, 'latitude'], df.loc[neighbor_index, 'longitude']
            )
            rows.append(i)
            cols.append(neighbor_index)
            data.append(neighbor_distance)

    # Create city_to_idx mapping
    city_to_idx = {}
    for idx, row in df.iterrows():
        city_name = row['name_en'].lower().strip()
        city_to_idx[city_name] = idx

    return csr_matrix((data, (rows, cols)), shape=(len(df), len(df))), city_to_idx

# -----------------------
# Utilities
# -----------------------
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def compute_path_km(path_indices, df):
    if not path_indices or len(path_indices) < 2:
        return 0.0
    total_km = 0.0
    for i in range(len(path_indices)-1):
        a, b = path_indices[i], path_indices[i+1]
        total_km += haversine_km(df.loc[a,'latitude'], df.loc[a,'longitude'],
                                 df.loc[b,'latitude'], df.loc[b,'longitude'])
    return total_km

# -----------------------
# Algorithms
# -----------------------
def dijkstra(graph, start, end):
    start_time = time.time()
    n = graph.shape[0]
    distances = [float('inf')] * n
    previous_nodes = [None] * n
    distances[start] = 0
    pq = [(0, start)]

    while pq:
        current_distance, current_node = heapq.heappop(pq)
        if current_node == end:
            break
        if current_distance > distances[current_node]:
            continue
        row = graph.getrow(current_node)
        for neighbor, weight in zip(row.indices, row.data):
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous_nodes[neighbor] = current_node
                heapq.heappush(pq, (distance, neighbor))

    path = reconstruct_path(previous_nodes, start, end)
    exec_time = time.time() - start_time
    return path, distances[end] if path else float('inf'), exec_time

def a_star(sparse_matrix, start, goal, df):
    def heuristic(n1, n2):
        return haversine_km(df.loc[n1,'latitude'], df.loc[n1,'longitude'],
                            df.loc[n2,'latitude'], df.loc[n2,'longitude'])

    start_time = time.time()
    open_set = [(heuristic(start, goal), 0, start, [start])]
    g_costs = {start: 0}

    while open_set:
        f_cost, current_dist, current_node, path = heapq.heappop(open_set)
        if current_node == goal:
            exec_time = time.time() - start_time
            return path, current_dist, exec_time
        row = sparse_matrix.getrow(current_node)
        for neighbor, weight in zip(row.indices, row.data):
            tentative_g = current_dist + weight
            if neighbor not in g_costs or tentative_g < g_costs[neighbor]:
                g_costs[neighbor] = tentative_g
                new_f = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (new_f, tentative_g, neighbor, path+[neighbor]))
    
    exec_time = time.time() - start_time
    return None, float('inf'), exec_time

def bellman_ford(graph, start_node, end_node, num_nodes):
    start_time = time.time()
    distances = [float('inf')] * num_nodes
    previous_nodes = [None] * num_nodes
    distances[start_node] = 0.0
    indices, data, indptr = graph.indices, graph.data, graph.indptr

    for _ in range(num_nodes-1):
        updated = False
        for u in range(num_nodes):
            if distances[u] == float('inf'):
                continue
            for idx in range(indptr[u], indptr[u+1]):
                v, w = indices[idx], data[idx]
                if distances[u] + w < distances[v]:
                    distances[v] = distances[u] + w
                    previous_nodes[v] = u
                    updated = True
        if not updated:
            break
    
    path = reconstruct_path(previous_nodes, start_node, end_node)
    exec_time = time.time() - start_time
    return path, distances[end_node] if path else float('inf'), exec_time

def reconstruct_path(prev, start, end):
    path = []
    cur = end
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    if path and path[0] == start:
        return path
    return None


# -----------------------
# Visualization Functions
# -----------------------
def create_path_map(df, paths_data, start_idx, end_idx, start_city, end_city):
    """Create folium map with paths"""
    try:
        # Calculate map center
        if len(df) > 0:
            center_lat = (df.iloc[start_idx]['latitude'] + df.iloc[end_idx]['latitude']) / 2
            center_lon = (df.iloc[start_idx]['longitude'] + df.iloc[end_idx]['longitude']) / 2
        else:
            center_lat, center_lon = 7.8731, 80.7718  # Sri Lanka center
        
        # Create map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=8,
            tiles='OpenStreetMap'
        )
        
        # Add start marker
        folium.Marker(
            location=[df.iloc[start_idx]['latitude'], df.iloc[start_idx]['longitude']],
            popup=f"Start: {start_city}",
            icon=folium.Icon(color='green', icon='play'),
            tooltip="Start Point"
        ).add_to(m)
        
        # Add end marker
        folium.Marker(
            location=[df.iloc[end_idx]['latitude'], df.iloc[end_idx]['longitude']],
            popup=f"End: {end_city}",
            icon=folium.Icon(color='red', icon='stop'),
            tooltip="End Point"
        ).add_to(m)
        
        # Colors for different algorithms
        colors = {'Dijkstra': '#1f77b4', 'A*': '#ff7f0e', 'Bellman-Ford': '#2ca02c'}
        
        # Add paths
        for algo_name, path, distance, exec_time in paths_data:
            if path and len(path) > 1:
                # Create coordinates list
                path_coords = []
                for node_idx in path:
                    lat = df.iloc[node_idx]['latitude']
                    lon = df.iloc[node_idx]['longitude']
                    path_coords.append([lat, lon])
                
                # Add path line
                folium.PolyLine(
                    locations=path_coords,
                    weight=4,
                    color=colors.get(algo_name, '#000000'),
                    opacity=0.8,
                    #tooltip=f"{algo_name}: {distance:.2f} km"
                ).add_to(m)
                
                # Add intermediate points
                for i, node_idx in enumerate(path[1:-1], 1):
                    folium.CircleMarker(
                    [df.iloc[node_idx]['latitude'],df.iloc[node_idx]['longitude']],
                    radius=4,
                    color=colors.get(algo_name,'#000'),
                    fill=True,
                    fillOpacity=0.7,
                    tooltip=df.iloc[node_idx]['name_en'] 
                    ).add_to(m)
        
        return m
        
    except Exception as e:
        st.error(f"Error creating map: {str(e)}")
        return None

def run_pathfinding(start_city, end_city, algorithms, df, graph, city_to_idx):
    """Run pathfinding algorithms and return results"""
    # Get city indices
    start_idx = city_to_idx.get(start_city.lower().strip())
    end_idx = city_to_idx.get(end_city.lower().strip())
    
    if start_idx is None or end_idx is None:
        return None, "Could not find selected cities in database."
    
    # Check if any algorithm is selected
    selected_algos = [name for name, selected in algorithms.items() if selected]
    if not selected_algos:
        return None, "Please select at least one algorithm."
    
    # Run algorithms
    results = []
    
    if algorithms.get("Dijkstra"):
        path, dist, time_taken = dijkstra(graph, start_idx, end_idx)
        if path is not None:
            results.append(("Dijkstra", path, dist, time_taken))
    
    if algorithms.get("A*"):
        path, dist, time_taken = a_star(graph, start_idx, end_idx, df)
        if path is not None:
            results.append(("A*", path, dist, time_taken))
    
    if algorithms.get("Bellman-Ford"):
        path, dist, time_taken = bellman_ford(graph, start_idx, end_idx, len(df))
        if path is not None:
            results.append(("Bellman-Ford", path, dist, time_taken))
    
    if not results:
        return None, "No path found between selected cities."
    
    return results, None

# -----------------------
# Main Application
# -----------------------
def main():
    # Initialize session state variables
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'graph' not in st.session_state:
        st.session_state.graph = None
    if 'city_to_idx' not in st.session_state:
        st.session_state.city_to_idx = None
    if 'results' not in st.session_state:
        st.session_state.results = []
    if 'last_search' not in st.session_state:
        st.session_state.last_search = None

    st.title("🛣️ Sri Lanka City Shortest Path Finder")
    st.markdown("Find the shortest path between cities in Sri Lanka using different algorithms.")
    
    # Load data if not already loaded
    if st.session_state.df is None:
        with st.spinner("Loading city data..."):
            st.session_state.df = load_data()
        
        if st.session_state.df is None:
            st.stop()
    
    # Build graph if not already built
    if st.session_state.graph is None or st.session_state.city_to_idx is None:
        with st.spinner("Building city network..."):
            st.session_state.graph, st.session_state.city_to_idx = build_sparse_matrix(st.session_state.df)
        
        if st.session_state.graph is None or st.session_state.city_to_idx is None:
            st.error("Failed to build city network.")
            st.stop()
    
    #st.success(f"✅ Loaded {len(st.session_state.df)} cities successfully!")
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("Path Configuration")
        
        # City selection
        city_names = sorted(st.session_state.df['name_en'].unique())
        
        start_city = st.selectbox(
            "🏁 Start City:",
            city_names,
            index=0,
            key="start_city_select"
        )
        
        end_city = st.selectbox(
            "🎯 Destination City:",
            city_names,
            index=min(1, len(city_names)-1) if len(city_names) > 1 else 0,
            key="end_city_select"
        )
        
        st.header("Algorithm Selection")
        algorithms = {
            "Dijkstra": st.checkbox("Dijkstra's Algorithm", value=True, key="dijkstra_check"),
            "A*": st.checkbox("A* Algorithm", value=True, key="astar_check"),
            "Bellman-Ford": st.checkbox("Bellman-Ford Algorithm", value=False, key="bellman_check")
        }
        
        find_path = st.button("🔍 Find Shortest Path", type="primary", key="find_path_btn")
        
        # Clear results button
        if st.button("🗑️ Clear Results", key="clear_results_btn"):
            st.session_state.results = []
            st.session_state.last_search = None
            st.rerun()
    
    # Process search when button is clicked
    if find_path:
        if start_city == end_city:
            st.warning("⚠️ Please select different start and destination cities.")
        else:
            with st.spinner("Computing shortest paths..."):
                results, error = run_pathfinding(
                    start_city, end_city, algorithms, 
                    st.session_state.df, st.session_state.graph, st.session_state.city_to_idx
                )
                
                if error:
                    st.error(f"❌ {error}")
                else:
                    st.success("✅ Pathfinding completed!")
                    st.session_state.results = results
                    st.session_state.last_search = {
                        'start_city': start_city,
                        'end_city': end_city,
                        'start_idx': st.session_state.city_to_idx.get(start_city.lower().strip()),
                        'end_idx': st.session_state.city_to_idx.get(end_city.lower().strip())
                    }
                    
    
    # Display results if available
    if st.session_state.results and st.session_state.last_search:
        #st.header("📊 Results")
        
        search_info = st.session_state.last_search
        st.info(f"🛣️ Path from **{search_info['start_city']}** to **{search_info['end_city']}**")
        
        # Create columns for results
        results = st.session_state.results
        cols = st.columns(len(results))
        
        for i, (algo_name, path, distance, exec_time) in enumerate(results):
            with cols[i]:
                st.subheader(f"{algo_name}")
                st.metric("Distance", f"{distance:.2f} km")
                st.metric("Execution Time", f"{exec_time:.6f} s")
                st.metric("Cities Visited", len(path))
                
                # Show path
                with st.expander("View Path"):
                    path_names = [st.session_state.df.iloc[idx]['name_en'] for idx in path]
                    for j, city in enumerate(path_names):
                        if j == 0:
                            st.write(f"🏁 {city}")
                        elif j == len(path_names) - 1:
                            st.write(f"🎯 {city}")
                        else:
                            st.write(f"📍 {city}")
        
        # Comparison table
        if len(results) > 1:
            st.subheader("📈 Algorithm Comparison")
            comparison_data = []
            for algo_name, path, distance, exec_time in results:
                comparison_data.append({
                    "Algorithm": algo_name,
                    "Distance (km)": f"{distance:.2f}",
                    "Time (seconds)": f"{exec_time:.6f}",
                    "Cities": len(path)
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
        
        # Map visualization
        st.subheader("🗺️ Path Visualization")
        
        try:
            path_map = create_path_map(
                st.session_state.df, 
                results, 
                search_info['start_idx'], 
                search_info['end_idx'],
                search_info['start_city'],
                search_info['end_city']
            )
            if path_map is not None:
                map_data = st_folium(path_map, width=700, height=500, key="path_map")
            else:
                st.error("Could not create map visualization.")
        except Exception as e:
            st.error(f"Map visualization error: {str(e)}")
    
    elif not st.session_state.results:
        # Show initial instructions
        st.info("👈 Select start and destination cities from the sidebar, choose algorithms, and click 'Find Shortest Path' to begin.")
    
    # Show dataset info
    # with st.expander("📋 Dataset Information"):
    #     st.write(f"**Total Cities:** {len(st.session_state.df)}")
    #     st.write("**Sample Data:**")
    #     st.dataframe(st.session_state.df.head(), use_container_width=True)

if __name__ == "__main__":
    main()