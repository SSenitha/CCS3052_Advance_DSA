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

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = []
if 'last_search' not in st.session_state:
    st.session_state.last_search = None
if 'df' not in st.session_state:
    st.session_state.df = None
if 'graph' not in st.session_state:
    st.session_state.graph = None
if 'city_to_idx' not in st.session_state:
    st.session_state.city_to_idx = None

# -----------------------
# Data Loading and Processing
# -----------------------
@st.cache_data(show_spinner=False)
def load_data():
    """Load and clean the dataset"""
    try:
        # Load the dataset
        url = 'https://raw.githubusercontent.com/SSenitha/CCS3052_Advance_DSA/refs/heads/main/Cities_of_SriLanka.csv'
        df = pd.read_csv(url)
        
        # Handle different possible column name formats
        column_mapping = {
            'city id': 'city_id',
            'City ID': 'city_id',
            'cityid': 'city_id',
            'CityID': 'city_id'
        }
        
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                df.rename(columns={old_name: new_name}, inplace=True)
                break
        
        # Required columns
        required_cols = ['name_en', 'latitude', 'longitude']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
            return None
        
        # Clean the data
        df = df.dropna(subset=required_cols)
        df = df.drop_duplicates(subset=['latitude', 'longitude'], keep='first')
        df = df.reset_index(drop=True)
        
        # Create new city_id column
        df['city_id'] = range(len(df))
        
        # Ensure coordinates are numeric
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
        
        # Remove any rows with invalid coordinates
        df = df.dropna(subset=['latitude', 'longitude'])
        df = df.reset_index(drop=True)
        df['city_id'] = range(len(df))
        
        return df
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

@st.cache_data(show_spinner=False)
def build_city_graph(_df, k=8):
    """Build the graph structure for pathfinding"""
    try:
        n = len(_df)
        if n == 0:
            return None, None
        
        # Adjust k if necessary
        k = min(k, n)
        
        # Convert to radians for haversine calculation
        coords = np.radians(_df[['latitude', 'longitude']].values)
        
        # Build nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree', metric='haversine')
        nbrs.fit(coords)
        distances, indices = nbrs.kneighbors(coords)
        
        # Convert distances to kilometers
        distances = distances * 6371.0
        
        # Build adjacency lists
        adj_list = {i: [] for i in range(n)}
        
        for i in range(n):
            for j in range(1, k):  # Skip self (index 0)
                if j < len(indices[i]):
                    neighbor = indices[i][j]
                    dist = distances[i][j]
                    adj_list[i].append((neighbor, dist))
                    # Make it bidirectional
                    adj_list[neighbor].append((i, dist))
        
        # Remove duplicates and sort by distance
        for i in adj_list:
            seen = set()
            unique_neighbors = []
            for neighbor, dist in adj_list[i]:
                if neighbor not in seen and neighbor != i:
                    seen.add(neighbor)
                    unique_neighbors.append((neighbor, dist))
            adj_list[i] = sorted(unique_neighbors, key=lambda x: x[1])[:k-1]
        
        # Create city name mapping
        city_to_idx = {name.strip().lower(): idx for idx, name in enumerate(_df['name_en'])}
        
        return adj_list, city_to_idx
        
    except Exception as e:
        st.error(f"Error building graph: {str(e)}")
        return None, None

# -----------------------
# Pathfinding Algorithms
# -----------------------
def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate haversine distance between two points"""
    R = 6371.0  # Earth's radius in km
    
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    return R * c

def reconstruct_path(came_from, current):
    """Reconstruct path from came_from dictionary"""
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    return path[::-1]

def dijkstra_pathfind(graph, start, goal):
    """Dijkstra's algorithm implementation"""
    start_time = time.perf_counter()
    
    if start == goal:
        return [start], 0.0, time.perf_counter() - start_time
    
    # Priority queue: (distance, node)
    pq = [(0, start)]
    distances = {start: 0}
    came_from = {}
    visited = set()
    
    while pq:
        current_dist, current = heapq.heappop(pq)
        
        if current in visited:
            continue
            
        visited.add(current)
        
        if current == goal:
            path = reconstruct_path(came_from, goal)
            return path, distances[goal], time.perf_counter() - start_time
        
        for neighbor, weight in graph.get(current, []):
            if neighbor in visited:
                continue
                
            new_dist = current_dist + weight
            
            if neighbor not in distances or new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                came_from[neighbor] = current
                heapq.heappush(pq, (new_dist, neighbor))
    
    return None, float('inf'), time.perf_counter() - start_time

def a_star_pathfind(graph, start, goal, df):
    """A* algorithm implementation"""
    start_time = time.perf_counter()
    
    if start == goal:
        return [start], 0.0, time.perf_counter() - start_time
    
    def heuristic(node):
        return haversine_distance(
            df.iloc[node]['latitude'], df.iloc[node]['longitude'],
            df.iloc[goal]['latitude'], df.iloc[goal]['longitude']
        )
    
    # Priority queue: (f_score, g_score, node)
    pq = [(heuristic(start), 0, start)]
    g_scores = {start: 0}
    came_from = {}
    visited = set()
    
    while pq:
        f_score, g_score, current = heapq.heappop(pq)
        
        if current in visited:
            continue
            
        visited.add(current)
        
        if current == goal:
            path = reconstruct_path(came_from, goal)
            return path, g_scores[goal], time.perf_counter() - start_time
        
        for neighbor, weight in graph.get(current, []):
            if neighbor in visited:
                continue
                
            tentative_g = g_score + weight
            
            if neighbor not in g_scores or tentative_g < g_scores[neighbor]:
                g_scores[neighbor] = tentative_g
                came_from[neighbor] = current
                f_score = tentative_g + heuristic(neighbor)
                heapq.heappush(pq, (f_score, tentative_g, neighbor))
    
    return None, float('inf'), time.perf_counter() - start_time

def bellman_ford_pathfind(graph, start, goal, num_nodes):
    """Bellman-Ford algorithm implementation"""
    start_time = time.perf_counter()
    
    if start == goal:
        return [start], 0.0, time.perf_counter() - start_time
    
    # Initialize distances
    distances = {i: float('inf') for i in range(num_nodes)}
    distances[start] = 0
    predecessors = {}
    
    # Create edge list
    edges = []
    for u in graph:
        for v, weight in graph[u]:
            edges.append((u, v, weight))
    
    # Relax edges
    for _ in range(num_nodes - 1):
        updated = False
        for u, v, weight in edges:
            if distances[u] != float('inf') and distances[u] + weight < distances[v]:
                distances[v] = distances[u] + weight
                predecessors[v] = u
                updated = True
        if not updated:
            break
    
    # Reconstruct path
    if goal not in predecessors and goal != start:
        return None, float('inf'), time.perf_counter() - start_time
    
    path = reconstruct_path(predecessors, goal)
    return path, distances[goal], time.perf_counter() - start_time

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
        path, dist, time_taken = dijkstra_pathfind(graph, start_idx, end_idx)
        if path is not None:
            results.append(("Dijkstra", path, dist, time_taken))
    
    if algorithms.get("A*"):
        path, dist, time_taken = a_star_pathfind(graph, start_idx, end_idx, df)
        if path is not None:
            results.append(("A*", path, dist, time_taken))
    
    if algorithms.get("Bellman-Ford"):
        path, dist, time_taken = bellman_ford_pathfind(graph, start_idx, end_idx, len(df))
        if path is not None:
            results.append(("Bellman-Ford", path, dist, time_taken))
    
    if not results:
        return None, "No path found between selected cities."
    
    return results, None

# -----------------------
# Main Application
# -----------------------
def main():
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
            st.session_state.graph, st.session_state.city_to_idx = build_city_graph(st.session_state.df)
        
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
                    st.session_state.results = results
                    st.session_state.last_search = {
                        'start_city': start_city,
                        'end_city': end_city,
                        'start_idx': st.session_state.city_to_idx.get(start_city.lower().strip()),
                        'end_idx': st.session_state.city_to_idx.get(end_city.lower().strip())
                    }
                    st.success("✅ Pathfinding completed!")
    
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