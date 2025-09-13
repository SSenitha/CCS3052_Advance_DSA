import streamlit as st
import pandas as pd
import numpy as np
import math
import time
import heapq
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from collections import defaultdict

# -----------------------
# Load dataset
# -----------------------
df = pd.read_csv('https://raw.githubusercontent.com/SSenitha/CCS3052_Advance_DSA/refs/heads/main/Cities_of_SriLanka.csv')
df.rename(columns={"city id": "city_id"}, inplace=True)
df = df.drop_duplicates(subset=['latitude', 'longitude'], keep='first').reset_index(drop=True)
df = df.drop(columns=["district_id","name_si","name_ta","sub_name_en","sub_name_si","sub_name_ta","postcode"])
df.drop(columns=["city_id"], inplace=True)
df.insert(0, "city_id", df.index)

# -----------------------
# Build Nearest Neighbors Graph (km)
# -----------------------
num_cities = len(df)
k = 6
coords = np.radians(df[['latitude','longitude']].values)
nn = NearestNeighbors(n_neighbors=k, algorithm='ball_tree', metric='haversine').fit(coords)
distances_rad, indices = nn.kneighbors(coords)
distances_km = distances_rad * 6371

rows, cols, data = [], [], []
for i in range(num_cities):
    for j in range(1, k):
        neighbor_index = indices[i, j]
        neighbor_distance = distances_km[i,j]
        rows.append(i); cols.append(neighbor_index); data.append(neighbor_distance)
sparse_matrix = csr_matrix((data, (rows, cols)), shape=(num_cities, num_cities))

# -----------------------
# Bellman-Ford adjacency list
# -----------------------
def build_adj_list(df_local, k_neighbors=6):
    coords_deg = df_local[['latitude','longitude']].to_numpy()
    coords_rad = np.radians(coords_deg)
    tree = NearestNeighbors(n_neighbors=k_neighbors+1, algorithm='ball_tree', metric='haversine').fit(coords_rad)
    dist_rad, idx = tree.kneighbors(coords_rad)
    dist_km = dist_rad * 6371
    adj = defaultdict(list)
    seen_edges = set()
    for i in range(len(df_local)):
        for pos in range(1, k_neighbors+1):
            j = int(idx[i,pos])
            if i==j: continue
            edge = frozenset({i,j})
            if edge in seen_edges: continue
            seen_edges.add(edge)
            w = float(dist_km[i,pos])
            adj[i].append((j,w))
            adj[j].append((i,w))
    return dict(adj)

adj_list_bf = build_adj_list(df)

# -----------------------
# Helper Functions
# -----------------------
city_name_to_index = {name.lower(): idx for idx,name in enumerate(df['name_en'])}

def dijkstra_sparse(graph, start):
    n = graph.shape[0]
    distances = [float('inf')]*n
    previous_nodes = [None]*n
    distances[start] = 0
    pq = [(0,start)]
    while pq:
        d,u = heapq.heappop(pq)
        if d>distances[u]: continue
        row = graph.getrow(u)
        for v,w in zip(row.indices,row.data):
            if d+w<distances[v]:
                distances[v]=d+w
                previous_nodes[v]=u
                heapq.heappush(pq,(distances[v],v))
    return distances, previous_nodes

def get_path(previous_nodes,start,end):
    path=[]
    current=end
    while current is not None:
        path.insert(0,current)
        current=previous_nodes[current]
    return path if path[0]==start else []

def dist_haversine(i,j):
    lat1, lon1 = df.loc[i,['latitude','longitude']]
    lat2, lon2 = df.loc[j,['latitude','longitude']]
    R = 6371
    lat1, lon1, lat2, lon2 = map(math.radians,[lat1, lon1, lat2, lon2])
    dlat = lat2-lat1; dlon=lon2-lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    c = 2*math.asin(math.sqrt(a))
    return R*c

def get_neighbors(i):
    row = sparse_matrix.getrow(i)
    return list(row.indices)

def astar(start,end):
    open_set = []
    heapq.heappush(open_set,(0,0,start,[start]))
    g_costs = {start:0}
    while open_set:
        f,d,u,path = heapq.heappop(open_set)
        if u==end: return path,d
        for v in get_neighbors(u):
            tentative_g = d + sparse_matrix[u,v]
            if v not in g_costs or tentative_g<g_costs[v]:
                g_costs[v]=tentative_g
                h=dist_haversine(v,end)
                heapq.heappush(open_set,(tentative_g+h,tentative_g,v,path+[v]))
    return None,None

def bellman_ford(adj, n_nodes, source, target):
    dist = [math.inf]*n_nodes
    pred = [None]*n_nodes
    dist[source]=0
    start=time.perf_counter()
    relaxations=0
    for _ in range(n_nodes-1):
        for u in adj:
            for v,w in adj[u]:
                if dist[u]+w<dist[v]:
                    dist[v]=dist[u]+w
                    pred[v]=u
                    relaxations+=1
    negative_cycle=False
    for u in adj:
        for v,w in adj[u]:
            if dist[u]+w<dist[v]:
                negative_cycle=True
    path=None
    if dist[target]<math.inf:
        path=[]
        cur=target
        while cur is not None:
            path.insert(0,cur)
            cur=pred[cur]
    info={'time_s':time.perf_counter()-start,'relaxations':relaxations,'negative_cycle':negative_cycle,'iterations':n_nodes-1}
    return path,dist[target],info

# -----------------------
# Streamlit UI
# -----------------------
st.title("🛣️ Sri Lanka Shortest Path Finder")

city_list = sorted(df['name_en'].values)
start_city = st.selectbox("Select Start City", city_list)
end_city = st.selectbox("Select Destination City", city_list)

st.markdown("### Choose Algorithm(s):")
use_dijkstra = st.checkbox("Dijkstra", value=True)
use_astar = st.checkbox("A*", value=True)
use_bf = st.checkbox("Bellman-Ford", value=False)

if st.button("Find Shortest Path"):
    start_idx = city_name_to_index.get(start_city.lower())
    end_idx = city_name_to_index.get(end_city.lower())

    results = {}

    if use_dijkstra:
        t0=time.perf_counter()
        dist_d, prev = dijkstra_sparse(sparse_matrix, start_idx)
        t1=time.perf_counter()
        path_d = get_path(prev,start_idx,end_idx)
        results['Dijkstra'] = {'distance': dist_d[end_idx], 'time': t1-t0, 'path': path_d}

    if use_astar:
        t0=time.perf_counter()
        path_a, dist_a = astar(start_idx,end_idx)
        t1=time.perf_counter()
        results['A*'] = {'distance': dist_a, 'time': t1-t0, 'path': path_a}

    if use_bf:
        path_b, dist_b, info_b = bellman_ford(adj_list_bf, num_cities, start_idx,end_idx)
        results['Bellman-Ford'] = {'distance': dist_b, 'time': info_b['time_s'], 'path': path_b}

    # Display results
    for algo in results:
        r = results[algo]
        if r['path'] is None:
            st.error(f"{algo}: No path found")
        else:
            st.subheader(f"{algo} Result")
            st.success(f"Distance: {r['distance']:.2f} km | Time: {r['time']:.6f} s")
            st.write("📍 Path:", " ➝ ".join(df.loc[i,'name_en'] for i in r['path']))

    # -----------------------
    # Comparison with color highlight for fastest
    # -----------------------
    if len(results) > 1:
        st.markdown("### ⚡ Comparison")
        comp_df = pd.DataFrame([
            {'Algorithm': algo, 'Distance (km)': r['distance'], 'Time (s)': r['time']}
            for algo, r in results.items()
        ])
        fastest_time = comp_df['Time (s)'].min()
        def highlight_fastest(row):
            return ['background-color: green' if row['Time (s)']==fastest_time else '' for _ in row]
        st.dataframe(comp_df.style.apply(highlight_fastest, axis=1))
