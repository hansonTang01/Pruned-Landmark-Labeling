import osmnx as ox
G = ox.graph_from_place("北京市", network_type='drive_service',simplify=True)
ox.plot_graph(G)