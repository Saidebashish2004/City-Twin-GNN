import sumolib
import torch
from torch_geometric.data import Data

def get_city_graph(net_file="city.net.xml"):
    print(f"📂 Loading map: {net_file}...")
    
    # 1. Read the SUMO Network file
    net = sumolib.net.readNet(net_file)
    
    # 2. Get all Junctions (Nodes)
    # We filter out internal junctions (start with ":") to keep it simple
    all_nodes = net.getNodes()
    valid_nodes = [n for n in all_nodes if n.getType() != "internal"]
    
    # Create a mapping: "Junction1" -> Index 0
    node_mapping = {node.getID(): i for i, node in enumerate(valid_nodes)}
    
    print(f"📍 Found {len(valid_nodes)} Intersections (Nodes)")

    # 3. Build the Connections (Edges)
    # GNNs need to know: [Source Node, Target Node]
    sources = []
    targets = []
    
    all_edges = net.getEdges()
    count = 0
    
    for edge in all_edges:
        # Get start and end points
        from_id = edge.getFromNode().getID()
        to_id = edge.getToNode().getID()
        
        # Only add if both are valid intersections
        if from_id in node_mapping and to_id in node_mapping:
            u = node_mapping[from_id] # Source Index
            v = node_mapping[to_id]   # Target Index
            
            sources.append(u)
            targets.append(v)
            count += 1

    # 4. Convert to PyTorch Tensors
    # edge_index is the core "Skeleton" of the graph
    edge_index = torch.tensor([sources, targets], dtype=torch.long)
    
    # Dummy features for now (just ones)
    x = torch.ones((len(valid_nodes), 1), dtype=torch.float)

    # Create the Data Object
    data = Data(x=x, edge_index=edge_index)
    
    return data

if __name__ == "__main__":
    try:
        graph = get_city_graph()
        print("\n✅ SUCCESS: City converted to Graph!")
        print("-" * 30)
        print(f"🧠 Graph Structure: {graph}")
        print(f"🔗 Total Connections (Edges): {graph.num_edges}")
        print(f"🚦 Total Junctions (Nodes): {graph.num_nodes}")
        print("-" * 30)
    except Exception as e:
        print(f"❌ Error: {e}")