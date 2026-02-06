import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class TrafficGNN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(TrafficGNN, self).__init__()
        
        # Layer 1: Graph Convolution
        # Takes info from neighbors and aggregates it
        self.conv1 = GCNConv(num_node_features, 16)
        
        # Layer 2: Another Graph Convolution
        # Allows info to hop 2 nodes away (predicting future flow)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # 1. Pass messages between intersections
        x = self.conv1(x, edge_index)
        x = F.relu(x)  # Activation function (add non-linearity)
        x = F.dropout(x, training=self.training)

        # 2. Second hop of messages
        x = self.conv2(x, edge_index)

        # 3. Output a decision for EACH intersection
        # (e.g., Probability of needing a Green Light)
        return F.softmax(x, dim=1)

if __name__ == "__main__":
    # Quick Test to verify the brain shape
    print("🧠 Initializing TrafficGNN...")
    
    # Input Features = 1 (Number of waiting cars)
    # Output Classes = 2 (Action: 0=Keep Red, 1=Turn Green)
    model = TrafficGNN(num_node_features=1, num_classes=2)
    
    print(model)
    print("\n✅ Success: Model architecture is ready!")