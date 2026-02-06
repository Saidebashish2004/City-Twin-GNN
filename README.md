# 🚦 City-Twin: Autonomous Traffic Control via Graph Neural Networks

**City-Twin** is a Research-Grade AI system that optimizes city-wide traffic flow using **Deep Reinforcement Learning (DQN)** and **Graph Neural Networks (GNNs)**.

Unlike standard traffic lights that use fixed timers, City-Twin uses a "Digital Twin" simulation to observe congestion in real-time and autonomously adjusts signal phases to clear jams.

---

## 🧠 The Architecture

### 1. The Environment (Digital Twin)
* **Engine:** [Eclipse SUMO](https://eclipse.dev/sumo/) (Simulation of Urban MObility).
* **Grid:** A dynamically generated 3x3 city grid with 9 intersections and smart traffic lights.
* **Traffic:** Stochastic vehicle generation to simulate rush-hour variability.

### 2. The Brain (Graph Neural Network)
Instead of treating the city as an image (CNN), we model it as a **Graph**:
* **Nodes:** Intersections (Traffic Lights).
* **Edges:** Roads connecting them.
* **State:** The GNN aggregates "message passing" data from neighboring intersections to predict flow.
* **Model:** 2-Layer `GCNConv` (Graph Convolutional Network) using PyTorch Geometric.

---

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **AI/ML:** PyTorch, PyTorch Geometric (PyG)
* **Simulation:** SUMO, TraCI (Traffic Control Interface)
* **Optimization:** Deep Q-Learning (DQN) with Epsilon-Greedy Strategy

## 🚀 How to Run
1.  **Install SUMO:** Ensure Eclipse SUMO is installed and added to PATH.
2.  **Install Dependencies:**
    ```bash
    pip install torch torch_geometric sumolib traci
    ```
3.  **Train the AI:**
    ```bash
    python train.py
    ```
    *This will launch the GUI and show the AI controlling lights in real-time.*

## 📊 Results
* **Congestion Score:** The system is rewarded for minimizing total waiting time across all 9 intersections.
* **Learning:** Uses an Epsilon-Decay strategy to transition from exploration (random switching) to exploitation (optimized flow).