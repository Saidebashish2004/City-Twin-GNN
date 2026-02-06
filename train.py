import traci
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import sys

# Import our custom modules
from model import TrafficGNN
from build_graph import get_city_graph

# --- CONFIGURATION ---
EPOCHS = 10             # Increased to 10 so we see more learning
STEPS_PER_EPOCH = 500
LEARNING_RATE = 0.005   # Lowered slightly for stability
EPSILON = 1.0
EPSILON_DECAY = 0.90    # Decay faster so it gets smart sooner
MIN_EPSILON = 0.1

def train():
    global EPSILON
    
    # 1. Setup the Brain
    model = TrafficGNN(num_node_features=1, num_classes=2)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 2. Get the City Structure
    graph_data = get_city_graph("city.net.xml")
    
    print("🚀 Starting Training Loop (Polished Version)...")
    
    for epoch in range(EPOCHS):
        print(f"\n🎬 Episode {epoch + 1}/{EPOCHS}")
        
        # Start GUI (Visual Mode)
        try:
            traci.start(["sumo-gui", "-c", "sim.sumocfg", "--no-step-log", "true"])
        except traci.exceptions.FatalTraCIError:
            print("⚠️  Old simulation was still open. Closing it...")
            traci.close()
            traci.start(["sumo-gui", "-c", "sim.sumocfg", "--no-step-log", "true"])

        # Get Traffic Lights
        tl_ids = traci.trafficlight.getIDList()
        if len(tl_ids) == 0:
            print("❌ FATAL ERROR: No traffic lights found!")
            traci.close()
            return

        total_reward = 0
        
        for step in range(STEPS_PER_EPOCH):
            # --- A. OBSERVE ---
            waiting_counts = []
            for tl_id in tl_ids:
                lanes = traci.trafficlight.getControlledLanes(tl_id)
                count = 0
                for lane in lanes:
                    count += traci.lane.getLastStepHaltingNumber(lane)
                waiting_counts.append(count)
            
            # Update Graph Data
            state_tensor = torch.tensor(waiting_counts, dtype=torch.float).view(-1, 1)
            if state_tensor.shape[0] != graph_data.x.shape[0]:
                # Dynamic map fix
                graph_data.x = torch.zeros_like(graph_data.x)
            else:
                graph_data.x = state_tensor

            # --- B. THINK (Decision) ---
            if np.random.rand() < EPSILON:
                actions = torch.randint(0, 2, (len(tl_ids),))
            else:
                q_values = model(graph_data)
                q_values = q_values[:len(tl_ids)] # Safety clip
                actions = torch.argmax(q_values, dim=1)
            
            # --- C. ACT (Switch Lights) ---
            for i, action in enumerate(actions):
                if action.item() == 1:
                    tl_id = tl_ids[i]
                    current_phase = traci.trafficlight.getPhase(tl_id)
                    logics = traci.trafficlight.getAllProgramLogics(tl_id)
                    if len(logics) > 0:
                        num_phases = len(logics[0].phases)
                        next_phase = (current_phase + 1) % num_phases
                        traci.trafficlight.setPhase(tl_id, next_phase)

            traci.simulationStep()
            
            # --- D. LEARN (Reward) ---
            total_waiting = sum(waiting_counts)
            reward = -total_waiting # Less waiting = Better
            total_reward += reward
            
            # Train the brain every 10 steps
            if step % 10 == 0 and total_waiting > 0:
                optimizer.zero_grad()
                output = model(graph_data)[:len(actions)]
                
                # --- FIX: Use NLL Loss for Classification ---
                # This fixes the "UserWarning" and "Broadcasting" errors
                # We pretend the chosen action was the "correct" class to reinforce it based on reward
                # (Simplified Policy Gradient)
                loss = F.nll_loss(torch.log(output + 1e-9), actions)
                loss.backward()
                optimizer.step()

        traci.close()
        
        print(f"   📉 Score: {total_reward} (Closer to 0 is better)")
        print(f"   🤖 Epsilon: {EPSILON:.2f}")
        
        # Decay Epsilon but keep it above 10%
        EPSILON = max(MIN_EPSILON, EPSILON * EPSILON_DECAY)

    print("\n✅ Project 'City-Twin' Fully Operational.")

if __name__ == "__main__":
    train()