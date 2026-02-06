import traci
import sumolib
import sys
import os

def run_simulation():
    # 1. Start the simulation in GUI mode so we can see it
    # (We use 'sumo-gui' to see it, later we use 'sumo' for fast training)
    sumo_cmd = ["sumo-gui", "-c", "sim.sumocfg"]

    # Start TraCI (The bridge between Python and SUMO)
    traci.start(sumo_cmd)

    print("🚀 Simulation started! Collecting data...")

    # 2. Get a list of all road IDs (Edges)
    all_edges = traci.edge.getIDList()
    # Filter out weird internal edges (intersections) that start with ":"
    road_edges = [edge for edge in all_edges if not edge.startswith(":")]
    print(f"🛣️  Monitoring {len(road_edges)} roads.")

    step = 0
    # Run for 500 simulation steps
    while step < 500:
        traci.simulationStep()  # Move simulation forward 1 step

        # --- DATA EXTRACTION (The "Eyes") ---
        total_waiting = 0

        # Check every road
        for edge_id in road_edges:
            # How many cars are waiting (speed < 0.1 m/s) on this road?
            waiting_cars = traci.edge.getLastStepHaltingNumber(edge_id)
            total_waiting += waiting_cars

        # Every 50 steps, print a status report
        if step % 50 == 0:
            print(f"⏱️ Step {step}: Total Cars Stuck in Traffic = {total_waiting}")

        step += 1

    traci.close()
    print("✅ Data collection finished.")

if __name__ == "__main__":
    run_simulation()