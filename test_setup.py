import os
import sys
import sumolib
import traci

def check_sumo():
    # 1. Check if SUMO_HOME is found
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
        print(f"✅ SUMO_HOME found at: {os.environ['SUMO_HOME']}")
    else:
        print("❌ Error: SUMO_HOME variable not found.")
        print("   Did you check 'Add to PATH' during installation?")
        print("   Try restarting VS Code.")
        return

    # 2. Check if we can import the traffic control library
    try:
        print(f"✅ TraCI (Traffic Control Interface) version: {traci.__file__}")
        print("🚀 Ready for Phase 2!")
    except ImportError:
        print("❌ Error: Could not import traci.")

if __name__ == "__main__":
    check_sumo()