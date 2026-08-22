import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import logging
from src.models.ppo_agent import F1PPOAgent
from src.environment.race_simulator import RaceSimulator
from src.models.baselines import RandomAgent, OneStopAgent, RuleBasedAgent
from scripts.evaluate_agent import get_ppo_obs

logging.basicConfig(level=logging.INFO, format='%(message)s')

def main():
    print("======================================")
    print("  VISUALIZING RACE STRATEGY")
    print("======================================\n")
    
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
        
    circuit = config['data']['circuit']
    ppo_path = os.path.join("data", "models", f"ppo_{circuit}_final.zip")
    
    if not os.path.exists(ppo_path):
        print(f"Error: {ppo_path} not found. Run train_agent.py first.")
        return
    
    # Load RL Model
    print("Loading Trained PPO Model...")
    ppo_wrapper = F1PPOAgent(circuit=circuit)
    ppo_wrapper.load(ppo_path)
    ppo_model = ppo_wrapper.model
    
    policies = {
        "Agent_1": ("PPO_RL", ppo_model),
        "Agent_2": ("OneStop", OneStopAgent(target_compound_id=3, pit_window_start=0.45)),
        "Agent_3": ("RuleBased", RuleBasedAgent(tire_age_threshold=0.5, sc_opportunistic_threshold=0.2)),
        "Agent_4": ("Random", RandomAgent(pit_probability=0.02)),
        "Agent_5": ("Conservative", RuleBasedAgent(tire_age_threshold=0.35, sc_opportunistic_threshold=0.1))
    }
    
    simulator = RaceSimulator(circuit=circuit)
    simulator.reset()
    
    for car in simulator.state.cars.values():
        car.current_compound = "SOFT"
        
    # Storage for Visualization Data
    # Telemetry: agent_id -> list of lap dicts
    telemetry = {agent_id: [] for agent_id in policies.keys() if agent_id in simulator.state.cars}
    
    # Record Grid Start
    for agent_id in telemetry.keys():
        car = simulator.state.cars[agent_id]
        telemetry[agent_id].append({
            'lap': 0,
            'race_time': car.total_race_time,
            'compound': car.current_compound,
            'is_pit': False
        })
        
    print(f"Simulating telemetry for 1 full race at {circuit}...")
    
    # Race Execution Loop
    while simulator.state.lap_progress <= simulator.total_laps:
        actions = {}
        for agent_id, (name, policy) in policies.items():
            if agent_id not in simulator.state.cars:
                continue
            obs = get_ppo_obs(simulator, agent_id)
            action, _ = policy.predict(obs, deterministic=True)
            actions[agent_id] = int(action)
            
        simulator.step(actions)
        
        # Record Telemetry
        for agent_id in telemetry.keys():
            car = simulator.state.cars[agent_id]
            telemetry[agent_id].append({
                'lap': simulator.state.lap_progress - 1, 
                'race_time': car.total_race_time,
                'compound': car.current_compound,
                'is_pit': actions[agent_id] > 0
            })
            
    print("Rendering Matplotlib strategy graphs...")
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Determine the Leader's time at every single lap to calculate Gap-to-Leader
    laps = list(range(simulator.total_laps + 1))
    leader_times = []
    for l in laps:
        times = [telemetry[a][l]['race_time'] for a in telemetry.keys()]
        leader_times.append(min(times))
        
    # F1 TV Graphics Standard Colors
    compound_colors = {
        'SOFT': '#FF3333',   # Red
        'MEDIUM': '#FFFF33', # Yellow
        'HARD': '#FFFFFF'    # White
    }
    
    for agent_id, (name, _) in policies.items():
        if agent_id not in telemetry:
            continue
            
        agent_telem = telemetry[agent_id]
        x_laps = []
        y_gaps = []
        colors = []
        pit_laps = []
        pit_gaps = []
        
        for l in laps:
            data = agent_telem[l]
            gap = data['race_time'] - leader_times[l]
            x_laps.append(l)
            y_gaps.append(gap)
            colors.append(compound_colors.get(data['compound'], 'gray'))
            
            if data['is_pit']:
                pit_laps.append(l)
                pit_gaps.append(gap)
                
        # Draw multi-colored line segments for stints
        for i in range(len(x_laps)-1):
            ax.plot(x_laps[i:i+2], y_gaps[i:i+2], color=colors[i], linewidth=2.5, solid_capstyle='round')
            
        # Hidden plot just to populate the legend with the agent name
        ax.plot([], [], color='gray', label=name, linewidth=2)
        
        # Pit stop markers (Cyan 'X')
        if pit_laps:
            ax.scatter(pit_laps, pit_gaps, color='#00FFFF', marker='X', s=150, zorder=5, label=f'Pit ({name})')
            
    # In F1 gap charts, Leader is at Y=0 (Top), cars behind drop downwards (Y increases)
    ax.invert_yaxis() 
    
    ax.set_title(f'F1 Race Strategy Visualization - {circuit} (Gap to Leader)', fontsize=18, fontweight='bold')
    ax.set_xlabel('Lap', fontsize=14)
    ax.set_ylabel('Gap to Leader (Seconds)', fontsize=14)
    ax.grid(True, alpha=0.15)
    
    # Legend deduplication and positioning
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=11)
    
    plt.tight_layout()
    
    out_dir = os.path.join('outputs', 'plots', 'races')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'strategy_sim_{circuit}.png')
    
    plt.savefig(out_path, dpi=300)
    print(f"\n[SUCCESS] Race visualization saved to: {out_path}")

if __name__ == "__main__":
    main()
