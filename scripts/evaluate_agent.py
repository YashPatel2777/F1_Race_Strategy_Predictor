import os
import yaml
import numpy as np
import logging
from collections import defaultdict
from src.models.ppo_agent import F1PPOAgent
from src.environment.race_simulator import RaceSimulator
from src.models.baselines import RandomAgent, OneStopAgent, RuleBasedAgent

logging.basicConfig(level=logging.INFO, format='%(message)s')

def get_ppo_obs(simulator, agent_id):
    """
    Constructs the normalized observation vector identically to the Gym environment 
    so the PPO model and baselines can make decisions.
    """
    car = simulator.state.cars[agent_id]
    state = simulator.state
    
    lap_norm = state.lap_progress / state.total_laps
    tire_age_norm = min(1.0, car.tire_age / 50.0)
    
    comp_map = {"SOFT": 0.0, "MEDIUM": 0.5, "HARD": 1.0}
    compound_val = comp_map.get(car.current_compound, 0.5)
    
    gap_ahead_norm = min(1.0, car.gap_ahead / 20.0)
    gap_behind_norm = min(1.0, car.gap_behind / 20.0)
    pos_norm = (car.position - 1) / (simulator.num_cars - 1)
    
    pit_norm = min(1.0, car.pit_count / 4.0)
    stint_norm = min(1.0, car.stint_number / 5.0)
    
    sc_val = 1.0 if state.safety_car else 0.0
    vsc_val = 1.0 if state.vsc else 0.0
    
    return np.array([
        lap_norm, tire_age_norm, compound_val, gap_ahead_norm, gap_behind_norm,
        pos_norm, pit_norm, stint_norm, sc_val, vsc_val
    ], dtype=np.float32)

def main():
    print("======================================")
    print("  EVALUATING RL STRATEGY VS BASELINES")
    print("======================================\n")
    
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
        
    circuit = config['data']['circuit']
    
    ppo_path = os.path.join("data", "models", f"ppo_{circuit}_final.zip")
    if not os.path.exists(ppo_path):
        print(f"Error: {ppo_path} not found. Run train_agent.py first.")
        return
        
    print("Loading Trained PPO Model...")
    # Initialize PPO but avoid building vectorized envs since we are simulating manually
    ppo_wrapper = F1PPOAgent(circuit=circuit)
    ppo_wrapper.load(ppo_path)
    ppo_model = ppo_wrapper.model
    
    # Map agents to different AI architectures
    policies = {
        "Agent_1": ("PPO_RL_Agent", ppo_model),
        "Agent_2": ("OneStop_Bot", OneStopAgent(target_compound_id=3, pit_window_start=0.45)),
        "Agent_3": ("RuleBased_Bot", RuleBasedAgent(tire_age_threshold=0.5, sc_opportunistic_threshold=0.2)),
        "Agent_4": ("Random_Bot", RandomAgent(pit_probability=0.03)),
        "Agent_5": ("Conservative_Bot", RuleBasedAgent(tire_age_threshold=0.35, sc_opportunistic_threshold=0.1))
    }
    
    num_races = 100
    print(f"\nSimulating {num_races} races at {circuit}...")
    
    wins = defaultdict(int)
    total_positions = defaultdict(list)
    total_pits = defaultdict(list)
    
    simulator = RaceSimulator(circuit=circuit)
    
    # Run Evaluation Loop
    for race_idx in range(num_races):
        simulator.reset()
        
        # Override initial compounds so everyone starts on SOFT to make it a fair, level strategy fight
        for car in simulator.state.cars.values():
            car.current_compound = "SOFT"
            
        while simulator.state.lap_progress <= simulator.total_laps:
            actions = {}
            for agent_id, (policy_name, policy) in policies.items():
                if agent_id not in simulator.state.cars:
                    continue # In case config requests fewer than 5 cars
                    
                obs = get_ppo_obs(simulator, agent_id)
                
                # Predict action
                action, _ = policy.predict(obs, deterministic=True)
                actions[agent_id] = int(action)
                
            simulator.step(actions)
            
        # Race Finished. Log metrics.
        for agent_id, (policy_name, _) in policies.items():
            if agent_id not in simulator.state.cars:
                continue
            car = simulator.state.cars[agent_id]
            
            if car.position == 1:
                wins[policy_name] += 1
                
            total_positions[policy_name].append(car.position)
            total_pits[policy_name].append(car.pit_count)
            
    # Print Final Report
    print("\n=================================================================")
    print("                 FINAL RACE EVALUATION REPORT")
    print("=================================================================\n")
    
    print(f"{'Strategy Name':<25} | {'Win Rate':<10} | {'Avg Pos':<10} | {'Avg Pits':<10}")
    print("-" * 65)
    
    for policy_name in [p[0] for p in policies.values()]:
        if policy_name not in total_positions:
            continue
        win_rate = (wins[policy_name] / num_races) * 100
        avg_pos = np.mean(total_positions[policy_name])
        avg_pits = np.mean(total_pits[policy_name])
        
        print(f"{policy_name:<25} | {win_rate:>8.1f}% | {avg_pos:>8.1f}   | {avg_pits:>8.1f}")
        
    print("\n[SUCCESS] Evaluation complete.")

if __name__ == "__main__":
    main()
