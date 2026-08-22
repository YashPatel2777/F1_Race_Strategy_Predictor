import os
import yaml
import time
import logging
from src.models.ppo_agent import F1PPOAgent
from src.environment.race_simulator import RaceSimulator
from src.models.baselines import RandomAgent, OneStopAgent, RuleBasedAgent
from scripts.evaluate_agent import get_ppo_obs

# Suppress debug logging from third party libs during demo
logging.getLogger().setLevel(logging.WARNING)

def main():
    print("======================================================")
    print("       F1 RACE STRATEGY PREDICTOR - FINAL DEMO        ")
    print("======================================================\n")
    
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
        
    circuit = config['data']['circuit']
    ppo_path = os.path.join("data", "models", f"ppo_{circuit}_final.zip")
    
    if not os.path.exists(ppo_path):
        print(f"Error: {ppo_path} not found. Run train_agent.py first.")
        return
        
    print(f"Loading Supervised RL Agent for {circuit}...\n")
    ppo_wrapper = F1PPOAgent(circuit=circuit)
    ppo_wrapper.load(ppo_path)
    ppo_model = ppo_wrapper.model
    
    policies = {
        "Agent_1": ("PPO_RL_Agent", ppo_model),
        "Agent_2": ("OneStop_Bot", OneStopAgent(target_compound_id=3, pit_window_start=0.45)),
        "Agent_3": ("RuleBased_Bot", RuleBasedAgent(tire_age_threshold=0.5, sc_opportunistic_threshold=0.2)),
        "Agent_4": ("Random_Bot", RandomAgent(pit_probability=0.03)),
        "Agent_5": ("Conservative_Bot", RuleBasedAgent(tire_age_threshold=0.35, sc_opportunistic_threshold=0.1))
    }
    
    simulator = RaceSimulator(circuit=circuit)
    simulator.reset()
    
    # Initialize Grid
    print("--- STARTING GRID ---")
    for car in simulator.state.cars.values():
        # Start everyone on Softs so tire degradation forces strategic action quickly
        car.current_compound = "SOFT" 
        name = policies.get(car.driver_name, (car.driver_name,))[0]
        print(f"P{car.position}: {name} (SOFT)")
    
    print("\n--- LIGHTS OUT AND AWAY WE GO! ---")
    time.sleep(1)
    
    prev_leader = None
    was_sc = False
    was_vsc = False
    
    while simulator.state.lap_progress <= simulator.total_laps:
        lap = simulator.state.lap_progress
        
        # 1. Gather Actions
        actions = {}
        for agent_id, (name, policy) in policies.items():
            if agent_id not in simulator.state.cars:
                continue
            obs = get_ppo_obs(simulator, agent_id)
            action, _ = policy.predict(obs, deterministic=True)
            actions[agent_id] = int(action)
            
        # 2. Advance Physics
        simulator.step(actions)
        
        # 3. Generate Commentary
        commentary = []
        
        # Incidents
        if simulator.state.safety_car and not was_sc:
            commentary.append("[SAFETY CAR DEPLOYED] The pack is bunching up!")
            was_sc = True
        elif not simulator.state.safety_car and was_sc:
            commentary.append("[SAFETY CAR IN] Green flag racing resumes.")
            was_sc = False
            
        if simulator.state.vsc and not was_vsc:
            commentary.append("[VIRTUAL SAFETY CAR DEPLOYED] Delta times enforced.")
            was_vsc = True
        elif not simulator.state.vsc and was_vsc:
            commentary.append("[VSC ENDING] Green flag.")
            was_vsc = False
            
        # Pit stops
        for agent_id, action in actions.items():
            if action > 0:
                car = simulator.state.cars[agent_id]
                name = policies[agent_id][0]
                commentary.append(f"[PIT STOP] {name} dives into the pits for {car.current_compound} tires!")
                
        # Lead Changes
        current_leader = [c for c in simulator.state.cars.values() if c.position == 1][0]
        if prev_leader and prev_leader.driver_name != current_leader.driver_name:
            leader_name = policies[current_leader.driver_name][0]
            commentary.append(f"[LEAD CHANGE] {leader_name} takes P1!")
            
        prev_leader = current_leader
        
        # Print Commentary for the lap if anything interesting happened
        if commentary:
            print(f"\n[Lap {lap}/{simulator.total_laps}]")
            for line in commentary:
                print(f"  {line}")
            time.sleep(0.3) # Slight delay for dramatic effect
            
    print("\n[CHEQUERED FLAG!]")
    print("\n======================================================")
    print("                 FINAL CLASSIFICATION")
    print("======================================================")
    
    sorted_cars = sorted(simulator.state.cars.values(), key=lambda c: c.position)
    winner_time = sorted_cars[0].total_race_time
    
    print(f"{'POS':<4} | {'DRIVER':<20} | {'COMPOUND':<8} | {'PITS':<5} | {'GAP'}")
    print("-" * 62)
    
    for car in sorted_cars:
        name = policies.get(car.driver_name, (car.driver_name,))[0]
        if car.position == 1:
            gap_str = "Winner"
        else:
            gap = car.total_race_time - winner_time
            gap_str = f"+{gap:.3f}s"
            
        print(f"P{car.position:<3} | {name:<20} | {car.current_compound:<8} | {car.pit_count:<5} | {gap_str}")
        
if __name__ == "__main__":
    main()
