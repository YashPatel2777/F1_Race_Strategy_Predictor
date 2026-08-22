import os
import yaml
import logging
from src.models.ppo_agent import F1PPOAgent

logging.basicConfig(level=logging.INFO, format='%(message)s')

def main():
    print("======================================")
    print("  TRAINING RL AGENT (PPO)")
    print("======================================\n")
    
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
        
    circuit = config['data']['circuit']
    total_timesteps = config['training'].get('total_timesteps', 500000)
    
    print(f"Target Circuit: {circuit}")
    print(f"Total Timesteps: {total_timesteps}")
    print(f"Learning Rate: {config['training'].get('learning_rate', 'default')}")
    print(f"Batch Size: {config['training'].get('batch_size', 'default')}")
    
    # Initialize the wrapper class
    agent = F1PPOAgent(config_path="config.yaml", circuit=circuit)
    
    # We build a vectorized environment (e.g., 4 parallel racing instances) 
    # to dramatically speed up data collection and gradient updates.
    n_envs = 4
    print(f"\nBuilding Vectorized Environment (N={n_envs} Parallel Races)...")
    agent.build_env(n_envs=n_envs)
    
    print("Initializing PPO Model...")
    agent.initialize_model(tensorboard_log="outputs/logs/")
    
    print("\nStarting main training loop...")
    
    final_model_path = os.path.join("data", "models", f"ppo_{circuit}_final")
    agent.train(total_timesteps=total_timesteps, save_path=final_model_path)
    
    print(f"\n[SUCCESS] Training complete. Final model weights saved to {final_model_path}.zip")

if __name__ == "__main__":
    main()
