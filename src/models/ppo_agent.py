import os
import yaml
# pyrefly: ignore [missing-import]
from stable_baselines3 import PPO
# pyrefly: ignore [missing-import]
from stable_baselines3.common.callbacks import CheckpointCallback
# pyrefly: ignore [missing-import]
from stable_baselines3.common.env_util import make_vec_env
# pyrefly: ignore [missing-import]
from stable_baselines3.common.monitor import Monitor
# pyrefly: ignore [missing-import]
from stable_baselines3.common.callbacks import BaseCallback
from typing import Optional
import logging

from src.environment.f1_strategy_env import F1StrategyEnv

logger = logging.getLogger(__name__)

class F1PPOAgent:
    """
    Wrapper for Stable-Baselines3 PPO to manage the F1 Strategy RL training pipeline.
    """
    def __init__(self, config_path="config.yaml", circuit="Silverstone"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.circuit = circuit
        self.config_path = config_path
        self.train_config = self.config.get('training', {})
        self.model: Optional[PPO] = None
        self.env = None

    def build_env(self, n_envs=4):
        """
        Creates a vectorized environment for faster parallel training.
        """
        def make_env():
            env = F1StrategyEnv(config_path=self.config_path, circuit=self.circuit)
            env = Monitor(env) # Monitor wraps the env to log episode returns/lengths
            return env
            
        self.env = make_vec_env(make_env, n_envs=n_envs)
        return self.env

    def initialize_model(self, tensorboard_log="outputs/logs/"):
        """Initializes the PPO policy network with hyperparameters from config."""
        if self.env is None:
            self.build_env()
            
        self.model = PPO(
            "MlpPolicy",
            self.env,
            learning_rate=self.train_config.get('learning_rate', 3e-4),
            n_steps=self.train_config.get('n_steps', 2048),
            batch_size=self.train_config.get('batch_size', 64),
            n_epochs=self.train_config.get('n_epochs', 10),
            gamma=self.train_config.get('gamma', 0.99),
            gae_lambda=self.train_config.get('gae_lambda', 0.95),
            clip_range=self.train_config.get('clip_range', 0.2),
            ent_coef=self.train_config.get('ent_coef', 0.0),
            tensorboard_log=tensorboard_log,
            verbose=1
        )
        return self.model

    def train(self, total_timesteps: int, save_path="data/models/ppo_f1_agent"):
        """Executes the training loop and manages checkpoints."""
        if self.model is None:
            self.initialize_model()
            
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save a checkpoint every N steps
        checkpoint_freq = max(1, total_timesteps // (5 * self.env.num_envs))
        checkpoint_callback = CheckpointCallback(
            save_freq=checkpoint_freq, 
            save_path=os.path.dirname(save_path),
            name_prefix=f"ppo_{self.circuit}_ckpt"
        )
        
        logger.info(f"Starting PPO training for {total_timesteps} timesteps...")
        
        self.model.learn(
            total_timesteps=total_timesteps, 
            callback=checkpoint_callback,
            progress_bar=False 
        )
        
        # Save final model state
        self.model.save(save_path)
        logger.info(f"Final model saved to {save_path}.zip")

    def load(self, path: str):
        """Loads a pre-trained model."""
        if self.env is None:
            self.build_env(n_envs=1)
        self.model = PPO.load(path, env=self.env)
