# pyrefly: ignore [missing-import]
import pytest
import numpy as np
from src.environment.f1_strategy_env import F1StrategyEnv

@pytest.fixture
def env():
    return F1StrategyEnv(circuit="Silverstone")

def test_env_initialization(env):
    obs, info = env.reset()
    
    assert env.observation_space.shape == (10,)
    assert env.action_space.n == 4
    assert len(obs) == 10
    assert np.all((obs >= 0.0) & (obs <= 1.0)), "Observation space must be strictly normalized [0, 1]"

def test_env_step_mechanics(env):
    env.reset()
    
    # Stay out
    obs, reward, terminated, truncated, info = env.step(0)
    
    assert not terminated
    assert isinstance(reward, float)
    assert len(obs) == 10

def test_reward_penalties(env):
    env.reset()
    # Force the ego car to have brand new tires
    env.simulator.state.cars[env.ego_agent_id].tire_age = 1
    
    # Attempting to pit on 1-lap old tires should yield a massive penalty
    obs, reward, terminated, truncated, info = env.step(1)
    
    assert reward <= -5.0, "Agent was not properly penalized for an unnecessary pit stop"

def test_episode_termination(env):
    env.reset()
    # Fast forward to end of race
    env.simulator.state.lap_progress = env.simulator.total_laps
    
    obs, reward, terminated, truncated, info = env.step(0)
    
    assert terminated is True
    # The final reward should be applied
    assert reward != 0.0 
