# pyrefly: ignore [missing-import]
import pytest
import numpy as np
from src.models.baselines import RandomAgent, OneStopAgent, RuleBasedAgent

@pytest.fixture
def dummy_obs():
    # lap_norm, tire_age_norm, comp_val, gap_a, gap_b, pos, pit_norm, stint_norm, sc, vsc
    return np.array([0.1, 0.1, 0.5, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0])

def test_random_agent(dummy_obs):
    agent = RandomAgent(pit_probability=1.0)
    action, _ = agent.predict(dummy_obs)
    assert action in [1, 2, 3], "Agent with 1.0 pit prob must pit"
    
    agent = RandomAgent(pit_probability=0.0)
    action, _ = agent.predict(dummy_obs)
    assert action == 0, "Agent with 0.0 pit prob must stay out"

def test_one_stop_agent(dummy_obs):
    agent = OneStopAgent(target_compound_id=3, pit_window_start=0.5)
    
    # Early in race, no pits
    dummy_obs[0] = 0.4 # lap 40%
    dummy_obs[6] = 0.0 # 0 pits
    action, _ = agent.predict(dummy_obs)
    assert action == 0
    
    # Entered pit window
    dummy_obs[0] = 0.51 # lap 51%
    action, _ = agent.predict(dummy_obs)
    assert action == 3
    
    # Already pitted
    dummy_obs[6] = 0.25 # 1 pit (1/4 normalized)
    action, _ = agent.predict(dummy_obs)
    assert action == 0

def test_rule_based_agent(dummy_obs):
    agent = RuleBasedAgent(tire_age_threshold=0.5, sc_opportunistic_threshold=0.2)
    
    # Normal racing, fresh tires
    dummy_obs[1] = 0.1 # Tire age 10%
    dummy_obs[8] = 0.0 # No SC
    action, _ = agent.predict(dummy_obs)
    assert action == 0
    
    # Dead tires
    dummy_obs[1] = 0.6 # Tire age 60%
    action, _ = agent.predict(dummy_obs)
    assert action == 3
    
    # Fresh tires, but SC
    dummy_obs[1] = 0.1
    dummy_obs[8] = 1.0 # SC
    action, _ = agent.predict(dummy_obs)
    assert action == 0 # Too fresh to pit opportunistically
    
    # Somewhat old tires + SC
    dummy_obs[1] = 0.25 # Over 0.2 threshold
    action, _ = agent.predict(dummy_obs)
    assert action == 3 # Should pit opportunistically
