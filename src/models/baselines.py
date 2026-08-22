import random
import numpy as np

class RandomAgent:
    """
    Baseline: Picks random valid actions.
    Mostly stays out, occasionally pits randomly.
    """
    def __init__(self, pit_probability=0.05):
        self.pit_probability = pit_probability

    def predict(self, obs, state=None, deterministic=False):
        # Action space: 0 (Stay out), 1 (Soft), 2 (Medium), 3 (Hard)
        if random.random() < self.pit_probability:
            action = random.randint(1, 3)
        else:
            action = 0
        return action, None

class OneStopAgent:
    """
    Baseline: Pits exactly once near the middle of the race (e.g., Medium -> Hard).
    """
    def __init__(self, target_compound_id=3, pit_window_start=0.45):
        self.target_compound_id = target_compound_id
        # pit_window_start is the normalized lap progress (e.g. 0.45 = 45% race distance)
        self.pit_window_start = pit_window_start

    def predict(self, obs, state=None, deterministic=False):
        # According to F1StrategyEnv observation space:
        # obs[0] = lap_progress (normalized)
        # obs[6] = pit_count (normalized by 4.0)
        lap_progress = obs[0]
        pit_count_norm = obs[6] 
        
        # If we haven't pitted yet and we've reached the pit window
        if pit_count_norm == 0.0 and lap_progress >= self.pit_window_start:
            return self.target_compound_id, None
            
        return 0, None

class RuleBasedAgent:
    """
    Baseline: Pits if tire_age > threshold OR if a Safety Car occurs 
    and tires are old enough to justify an opportunistic pit stop.
    """
    def __init__(self, tire_age_threshold=0.5, sc_opportunistic_threshold=0.2):
        # Thresholds are relative to normalized tire age (max 50 laps)
        # 0.5 = 25 laps old. 0.2 = 10 laps old.
        self.tire_age_threshold = tire_age_threshold
        self.sc_opportunistic_threshold = sc_opportunistic_threshold

    def predict(self, obs, state=None, deterministic=False):
        # obs[1] = tire_age (normalized)
        # obs[8] = sc_flag
        # obs[9] = vsc_flag
        tire_age = obs[1]
        sc_flag = obs[8]
        vsc_flag = obs[9]
        
        if tire_age >= self.tire_age_threshold:
            # Tires are dead, must pit. (Default to Hard tire (3) for safety)
            return 3, None
            
        if (sc_flag == 1.0 or vsc_flag == 1.0) and tire_age >= self.sc_opportunistic_threshold:
            # Opportunistic pit stop under SC/VSC to save time
            return 3, None
            
        return 0, None
