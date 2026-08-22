# pyrefly: ignore [missing-import]
import gymnasium as gym
# pyrefly: ignore [missing-import]
from gymnasium import spaces
import numpy as np
from src.environment.race_simulator import RaceSimulator
import logging

logger = logging.getLogger(__name__)

class F1StrategyEnv(gym.Env):
    """
    Gymnasium environment wrapping the F1 RaceSimulator.
    
    DESIGN DECISION: SB3 Multi-Agent Limitation
    Decision: Single "Ego" Agent Architecture.
    Reason: Standard Stable-Baselines3 PPO only supports single-agent environments 
            (mapping one action to one observation). We cannot naturally output N 
            actions and N observations simultaneously using standard `gym.Env`.
    Implementation: The environment controls ONE specific car (the 'Ego Agent'). 
            The other cars are currently controlled by baseline rule-based bots.
            To allow the policy to learn from different starting positions, the Ego Agent 
            is randomly assigned to a different grid slot on every `reset()`.
    Future Migration: To convert this to a true multi-agent shared-policy setup, this class 
            should be rewritten using PettingZoo's `AECEnv` or `ParallelEnv` API, where 
            `step()` accepts a dictionary of actions and returns dictionaries of observations.
    """
    metadata = {"render_modes": ["console"]}
    
    def __init__(self, config_path="config.yaml", circuit="Silverstone"):
        super().__init__()
        self.simulator = RaceSimulator(config_path, circuit)
        
        # Strategic Action Space:
        # 0: Stay Out
        # 1: Pit -> SOFT
        # 2: Pit -> MEDIUM
        # 3: Pit -> HARD
        self.action_space = spaces.Discrete(4)
        
        # Observation Space:
        # 10 Continuous/Normalized features
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(10,), dtype=np.float32)
        
        self.ego_agent_id = "Agent_1"
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.simulator.reset()
        
        # Randomize which car the RL agent controls to prevent grid-slot bias
        idx = self.np_random.integers(1, self.simulator.num_cars + 1)
        self.ego_agent_id = f"Agent_{idx}"
        
        return self._get_observation(), {}
        
    def step(self, action: int):
        # 1. Generate bot actions for all other cars
        actions = self._generate_bot_actions()
        # 2. Inject RL agent action
        actions[self.ego_agent_id] = action
        
        prev_pos = self.simulator.state.cars[self.ego_agent_id].position
        
        # 3. Advance the physics simulator 
        # (This implicitly handles _calculate_lap_time, _execute_pit_stop, _update_positions)
        self.simulator.step(actions)
        
        # 4. Extract ML signals
        obs = self._get_observation()
        reward = self._calculate_reward(prev_pos, action)
        
        terminated = self.simulator.state.lap_progress > self.simulator.total_laps
        truncated = False
        
        return obs, reward, terminated, truncated, {}
        
    def _generate_bot_actions(self):
        """Baseline rule-based bots for the non-ego cars."""
        actions = {}
        for agent_id, car in self.simulator.state.cars.items():
            if agent_id == self.ego_agent_id:
                continue
                
            # Rule-based logic: Pit if tires > 20 laps old.
            # (We will expand on baseline strategies in Step 9)
            if car.tire_age > 20:
                actions[agent_id] = self.np_random.integers(1, 4)
            else:
                actions[agent_id] = 0
        return actions

    def _get_observation(self):
        """
        DESIGN DECISION
        Decision: Normalize all continuous features between 0.0 and 1.0.
        Reason: PPO uses neural networks which are highly sensitive to scale. 
                If gap_ahead is [0, 40] and compound is [0, 2], the gradients will 
                destabilize. Normalization ensures the policy converges smoothly.
        """
        car = self.simulator.state.cars[self.ego_agent_id]
        state = self.simulator.state
        
        lap_norm = state.lap_progress / state.total_laps
        
        # Tire age normalized (assuming 50 laps is absolute maximum life)
        tire_age_norm = min(1.0, car.tire_age / 50.0) 
        
        # Compound ID mapping
        comp_map = {"SOFT": 0.0, "MEDIUM": 0.5, "HARD": 1.0}
        compound_val = comp_map.get(car.current_compound, 0.5)
        
        # Gaps (clipped at 20 seconds, any further doesn't strictly matter for immediate combat)
        gap_ahead_norm = min(1.0, car.gap_ahead / 20.0) 
        gap_behind_norm = min(1.0, car.gap_behind / 20.0)
        
        # Position (P1 = 0.0, P5 = 1.0)
        pos_norm = (car.position - 1) / (self.simulator.num_cars - 1)
        
        pit_norm = min(1.0, car.pit_count / 4.0)
        stint_norm = min(1.0, car.stint_number / 5.0)
        
        sc_val = 1.0 if state.safety_car else 0.0
        vsc_val = 1.0 if state.vsc else 0.0
        
        obs = np.array([
            lap_norm, tire_age_norm, compound_val, gap_ahead_norm, gap_behind_norm,
            pos_norm, pit_norm, stint_norm, sc_val, vsc_val
        ], dtype=np.float32)
        
        return obs
        
    def _calculate_reward(self, prev_pos, action):
        """
        DESIGN DECISION
        Decision: Reward Shaping.
        Reason: Waiting 50 laps for a single sparse reward (finish position) makes 
                RL training painfully slow. We add intermediate shaping.
        Reward Hacking Risk: If the agent gets +1 for overtaking, it might learn to intentionally 
                let cars pass it just so it can overtake them again to farm points. 
                To prevent this, losing a position penalizes heavily (-2.0), making farming unprofitable.
        """
        car = self.simulator.state.cars[self.ego_agent_id]
        reward = 0.0
        
        # 1. Intermediate Progress
        if car.position < prev_pos:
            reward += 2.0  # Passed someone
        elif car.position > prev_pos:
            reward -= 2.0  # Got passed
            
        # 2. Strategic Penalties
        if car.tire_age > 25:
            reward -= 0.1 # Slight penalty for holding dead tires
            
        if action != 0 and car.tire_age < 5:
            reward -= 5.0 # Massive penalty for pitching 2 laps after a pit stop (Illegal/Unnecessary)
            
        # 3. Final Race Result
        if self.simulator.state.lap_progress > self.simulator.total_laps:
            # Reward formula: P1 gets +20, P5 gets -20
            # (assuming 5 cars: 5 - 1 = 4.  (4/4)*40 - 20 = +20)
            score = ((self.simulator.num_cars - car.position) / (self.simulator.num_cars - 1)) * 40.0 - 20.0
            reward += score
            
        return float(reward)
        
    def render(self):
        car = self.simulator.state.cars[self.ego_agent_id]
        print(f"Lap {self.simulator.state.lap_progress} | Pos: {car.position} | Tire: {car.current_compound} ({car.tire_age}L)")
