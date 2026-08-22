import yaml
import random
from typing import Dict, Optional
from src.environment.race_state import RaceState, CarState
from src.models.lap_time_model import LapTimeModel
import logging

logger = logging.getLogger(__name__)

class RaceSimulator:
    """
    Models a complete multi-agent F1 race simulation by managing race states, 
    incidents (SC/VSC), and utilizing the LapTimeModel to advance cars mathematically.
    """
    def __init__(self, config_path="config.yaml", circuit="Silverstone"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.num_cars = self.config['simulation']['num_cars']
        self.total_laps = self.config['simulation']['total_laps']
        
        self.sc_prob = self.config['safety_car']['probability_per_lap']
        self.sc_min = self.config['safety_car']['min_duration']
        self.sc_max = self.config['safety_car']['max_duration']
        
        self.vsc_prob = self.config['vsc']['probability_per_lap']
        self.vsc_min = self.config['vsc']['min_duration']
        self.vsc_max = self.config['vsc']['max_duration']
        
        self.lap_model = LapTimeModel(circuit=circuit, config_path=config_path)
        
        self.state: Optional[RaceState] = None
        self._sc_timer = 0
        self._vsc_timer = 0

    def reset(self) -> RaceState:
        cars = {}
        for i in range(1, self.num_cars + 1):
            agent_id = f"Agent_{i}"
            # Assign random initial compounds for varied starting strategies
            compound = random.choice(["SOFT", "MEDIUM", "HARD"])
            
            cars[agent_id] = CarState(
                driver_name=agent_id,
                team_name=f"Team_{i}",
                position=i,
                lap=1,
                # Simulate starting grid gaps: P1 crosses start line first, P2 crosses 2s later
                total_race_time=i * 2.0, 
                current_compound=compound,
                tire_age=1,
                stint_number=1,
                pit_count=0,
                gap_ahead=2.0 if i > 1 else 0.0,
                gap_behind=2.0 if i < self.num_cars else 0.0
            )
            
        self.state = RaceState(
            lap_progress=1,
            total_laps=self.total_laps,
            safety_car=False,
            vsc=False,
            laps_under_sc=0,
            weather_condition=0.0,
            cars=cars
        )
        self._sc_timer = 0
        self._vsc_timer = 0
        return self.state

    def step(self, actions: Dict[str, int]) -> RaceState:
        """
        Executes one lap for all cars based on their chosen strategic action.
        Actions: 0 = Stay Out, 1 = Pit -> Soft, 2 = Pit -> Medium, 3 = Pit -> Hard
        """
        # 1. Resolve stochastic track incidents (SC/VSC)
        self._update_incidents()
        
        # 2. Execute Lap Physics 
        for agent_id, action in actions.items():
            self._process_car_lap(agent_id, action)
            
        # 3. Update Positions and Gaps
        self._update_positions()
        
        # 4. Advance Race Time
        self.state.lap_progress += 1
        return self.state

    def _update_incidents(self):
        """
        DESIGN DECISION
        Decision: Stochastic Safety Car / VSC generation using fixed lap probabilities.
        Reason: Random SC deployments force RL agents to learn opportunistic pitting.
                Durations are clamped to historical averages (min/max config).
        """
        if self._sc_timer > 0:
            self._sc_timer -= 1
            self.state.laps_under_sc += 1
            if self._sc_timer == 0:
                self.state.safety_car = False
        elif self._vsc_timer > 0:
            self._vsc_timer -= 1
            if self._vsc_timer == 0:
                self.state.vsc = False
        else:
            if self.config['safety_car']['enabled'] and random.random() < self.sc_prob:
                self.state.safety_car = True
                self._sc_timer = random.randint(self.sc_min, self.sc_max)
            elif self.config['vsc']['enabled'] and random.random() < self.vsc_prob:
                self.state.vsc = True
                self._vsc_timer = random.randint(self.vsc_min, self.vsc_max)

    def _process_car_lap(self, agent_id: str, action: int):
        car = self.state.cars[agent_id]
        
        is_pitting = False
        new_compound = None
        
        # Map Discrete Action Space to Pit Decisions
        if action == 1:
            is_pitting = True
            new_compound = "SOFT"
        elif action == 2:
            is_pitting = True
            new_compound = "MEDIUM"
        elif action == 3:
            is_pitting = True
            new_compound = "HARD"
            
        # Logical Constraint: Prevent pitting if tires are brand new (prevents infinite pitting loop exploit)
        if is_pitting and car.tire_age < 2:
            is_pitting = False
            new_compound = None

        # Traffic Effect: Dirty air if within 1.5 seconds of car ahead
        in_traffic = car.gap_ahead > 0.0 and car.gap_ahead < 1.5 
        
        # 1. Determine absolute lap time using LapTimeModel
        lap_time = self.lap_model.predict_lap_time(
            compound=car.current_compound,
            tire_age=car.tire_age,
            lap_number=self.state.lap_progress,
            total_laps=self.total_laps,
            in_traffic=in_traffic,
            safety_car=self.state.safety_car,
            vsc=self.state.vsc,
            pit_stop=is_pitting
        )
        
        # 2. Advance car mathematically
        car.total_race_time += lap_time
        car.lap += 1
        
        # 3. Apply strategic changes
        if is_pitting:
            car.current_compound = new_compound
            car.tire_age = 1
            car.stint_number += 1
            car.pit_count += 1
        else:
            car.tire_age += 1

    def _update_positions(self):
        """
        DESIGN DECISION
        Decision: Mathematical Undercut/Overcut through Total Race Time Sorting.
        Reason: We explicitly DO NOT give speed boosts for undercuts. Instead, 
                because we add the pit loss (~25s) to `total_race_time`, the pitting car 
                drops in the sorted array (loses track position). On the next lap, their fresh 
                tires make their `lap_time` significantly faster, meaning their `total_race_time` 
                grows slower than cars ahead on old tires. They naturally close the gap and execute an undercut.
        """
        # Sort cars dynamically by who has completed the race distance the fastest
        sorted_cars = sorted(self.state.cars.values(), key=lambda c: c.total_race_time)
        
        for idx, car in enumerate(sorted_cars):
            car.position = idx + 1
            
            if idx == 0:
                car.gap_ahead = 0.0
            else:
                car.gap_ahead = car.total_race_time - sorted_cars[idx - 1].total_race_time
                
            if idx == len(sorted_cars) - 1:
                car.gap_behind = 0.0
            else:
                car.gap_behind = sorted_cars[idx + 1].total_race_time - car.total_race_time
