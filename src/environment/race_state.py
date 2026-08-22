from dataclasses import dataclass
from typing import Dict

@dataclass
class CarState:
    driver_name: str
    team_name: str
    position: int
    lap: int
    total_race_time: float
    current_compound: str
    tire_age: int
    stint_number: int
    pit_count: int
    gap_ahead: float
    gap_behind: float

@dataclass
class RaceState:
    lap_progress: int
    total_laps: int
    safety_car: bool
    vsc: bool
    laps_under_sc: int
    weather_condition: float  # 0.0 (Dry) to 1.0 (Wet) for future expansion
    cars: Dict[str, CarState] # agent_id -> CarState
