# pyrefly: ignore [missing-import]
import pytest
from src.environment.race_simulator import RaceSimulator

@pytest.fixture
def sim():
    simulator = RaceSimulator(circuit="Silverstone")
    simulator.reset()
    return simulator

def test_simulator_initialization(sim):
    assert sim.state.lap_progress == 1
    assert len(sim.state.cars) == sim.num_cars
    # Ensure grid stagger works (P1 ahead of P2)
    p1 = [c for c in sim.state.cars.values() if c.position == 1][0]
    p2 = [c for c in sim.state.cars.values() if c.position == 2][0]
    assert p1.total_race_time < p2.total_race_time
    assert p2.gap_ahead > 0

def test_lap_advancement(sim):
    # Action 0 = Stay Out
    actions = {f"Agent_{i}": 0 for i in range(1, sim.num_cars + 1)}
    sim.step(actions)
    
    assert sim.state.lap_progress == 2
    for car in sim.state.cars.values():
        assert car.tire_age == 2
        assert car.lap == 2
        assert car.pit_count == 0

def test_pit_stop_logic_and_undercut_sorting(sim):
    actions_lap1 = {f"Agent_{i}": 0 for i in range(1, sim.num_cars + 1)}
    sim.step(actions_lap1) # Advance to lap 2
    
    # Agent 1 (P1) pits for Mediums, Agent 2 stays out
    actions_lap2 = {f"Agent_{i}": 0 for i in range(1, sim.num_cars + 1)}
    actions_lap2["Agent_1"] = 2 # Pit -> Medium
    
    sim.step(actions_lap2)
    
    a1 = sim.state.cars["Agent_1"]
    a2 = sim.state.cars["Agent_2"]
    
    # Agent 1 should have fresh tires
    assert a1.tire_age == 1
    assert a1.current_compound == "MEDIUM"
    assert a1.stint_number == 2
    assert a1.pit_count == 1
    
    # Agent 1 should have lost track position to Agent 2 due to pit loss
    assert a1.position > a2.position
    assert a1.total_race_time > a2.total_race_time
    
def test_safety_car_mechanics(sim):
    sim.state.safety_car = True
    sim._sc_timer = 2
    
    actions = {f"Agent_{i}": 0 for i in range(1, sim.num_cars + 1)}
    sim.step(actions)
    
    assert sim.state.safety_car is True
    assert sim._sc_timer == 1
    
    sim.step(actions)
    
    # SC timer expires
    assert sim.state.safety_car is False
    assert sim._sc_timer == 0
