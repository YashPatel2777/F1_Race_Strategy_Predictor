# pyrefly: ignore [missing-import]
import pytest
from src.models.lap_time_model import LapTimeModel

@pytest.fixture
def lap_model():
    # Initializes with Silverstone models assuming they were created in Step 5
    return LapTimeModel(circuit="Silverstone")

def test_base_pace_differences(lap_model):
    """Ensure compound offsets and fuel drops are applied logically."""
    # Lap 1, fresh Softs (Heavy car)
    soft_lap = lap_model.predict_lap_time(compound='SOFT', tire_age=1, lap_number=1, total_laps=52)
    # Lap 1, fresh Hards (Heavy car)
    hard_lap = lap_model.predict_lap_time(compound='HARD', tire_age=1, lap_number=1, total_laps=52)
    
    # We must subtract the inherent degradation loss at lap 1 before comparing base pace offsets
    soft_deg_1 = lap_model.deg_models['SOFT'].predict([[1]])[0]
    hard_deg_1 = lap_model.deg_models['HARD'].predict([[1]])[0]
    
    soft_base = soft_lap - max(0, soft_deg_1)
    hard_base = hard_lap - max(0, hard_deg_1)
    
    assert hard_base > soft_base, "Hard tires should be slower than Softs"
    assert abs((hard_base - soft_base) - lap_model.compound_offsets['HARD']) < 1e-5, "Offset should strictly match configured delta"

def test_fuel_burn_effect(lap_model):
    """Car should get faster as fuel burns (excluding tire degradation)."""
    lap_1 = lap_model.predict_lap_time(compound='HARD', tire_age=1, lap_number=1, total_laps=52)
    lap_50 = lap_model.predict_lap_time(compound='HARD', tire_age=1, lap_number=50, total_laps=52)
    
    assert lap_50 < lap_1, "Car should be faster on lap 50 than lap 1 due to low fuel weight"

def test_tire_degradation_effect(lap_model):
    """Older tires should result in slower lap times."""
    fresh_tire = lap_model.predict_lap_time(compound='SOFT', tire_age=1, lap_number=20, total_laps=52)
    old_tire = lap_model.predict_lap_time(compound='SOFT', tire_age=20, lap_number=20, total_laps=52)
    
    assert old_tire > fresh_tire, "20-lap old tires should be slower than 1-lap old tires"

def test_pit_stop_loss(lap_model):
    """Pitting must strictly increase absolute lap time."""
    normal_lap = lap_model.predict_lap_time(compound='MEDIUM', tire_age=10, lap_number=20, total_laps=52)
    pit_lap = lap_model.predict_lap_time(compound='MEDIUM', tire_age=10, lap_number=20, total_laps=52, pit_stop=True)
    
    expected_loss = lap_model.pit_loss_stationary + lap_model.pit_loss_transit
    assert (pit_lap - normal_lap) == expected_loss, "Pit lap time did not increase by correct transit+stationary amount"

def test_safety_car_slowdown(lap_model):
    """Safety Car should massively slow down lap times."""
    normal_lap = lap_model.predict_lap_time(compound='HARD', tire_age=5, lap_number=20, total_laps=52)
    sc_lap = lap_model.predict_lap_time(compound='HARD', tire_age=5, lap_number=20, total_laps=52, safety_car=True)
    
    assert sc_lap > normal_lap + 30.0, "Safety Car must slow down pace significantly"
