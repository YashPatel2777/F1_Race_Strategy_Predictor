# pyrefly: ignore [missing-import]
import pytest
import os
import numpy as np
import joblib
from src.models.tire_degradation import TireDegradationModel

def test_degradation_model_slopes():
    """
    Ensures that the historically fitted ML degradation models mathematically
    enforce that older tires lose more pace than fresh tires.
    """
    circuit = "Silverstone"
    compounds = ["SOFT", "MEDIUM", "HARD"]
    
    for compound in compounds:
        path = os.path.join("data", "models", f"deg_{circuit}_{compound}.joblib")
        if not os.path.exists(path):
            pytest.skip(f"Trained model {path} not found. Run fit_degradation.py first.")
            
        model = joblib.load(path)
        
        # Predict pace loss (in seconds)
        fresh_tire_loss = model.predict(np.array([[1]]))[0]
        old_tire_loss = model.predict(np.array([[20]]))[0]
        
        # Since 'loss' is added to lap time, a higher loss means a slower lap
        assert old_tire_loss > fresh_tire_loss, f"{compound} model failed: 20-lap tire is mathematically faster than 1-lap tire!"

def test_fuel_correction_logic():
    """
    Ensure the data normalization logic correctly applies fuel adjustments.
    """
    import pandas as pd
    modeler = TireDegradationModel()
    
    # Mock some basic lap data
    data = pd.DataFrame({
        'season': [2024, 2024],
        'circuit': ['Silverstone', 'Silverstone'],
        'driver': ['VER', 'VER'],
        'stint': [1, 1],
        'lap_time': [90.0, 89.0], # Raw lap times getting faster
        'lap_number': [1, 20],
        'pit_in': [False, False],
        'pit_out': [False, False],
        'is_safety_car': [False, False],
        'is_vsc': [False, False]
    })
    
    normalized_df = modeler._normalize_lap_times(data)
    
    # Lap 20 should have 20 units of fuel penalty added to its baseline to isolate tire deg
    assert 'fuel_corrected_time' in normalized_df.columns
    
    lap_20_corrected = normalized_df[normalized_df['lap_number'] == 20]['fuel_corrected_time'].values[0]
    expected = 89.0 + (20 * modeler.fuel_effect)
    
    assert lap_20_corrected == expected, "Fuel correction mathematics are incorrect"
