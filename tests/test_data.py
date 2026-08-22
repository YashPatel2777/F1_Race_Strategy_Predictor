import os
import pandas as pd

# pyrefly: ignore [missing-import]
import pytest
from src.data.fastf1_loader import FastF1Loader
from src.data.data_preprocessor import DataPreprocessor

@pytest.fixture(scope="module")
def processed_data():
    dataset_path = os.path.join('data', 'processed', 'lap_dataset.csv')
    if not os.path.exists(dataset_path):
        pytest.skip(f"Dataset not found at {dataset_path}. Run download_data.py first.")
    df = pd.read_csv(dataset_path)
    return df

def test_loader_caches_and_extracts():
    """Test that the FastF1 loader works and uses cached data."""
    loader = FastF1Loader(cache_dir='cache/fastf1')
    session = loader.load_session(2024, "Silverstone")
    laps = loader.get_laps(session)
    assert not laps.empty, "Failed to extract laps from session"
    compounds = loader.get_compounds(session)
    assert len(compounds) > 0, "Failed to extract compounds"

def test_dataset_not_empty(processed_data):
    """Ensure the processed dataset has rows."""
    assert not processed_data.empty, "Dataset is empty"

def test_required_columns_exist(processed_data):
    """Ensure all expected features were engineered."""
    required_cols = [
        'season', 'event', 'circuit', 'driver', 'team', 'lap_number', 'lap_time', 
        'compound', 'tyre_life', 'stint', 'track_status', 'position', 'pit_in', 'pit_out', 
        'is_safety_car', 'is_vsc', 'air_temp', 'track_temp', 'humidity', 'rainfall'
    ]
    for col in required_cols:
        assert col in processed_data.columns, f"Missing required column: {col}"

def test_lap_time_validity(processed_data):
    """Lap times must fall within realistic F1 boundaries."""
    valid_laps = processed_data.dropna(subset=['lap_time'])
    # FastF1 can sometimes have very slow laps for pit in/out or SC periods
    # But extremely fast laps (<50s) on standard tracks are erroneous
    assert (valid_laps['lap_time'] > 50).all(), "Found unrealistically fast lap times (< 50s)"

def test_tyre_life_validity(processed_data):
    """Tire life cannot be less than 1."""
    valid_tyres = processed_data.dropna(subset=['tyre_life'])
    assert (valid_tyres['tyre_life'] >= 1).all(), "Tyre life cannot be less than 1"
    
def test_compounds_identified(processed_data):
    """Ensure compounds are recognized Pirelli types."""
    compounds = processed_data['compound'].dropna().unique()
    valid_compounds = {'SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE', 'WET', 'UNKNOWN', 'TEST-UNKNOWN'}
    for c in compounds:
        assert c in valid_compounds, f"Unknown compound found: {c}"

def test_stint_validity(processed_data):
    """Stint numbers must be logical."""
    valid_stints = processed_data.dropna(subset=['stint'])
    assert (valid_stints['stint'] >= 1).all(), "Stint number cannot be less than 1"

def test_incident_booleans(processed_data):
    """Safety Car and VSC flags must be strictly boolean."""
    assert processed_data['is_safety_car'].isin([True, False]).all(), "is_safety_car must be boolean"
    assert processed_data['is_vsc'].isin([True, False]).all(), "is_vsc must be boolean"

def test_data_leakage_checks(processed_data):
    """
    Ensure rows do not contain future information.
    For lap-level data, the row should only represent current state or past state.
    We check that there are no 'next_lap_time' columns generated accidentally.
    """
    leakage_keywords = ['next_', 'future', 'final_position']
    for col in processed_data.columns:
        for keyword in leakage_keywords:
            assert keyword not in col.lower(), f"Potential data leakage found in column: {col}"
