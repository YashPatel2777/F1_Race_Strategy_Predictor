import pandas as pd
import numpy as np
from src.data.fastf1_loader import FastF1Loader
import logging

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Preprocesses raw FastF1 sessions into structured lap-level ML datasets."""
    
    def __init__(self, cache_dir: str = "cache/fastf1"):
        self.loader = FastF1Loader(cache_dir=cache_dir)
        
    def process_session(self, season: int, event: str) -> pd.DataFrame:
        session = self.loader.load_session(season, event)
        laps = self.loader.get_laps(session)
        weather = self.loader.get_weather(session)
        
        df = laps.copy()
        
        # Add event info
        df['season'] = season
        df['event'] = session.event['EventName']
        df['circuit'] = session.event['Location']
        
        # Rename FastF1 columns to our standardized format
        rename_map = {
            'Driver': 'driver',
            'Team': 'team',
            'LapNumber': 'lap_number',
            'LapTime': 'lap_time',
            'Compound': 'compound',
            'TyreLife': 'tyre_life',
            'Stint': 'stint',
            'TrackStatus': 'track_status',
            'Position': 'position'
        }
        df = df.rename(columns=rename_map)
        
        # Derived Feature: Lap time in seconds (FastF1 provides timedelta)
        df['lap_time'] = df['lap_time'].dt.total_seconds()
        
        # Derived Feature: Pit stops (boolean) based on pit in/out timestamps
        df['pit_in'] = ~df['PitInTime'].isna()
        df['pit_out'] = ~df['PitOutTime'].isna()
        
        # Derived Feature: Track incidents (SC/VSC) using TrackStatus strings
        # FastF1 TrackStatus codes: '4' = SC, '6' = VSC, '7' = VSC ending
        df['is_safety_car'] = df['track_status'].astype(str).apply(lambda x: '4' in x)
        df['is_vsc'] = df['track_status'].astype(str).apply(lambda x: '6' in x or '7' in x)
        
        # Derived Feature: Merge timestamped weather data
        if not weather.empty and not df.empty:
            df = df.sort_values('Time')
            weather = weather.sort_values('Time')
            # merge_asof matches the closest previous weather reading to the lap's timestamp
            df = pd.merge_asof(
                df.dropna(subset=['Time']), 
                weather[['Time', 'AirTemp', 'TrackTemp', 'Humidity', 'Rainfall']], 
                on='Time', direction='backward'
            )
            df = df.rename(columns={
                'AirTemp': 'air_temp',
                'TrackTemp': 'track_temp',
                'Humidity': 'humidity',
                'Rainfall': 'rainfall'
            })
        else:
            df['air_temp'] = np.nan
            df['track_temp'] = np.nan
            df['humidity'] = np.nan
            df['rainfall'] = False
            
        # Select required columns
        cols = [
            'season', 'event', 'circuit', 'driver', 'team', 'lap_number', 'lap_time', 
            'compound', 'tyre_life', 'stint', 'track_status', 'position', 'pit_in', 'pit_out', 
            'is_safety_car', 'is_vsc', 'air_temp', 'track_temp', 'humidity', 'rainfall'
        ]
        final_cols = [c for c in cols if c in df.columns]
        
        return df[final_cols].copy()

    def process_multiple_sessions(self, config_data: dict) -> pd.DataFrame:
        circuit = config_data['circuit']
        seasons = config_data['seasons']
        
        all_laps = []
        for season in seasons:
            try:
                logger.info(f"Processing {season} {circuit}...")
                df = self.process_session(season, circuit)
                all_laps.append(df)
            except Exception as e:
                logger.error(f"Failed to process {season} {circuit}: {e}")
                
        if all_laps:
            return pd.concat(all_laps, ignore_index=True)
        return pd.DataFrame()
