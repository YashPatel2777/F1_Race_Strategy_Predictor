import os
# pyrefly: ignore [missing-import]
import fastf1
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class FastF1Loader:
    """Handles loading and extracting specific historical F1 data components from FastF1."""
    
    def __init__(self, cache_dir: str = "cache/fastf1"):
        os.makedirs(cache_dir, exist_ok=True)
        # Enable caching to avoid re-downloading telemetry and session data
        fastf1.Cache.enable_cache(cache_dir)
        
    def load_session(self, season: int, event: str) -> fastf1.core.Session:
        logger.info(f"Loading FastF1 session: {season} {event}")
        session = fastf1.get_session(season, event, 'R')
        # We only need weather and messages for strategy, telemetry is unnecessary overhead
        session.load(telemetry=False, weather=True, messages=True)
        return session
        
    def get_laps(self, session: fastf1.core.Session) -> pd.DataFrame:
        """Extracts complete lap-by-lap information."""
        return session.laps
        
    def get_drivers(self, session: fastf1.core.Session) -> dict:
        """Extracts dictionary of driver numbers to driver abbreviations."""
        drivers = {}
        for drv in session.drivers:
            drivers[drv] = session.get_driver(drv)['Abbreviation']
        return drivers
        
    def get_compounds(self, session: fastf1.core.Session) -> pd.Series:
        """Extracts tire compound information across the session."""
        return session.laps['Compound'].dropna().unique()
        
    def get_stints(self, session: fastf1.core.Session) -> pd.DataFrame:
        """Extracts stint information from laps."""
        return session.laps[['Driver', 'Stint', 'Compound', 'TyreLife']].drop_duplicates()
        
    def get_lap_times(self, session: fastf1.core.Session) -> pd.DataFrame:
        """Extracts just lap times and validity."""
        return session.laps[['Driver', 'LapNumber', 'LapTime', 'IsAccurate']]
        
    def get_weather(self, session: fastf1.core.Session) -> pd.DataFrame:
        """Extracts timestamped weather conditions."""
        return session.weather_data
        
    def get_race_control_messages(self, session: fastf1.core.Session) -> pd.DataFrame:
        """Extracts official race control messages."""
        return session.race_control_messages
        
    def get_sc_periods(self, session: fastf1.core.Session) -> pd.DataFrame:
        """Extracts Safety Car periods using TrackStatus."""
        laps = session.laps
        # Track status '4' signifies Safety Car
        sc_laps = laps[laps['TrackStatus'].astype(str).str.contains('4')]
        return sc_laps[['LapNumber', 'TrackStatus', 'Time']].drop_duplicates(subset=['LapNumber'])
        
    def get_vsc_periods(self, session: fastf1.core.Session) -> pd.DataFrame:
        """Extracts VSC periods using TrackStatus."""
        laps = session.laps
        # Track status '6' signifies VSC, '7' signifies VSC ending
        vsc_laps = laps[laps['TrackStatus'].astype(str).str.contains('6|7')]
        return vsc_laps[['LapNumber', 'TrackStatus', 'Time']].drop_duplicates(subset=['LapNumber'])
