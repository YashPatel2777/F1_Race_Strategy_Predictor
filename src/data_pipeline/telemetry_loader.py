import os
import yaml
import fastf1
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class TelemetryLoader:
    """
    Handles fetching and processing high-resolution GPS and car telemetry 
    for micro-level analysis (speed maps, driver comparisons).
    """
    def __init__(self, config_path="config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Unified Cache with the Strategy Simulator
        os.makedirs('cache/fastf1', exist_ok=True)
        fastf1.Cache.enable_cache('cache/fastf1')
        
    def load_session(self):
        """Loads the specific session requested in the config."""
        # Use the most recent season provided in the config list
        season = self.config['data']['seasons'][-1]
        circuit = self.config['data']['circuit']
        session_type = self.config['telemetry']['session_type']
        
        logger.info(f"Loading Telemetry Session: {season} {circuit} {session_type}")
        session = fastf1.get_session(season, circuit, session_type)
        session.load(telemetry=True, weather=False, messages=False)
        return session
        
    def get_fastest_lap_telemetry(self, session, driver: str):
        """Extracts the fastest lap and its telemetry with distance added."""
        laps = session.laps.pick_driver(driver)
        if len(laps) == 0:
            logger.warning(f"No laps found for driver {driver}.")
            return None, None
            
        fastest_lap = laps.pick_fastest()
        if pd.isna(fastest_lap['LapTime']):
            logger.warning(f"Fastest lap for {driver} has no valid LapTime.")
            return fastest_lap, None
            
        telemetry = fastest_lap.get_telemetry().add_distance()
        return fastest_lap, telemetry

    def get_interpolated_telemetry(self, telemetry1, telemetry2):
        """
        DESIGN DECISION: Distance-based Interpolation
        Interpolates two drivers' telemetry onto the exact same uniform distance axis
        so their deltas can be mathematically compared corner-by-corner.
        """
        step = self.config['telemetry']['distance_interpolation_step']
        
        # Determine the maximum distance covered by either driver (safeguard for different racing lines)
        max_dist = min(telemetry1['Distance'].max(), telemetry2['Distance'].max())
        uniform_dist = np.arange(0, max_dist, step)
        
        def interpolate_driver(tel):
            df = pd.DataFrame({'Distance': uniform_dist})
            for col in ['Speed', 'Throttle', 'Brake', 'nGear', 'Time']:
                if col in tel:
                    if col == 'Time':
                        # Convert timedelta to raw seconds for math operations
                        seconds = tel['Time'].dt.total_seconds()
                        df[col] = np.interp(uniform_dist, tel['Distance'], seconds)
                    else:
                        df[col] = np.interp(uniform_dist, tel['Distance'], tel[col])
            return df
            
        interp1 = interpolate_driver(telemetry1)
        interp2 = interpolate_driver(telemetry2)
        
        # Calculate Delta Time (Driver 1 - Driver 2)
        # Positive delta means Driver 1 is slower (took more time to reach this distance)
        delta_time = interp1['Time'] - interp2['Time']
        
        return uniform_dist, interp1, interp2, delta_time

    def get_sector_distances(self, lap, telemetry):
        """
        Extracts the exact track distance (in meters) where Sector 1 and Sector 2 end.
        Useful for shading graphs.
        """
        s1_time = lap['Sector1SessionTime']
        s2_time = lap['Sector2SessionTime']
        
        s1_dist = 0
        s2_dist = 0
        
        if not pd.isna(s1_time):
            # Find the telemetry row closest to the Sector 1 end time
            idx = (telemetry['SessionTime'] - s1_time).abs().idxmin()
            s1_dist = telemetry.loc[idx, 'Distance']
            
        if not pd.isna(s2_time):
            idx = (telemetry['SessionTime'] - s2_time).abs().idxmin()
            s2_dist = telemetry.loc[idx, 'Distance']
            
        return s1_dist, s2_dist
