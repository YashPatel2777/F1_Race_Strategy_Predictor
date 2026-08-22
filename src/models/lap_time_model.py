import os
import yaml
import numpy as np
import joblib
import logging

logger = logging.getLogger(__name__)

class LapTimeModel:
    """
    Estimates lap times by combining real historical degradation with simulation assumptions 
    for physics constraints (fuel, traffic, SC).
    """
    def __init__(self, circuit="Silverstone", config_path="config.yaml"):
        self.circuit = circuit
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # --- SIMULATION ASSUMPTIONS ---
        # F1 telemetry for exact fuel weight and dirty air is not public. 
        # We must use accepted motorsport approximations.
        self.fuel_effect = self.config['simulation']['fuel_effect']
        self.traffic_effect = self.config['simulation']['traffic_effect']
        
        # --- REAL DATA / ESTIMATES ---
        # We can extract average pit lane transit times from historical data, but for now
        # we configure them based on known averages for this circuit.
        self.pit_loss_stationary = self.config['simulation'].get('pit_loss_stationary', 2.5)
        self.pit_loss_transit = self.config['simulation'].get('pit_loss_transit', 20.0)
        
        # --- SIMULATION ASSUMPTIONS ---
        # Base pace of the track (e.g. 90.0s for a 1:30.000 lap on Softs, 0 fuel).
        self.base_pace = 90.0
        
        # Tire compound inherent performance offsets relative to the Soft tire.
        self.compound_offsets = {
            'SOFT': 0.0,
            'MEDIUM': 0.6,
            'HARD': 1.2
        }
        
        # Safety Car pace delta (cars drive roughly 30-40% slower)
        self.sc_delta = 45.0 
        self.vsc_delta = 30.0
        
        # --- REAL MODEL ESTIMATES ---
        # Load the regression models trained on FastF1 historical laps in Phase 1
        self.deg_models = {}
        for compound in ['SOFT', 'MEDIUM', 'HARD']:
            path = os.path.join('data', 'models', f'deg_{self.circuit}_{compound}.joblib')
            if os.path.exists(path):
                self.deg_models[compound] = joblib.load(path)
            else:
                logger.warning(f"Degradation model for {compound} not found at {path}. Assuming 0 degradation.")
                self.deg_models[compound] = None

    def predict_lap_time(self, 
                         compound: str, 
                         tire_age: int, 
                         lap_number: int, 
                         total_laps: int,
                         in_traffic: bool = False,
                         safety_car: bool = False,
                         vsc: bool = False,
                         pit_stop: bool = False) -> float:
        """
        DESIGN DECISION
        Decision: Additive Lap Time Model
        Reason: Lap time is modeled as Base Pace + Fuel Penalty + Tire Deg + Compound Offset + Traffic + Incidents.
                This modular approach allows us to cleanly separate what is historically learned (Degradation) 
                from what must be assumed (Fuel burn).
        When to change it: If we obtain a comprehensive neural network capable of predicting raw lap times directly.
        """
        # 1. Base Pace
        time = self.base_pace
        
        # 2. Compound Offset
        time += self.compound_offsets.get(compound, 1.0)
        
        # 3. Fuel Effect
        # Car is heaviest on lap 1, lightest on total_laps. 
        fuel_penalty = (total_laps - lap_number) * self.fuel_effect
        time += max(0.0, fuel_penalty)
        
        # 4. Tire Degradation (REAL DATA)
        if compound in self.deg_models and self.deg_models[compound] is not None:
            # Predict degradation time loss
            loss = self.deg_models[compound].predict(np.array([[tire_age]]))[0]
            time += max(0.0, loss)  # Tires shouldn't magically get faster
            
        # 5. Traffic (Dirty Air)
        if in_traffic and not safety_car and not vsc:
            time += self.traffic_effect
            
        # 6. Race Control Incidents
        if safety_car:
            time += self.sc_delta
        elif vsc:
            time += self.vsc_delta
            
        # 7. Pit Stop (Stationary + Transit)
        # Note: While pitting under SC saves *relative* time against opponents, 
        # the *absolute* lap time still increases due to pit lane transit.
        if pit_stop:
            time += self.pit_loss_stationary + self.pit_loss_transit
            
        return float(time)
