import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import yaml
import logging

logger = logging.getLogger(__name__)

class TireDegradationModel:
    """Calculates and models real F1 tire degradation from historical data."""
    
    def __init__(self, config_path="config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # SIMULATION ASSUMPTION: Linear fuel effect per lap
        self.fuel_effect = self.config['simulation']['fuel_effect'] 
        self.models = {}  
        
    def _normalize_lap_times(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        DESIGN DECISION
        Decision: Normalize lap times by removing fuel weight offsets and stint base pace.
        Reason: Raw lap times decrease (improve) over a stint because the car burns fuel and gets lighter, 
                masking the fact that the tires are degrading. By neutralizing fuel weight, we isolate 
                pure tire drop-off.
        Alternative: Use advanced Bayesian models or rely on uncorrected times.
        When to change it: If we obtain real telemetry on exact fuel loads per car.
        """
        # Filter strictly for clean racing laps
        valid = df[
            (df['pit_in'] == False) & 
            (df['pit_out'] == False) & 
            (df['is_safety_car'] == False) & 
            (df['is_vsc'] == False) & 
            (df['lap_time'].notna())
        ].copy()
        
        # Remove outlier laps (mistakes, heavy traffic) 
        # Using 105% of the median race lap time as a cutoff for "normal" racing speed
        valid['median_race_time'] = valid.groupby(['season', 'circuit'])['lap_time'].transform('median')
        valid = valid[valid['lap_time'] < valid['median_race_time'] * 1.05]
        
        # 1. Correct for fuel weight
        # As laps increase, we ADD the fuel effect to pretend the car stayed heavy.
        # This makes lap times naturally increase as tires degrade.
        valid['fuel_corrected_time'] = valid['lap_time'] + (valid['lap_number'] * self.fuel_effect)
        
        # 2. Baseline Pace 
        # We subtract the fastest lap of that specific driver's stint to get a delta
        valid['stint_base_pace'] = valid.groupby(['season', 'circuit', 'driver', 'stint'])['fuel_corrected_time'].transform('min')
        valid['degradation_delta'] = valid['fuel_corrected_time'] - valid['stint_base_pace']
        
        return valid

    def fit_models(self, data_path: str) -> pd.DataFrame:
        df = pd.read_csv(data_path)
        normalized_df = self._normalize_lap_times(df)
        
        # Group by circuit and compound for distinct models
        groups = normalized_df.groupby(['circuit', 'compound'])
        results = []
        
        for (circuit, compound), group in groups:
            # We ignore wet/intermediate compounds for now, and require enough data points
            if compound in ['UNKNOWN', 'TEST-UNKNOWN', 'WET', 'INTERMEDIATE'] or len(group) < 30:
                continue
                
            X = group[['tyre_life']].values
            y = group['degradation_delta'].values
            
            # DESIGN DECISION: Model Selection
            # Decision: Restrict to Linear or simple Poly(2) to prevent severe non-monotonic curves
            # Reason: 3rd-degree polynomials can cause the degradation curve to dip downwards, 
            # making 20-lap old tires theoretically "faster" than 1-lap old tires in the simulator.
            models_to_test = {
                'Linear': LinearRegression(),
                'Polynomial_d2': make_pipeline(PolynomialFeatures(2), LinearRegression()),
            }
            
            best_model_name = None
            best_model = None
            best_r2 = -float('inf')
            best_metrics = {}
            
            for name, model in models_to_test.items():
                model.fit(X, y)
                preds = model.predict(X)
                
                mae = mean_absolute_error(y, preds)
                rmse = np.sqrt(mean_squared_error(y, preds))
                r2 = r2_score(y, preds)
                
                if r2 > best_r2:
                    best_r2 = r2
                    best_model = model
                    best_model_name = name
                    best_metrics = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
                    
            self.models[(circuit, compound)] = best_model
            
            results.append({
                'circuit': circuit,
                'compound': compound,
                'best_model': best_model_name,
                **best_metrics
            })
            
            # Plot the fitted degradation curve vs actuals
            self._plot_degradation(circuit, compound, X, y, best_model, best_model_name)
            
            # Save the trained model artifact
            model_path = os.path.join('data', 'models', f'deg_{circuit}_{compound}.joblib')
            joblib.dump(best_model, model_path)
            
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join('outputs', 'reports', 'degradation_metrics.csv'), index=False)
        return results_df

    def _plot_degradation(self, circuit, compound, X, y, model, model_name):
        plt.figure(figsize=(10, 6))
        
        # Add jitter to X for visualization density
        x_jittered = X.flatten() + np.random.normal(0, 0.2, size=len(X))
        
        plt.scatter(x_jittered, y, alpha=0.1, color='gray', label='Historical Laps (Normalized)')
        
        # Predict along a smooth curve
        X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        y_pred = model.predict(X_range)
        plt.plot(X_range, y_pred, color='red', linewidth=3, label=f'{model_name} Fit')
        
        plt.title(f'Tire Degradation Profile: {circuit} - {compound}')
        plt.xlabel('Tire Age (Laps)')
        plt.ylabel('Pace Loss (Seconds)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        out_path = os.path.join('outputs', 'plots', 'degradation', f'{circuit}_{compound}.png')
        plt.savefig(out_path)
        plt.close()
