import os
import logging
from src.models.tire_degradation import TireDegradationModel

logging.basicConfig(level=logging.INFO, format='%(message)s')

def main():
    print("======================================")
    print("  FITTING TIRE DEGRADATION MODELS")
    print("======================================\n")
    
    data_path = os.path.join('data', 'processed', 'lap_dataset.csv')
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Please run download_data.py first.")
        return
        
    modeler = TireDegradationModel()
    print("Normalizing lap times, filtering anomalies, and fitting regressors...")
    metrics = modeler.fit_models(data_path)
    
    print("\nBest Model Fitted Per Compound:")
    print("-" * 50)
    print(metrics.to_string(index=False))
    print("-" * 50)
    print("\n[SUCCESS] Degradation curves plotted and saved to: outputs/plots/degradation/")
    print("[SUCCESS] Trained model artifacts (.joblib) saved to: data/models/")

if __name__ == "__main__":
    main()
