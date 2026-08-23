import logging
from src.data_pipeline.telemetry_loader import TelemetryLoader
from src.visualization.track_maps import plot_speed_map

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    print("======================================")
    print("  GENERATING SPEED MAP (P1)")
    print("======================================\n")
    
    loader = TelemetryLoader()
    circuit = loader.config['data']['circuit']
    year = loader.config['data']['seasons'][-1]
    
    # 1. Load Session
    session = loader.load_session()
    
    # 2. Dynamically find the Winner
    try:
        results = session.results.iloc[:1]
        driver1 = results.iloc[0]['Abbreviation']
        logger.info(f"Dynamically selected P1: {driver1} for Speed Map.")
    except Exception as e:
        logger.error(f"Failed to extract race results: {e}. Cannot select driver automatically. Aborting.")
        return
        
    # 3. Fetch Telemetry
    lap, tel = loader.get_fastest_lap_telemetry(session, driver1)
    
    if tel is None:
        logger.error("Missing telemetry for the driver. Aborting.")
        return
        
    # 4. Fetch Circuit Info (Corners)
    circuit_info = None
    try:
        circuit_info = session.get_circuit_info()
    except Exception as e:
        logger.warning(f"Could not fetch circuit info: {e}")
        
    # 5. Plot
    out_path = plot_speed_map(circuit, year, driver1, tel, circuit_info)
    
    if out_path:
        print(f"\n[SUCCESS] Speed map saved to: {out_path}")
    else:
        print(f"\n[ERROR] Failed to generate speed map.")

if __name__ == "__main__":
    main()
