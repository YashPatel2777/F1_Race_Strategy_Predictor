import logging
from src.data_pipeline.telemetry_loader import TelemetryLoader
from src.visualization.telemetry_plots import plot_driver_comparison

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    print("======================================")
    print("  GENERATING TELEMETRY COMPARISON")
    print("======================================\n")
    
    loader = TelemetryLoader()
    circuit = loader.config['data']['circuit']
    year = loader.config['data']['seasons'][-1]
    
    # 1. Load Session
    session = loader.load_session()
    
    # 2. Dynamically find the Winner and Runner-Up
    # session.results is a DataFrame sorted by finishing position
    try:
        results = session.results.iloc[:2]
        driver1 = results.iloc[0]['Abbreviation']
        driver2 = results.iloc[1]['Abbreviation']
        logger.info(f"Dynamically selected P1: {driver1}, P2: {driver2}")
    except Exception as e:
        logger.error(f"Failed to extract race results: {e}. Cannot select drivers automatically. Aborting.")
        return
        
    # 3. Fetch Laps
    lap1, tel1 = loader.get_fastest_lap_telemetry(session, driver1)
    lap2, tel2 = loader.get_fastest_lap_telemetry(session, driver2)
    
    if tel1 is None or tel2 is None:
        logger.error("Missing telemetry for one or both drivers. Aborting.")
        return
        
    # 4. Interpolate and calculate Delta
    logger.info("Interpolating telemetry arrays onto a uniform distance axis...")
    dist, i_tel1, i_tel2, delta = loader.get_interpolated_telemetry(tel1, tel2)
    s1_dist, s2_dist = loader.get_sector_distances(lap1, tel1)
    
    # 5. Plot
    logger.info("Rendering multi-panel comparison plot...")
    out_path = plot_driver_comparison(circuit, year, driver1, driver2, dist, i_tel1, i_tel2, delta, s1_dist, s2_dist)
    
    print(f"\n[SUCCESS] Telemetry comparison saved to: {out_path}")

if __name__ == "__main__":
    main()
