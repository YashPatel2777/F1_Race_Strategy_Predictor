import yaml
import os
import logging
from src.data.data_preprocessor import DataPreprocessor

# Configure logging to show process output
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

def main():
    # Load project configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    data_config = config['data']
    print(f"Downloading and processing data for circuit: {data_config['circuit']}, seasons: {data_config['seasons']}")
    
    # Initialize the preprocessor and load laps
    preprocessor = DataPreprocessor(cache_dir='cache/fastf1')
    df = preprocessor.process_multiple_sessions(data_config)
    
    if not df.empty:
        # Save processed dataset
        output_path = os.path.join('data', 'processed', 'lap_dataset.csv')
        df.to_csv(output_path, index=False)
        print(f"\nDataset successfully saved to {output_path}")
        print(f"Total laps extracted: {len(df)}")
        print("\nSample Data:")
        print(df.head())
    else:
        print("Failed to process any datasets.")

if __name__ == "__main__":
    main()
