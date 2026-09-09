# pyrefly: ignore [missing-import]
import fastf1
import yaml

def print_top_10():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    circuit = config['data']['circuit']
    latest_season = max(config['data']['seasons'])
    
    print(f"\nFetching top 10 drivers for {circuit} ({latest_season})...")
    
    fastf1.Cache.enable_cache('data')
    
    try:
        # Fetch Qualifying
        q_session = fastf1.get_session(latest_season, circuit, 'Q')
        q_session.load(telemetry=False, weather=False, messages=False, livedata=None)
        
        q_top = q_session.results.head(10)
        print(f"Top 10 Qualifiers - {circuit} {latest_season}")
        print(f"{'Pos':<5} | {'Driver':<25} | {'Team':<25}")
        print("-" * 63)
        for i, (_, row) in enumerate(q_top.iterrows(), 1):
            print(f"{i:<5} | {row['FullName']:<25} | {row['TeamName']:<25}")
            
        print("\n")
        print("-" * 67)
        print("\n")

        # Fetch Race
        r_session = fastf1.get_session(latest_season, circuit, 'R')
        r_session.load(telemetry=False, weather=False, messages=False, livedata=None)
        
        r_top = r_session.results.head(10)
        print(f"Top 10 Race Finishers - {circuit} {latest_season}")
        print(f"{'Pos':<5} | {'Driver':<25} | {'Team':<25}")
        print("-" * 63)
        for i, (_, row) in enumerate(r_top.iterrows(), 1):
            print(f"{i:<5} | {row['FullName']:<25} | {row['TeamName']:<25}")
    except Exception as e:
        print(f"Could not fetch top 10 drivers: {e}")

if __name__ == "__main__":
    print_top_10()
