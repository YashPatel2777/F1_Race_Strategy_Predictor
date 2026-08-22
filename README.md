# F1 Race Strategy Predictor 🏎️

A complete Reinforcement Learning pipeline that learns and predicts Formula 1 race strategies from scratch using historical `FastF1` telemetry data, Proximal Policy Optimization (PPO), and a custom multi-agent physics engine.

## 🌟 Overview
Unlike typical lap-time predictors, this project models an F1 race as a **Multi-Agent Reinforcement Learning (MARL) environment**. The agent must learn the delicate balance between:
- Tire Degradation (Soft vs Medium vs Hard)
- Pit Stop Time Penalties (~25 seconds)
- Undercutting & Overcutting opponents
- Opportunistic pitting during Safety Cars (SC) and Virtual Safety Cars (VSC)
- Traffic and dirty air penalties

## 🏗️ Architecture Stack
1. **Data Ingestion**: `FastF1` API (extracts real lap times and compound choices).
2. **Predictive Modeling**: `scikit-learn` (Polynomial Regression to isolate tire degradation pace-loss from fuel-burn effect).
3. **Simulation**: A purely decoupled, tick-based race engine (`RaceSimulator`) handling physics, undercuts, and SC deployments.
4. **Environment**: A custom `Gymnasium` environment wrapping the simulator.
5. **Reinforcement Learning**: `Stable-Baselines3` (PPO) using an Ego-Agent architecture.

## 📂 Project Structure
```text
F1_Race_Strategy_Predictor/
├── config.yaml                     # Global hyperparameters and track settings
├── data/
│   ├── cache/                      # FastF1 offline telemetry cache
│   └── models/                     # Saved Scikit-Learn (.joblib) & PPO (.zip) weights
├── outputs/
│   └── plots/races/                # Generated Matplotlib strategy charts
├── scripts/
│   ├── fetch_data.py               # Downloads FastF1 telemetry
│   ├── fit_degradation.py          # Trains ML tire degradation models
│   ├── train_agent.py              # Executes PPO RL Training Loop
│   ├── evaluate_agent.py           # Evaluates PPO vs Baselines (100 races)
│   ├── visualize_race.py           # Generates Gap-to-Leader graphics
│   └── demo.py                     # Interactive Lap-by-Lap terminal commentary
├── src/
│   ├── data_pipeline/              # Data downloading and cleaning
│   ├── models/                     # ML Degradation, LapTimeModel, Baselines, PPO Wrapper
│   └── environment/                # RaceSimulator, RaceState, Gym Env
└── tests/                          # 28 Pytest suites covering physics & ML integrity
```

## 🚀 Setup & Installation
Requires Python 3.10+
```bash
# 1. Clone repository & create virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate
# Activate (Mac/Linux)
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Setup Python Path
# Windows
$env:PYTHONPATH="."
# Mac/Linux
export PYTHONPATH="."
```

## 🏁 Running the Pipeline
You must execute the pipeline sequentially to train the AI from scratch.
You can configure the target circuit and RL hyperparameters in `config.yaml`.

### 1. Data Mining & ML Fitting
Extract historical data and fit the regression curves for tire degradation:
```bash
python scripts/fetch_data.py
python scripts/fit_degradation.py
```

### 2. Reinforcement Learning
Train the PPO neural network on the simulator (simulates thousands of races):
```bash
python scripts/train_agent.py
```

### 3. Evaluation & Visualization
Prove the AI's dominance by pitting it against textbook F1 baseline bots:
```bash
python scripts/evaluate_agent.py
python scripts/visualize_race.py
```

### 4. Interactive Demo
Watch a live text-commentary of the RL agent racing against the bots!
```bash
python scripts/demo.py
```

## 🧪 Testing
The project is strictly verified via Test-Driven Development (TDD). To run the full suite:
```bash
pytest tests/ -v
```

## ⚠️ Known Behaviors (Reward Hacking)
If you evaluate the AI without explicitly writing a rule that strictly mandates pitting (the "two compounds" rule), the PPO agent will quickly mathematically realize that staying out on completely dead tires for 52 laps is technically faster than wasting 25 seconds in the pit lane. 
This is a feature of Reinforcement Learning, not a bug! The agent optimally solved the environment precisely as it was coded. To fix this, you can easily add a heavy negative reward penalty in `src/environment/f1_strategy_env.py` for agents that fail to pit.
