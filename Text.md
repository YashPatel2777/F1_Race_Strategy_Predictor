# **F1 Race Strategy Predictor**

## **A Reinforcement Learning-Based Multi-Agent Formula 1 Race Strategy Simulation System**

A clean, standalone project that uses real Formula 1 historical data from FastF1 and Reinforcement Learning to learn race strategy decisions.

## Project Structure and Purpose

- `data/`: Stores the project datasets.
  - `raw/`: Raw data extracted from FastF1.
  - `processed/`: Processed lap-level datasets, ready for ML training.
  - `models/`: Fitted tire degradation models and saved RL models.
- `cache/fastf1/`: Dedicated directory for caching FastF1 API responses to avoid rate limits and improve speed.

- `outputs/`: All generated artifacts are saved here to avoid cluttering the repository.
  - `plots/`: Visualizations for tire degradation (`degradation/`) and RL training progress (`training/`).
  - `evaluations/`: Results and metrics from evaluating trained agents.
  - `simulations/`: Race replays, GIFs, and strategy plots from the simulator.
  - `reports/`: Text summaries or CSV logs of race results.

- `logs/tensorboard/`: TensorBoard logs for monitoring Stable-Baselines3 PPO training.

- `src/`: The core source code for the project.
  - `data/`: Scripts for loading FastF1 data, extracting lap/weather features, and building datasets.
  - `models/`: Mathematical models for tire degradation and lap time estimation.
  - `environment/`: The core race simulator and the Gymnasium environment for RL.
  - `agents/`: PPO Agent definitions and custom RL callbacks.
  - `training/`: Scripts to launch and configure PPO training.
  - `evaluation/`: Scripts to run evaluation races and evaluate baselines.
  - `visualization/`: Matplotlib plotting utilities for race replays and strategy visualization.

- `scripts/`: High-level entry points for downloading data, fitting models, training, evaluating, and simulating races.

- `tests/`: Unit tests (pytest) to validate data pipelines, degradation models, simulator logic, and environment constraints.

## Real Data vs Simulation Assumptions

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **Lap Times** | REAL DATA | Base pace derived directly from FastF1 historical laps. |
| **Tire Compounds** | REAL DATA | Historical compounds used by drivers. |
| **Tire Degradation** | MODEL ESTIMATE | Extracted and modeled from historical tire life vs lap time delta. |
| **Pit Stop Stationary Time** | REAL DATA | Based on real historical pit stop durations. |
| **Pit Lane Loss** | MODEL ESTIMATE | Modeled transit time based on average pit lane speed limits. |
| **Safety Car Frequency** | SIMULATION ASSUMPTION | Configurable probabilities for SC/VSC per lap. |
| **SC/VSC Duration** | SIMULATION ASSUMPTION | Configurable min/max duration ranges. |
| **Traffic Effect** | SIMULATION ASSUMPTION | Modeled time penalty when following closely, as aerodynamic dirty air is hard to perfectly extract. |
| **Fuel Effect** | SIMULATION ASSUMPTION | Assumed linear time penalty per lap of fuel burned (e.g., ~0.05s per kg). |
