# F1 Race Strategy Predictor

> **An AI-powered Formula 1 Race Strategy Prediction System using Reinforcement Learning and real F1 data.**

**Status: In Development**

## About

The **F1 Race Strategy Predictor** is a Reinforcement Learning project that aims to train AI agents to make strategic decisions during an F1 race.

The system will use **real historical Formula 1 data from FastF1** to model factors such as tire degradation, lap times, pit stops, and race conditions.

The AI will then learn through simulated races when to pit and which tire compound to choose.

## Main Goal

The agent will make decisions such as:

* Stay out or pit
* Choose Soft, Medium, or Hard tires
* React to Safety Car / VSC periods
* Attempt undercuts or overcuts
* Optimize overall race position

The goal is to learn strategies through **Reinforcement Learning rather than manually programming fixed strategies**.

## Technologies

* **Python**
* **FastF1** — F1 historical data & telemetry
* **Gymnasium** — race simulation environment
* **Stable-Baselines3 / PPO** — Reinforcement Learning
* **NumPy / pandas** — data processing
* **Scikit-learn** — statistical modeling
* **Matplotlib** — visualization
* **PyTorch** — ML backend

## Basic Architecture

```text
Real F1 Data (FastF1)
        ↓
Data Processing
        ↓
Tire Degradation & Lap-Time Models
        ↓
F1 Race Simulator
        ↓
RL Environment
        ↓
PPO Agents
        ↓
Race Strategy Decisions
        ↓
Evaluation & Visualization
```

## What I Want to Build

The final system should be able to simulate an F1 race with multiple AI-controlled cars and learn strategies based on:

* Tire age and degradation
* Track position
* Gaps to other cars
* Pit-stop timing
* Safety Car / VSC conditions
* Tire compound selection

The trained agents will then be evaluated against simple rule-based strategies and historical F1 strategies.

## Project Status

Currently building the project from scratch.

**Planned:**

* FastF1 data pipeline
* Tire degradation model
* Race simulator
* RL environment
* PPO training
* Strategy evaluation
* Race visualization

## Note

This is an **Educational / Research Simulation**, not an attempt to reproduce the complete physics or strategy systems used by real F1 teams.

The project focuses specifically on exploring **Reinforcement Learning for Race Strategy Optimization**.
