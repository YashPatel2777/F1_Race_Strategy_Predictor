# **F1 Race Strategy Predictor**

Have you ever yelled at your TV when a Formula 1 team completely botches a pit stop strategy? I definitely have. 

I built this project to see if an Artificial Intelligence could do a better job on the pit wall. Instead of just writing a basic calculator that guesses lap times, I turned an F1 race into a **Multi-Agent Reinforcement Learning (MARL)** playground. We let a neural network drive thousands of simulated races to learn the dark arts of F1 strategy from scratch!

## What makes this cool?
The AI doesn't just look at a spreadsheet. It actually has to "race" against other bots in a custom-built physics simulator and balance:
- **Tire Degradation:** Figuring out when the Soft tires fall off a cliff compared to the Hards.
- **The Pit Lane Penalty:** Deciding if fresh rubber is worth losing 25 seconds in the pits.
- **The Undercut:** Pitting early to jump the guy in front of you.
- **Safety Car Chaos:** Diving into the pits opportunistically when a Safety Car (SC) or Virtual Safety Car (VSC) is deployed to get a "cheap" stop.

## How it works under the hood
1. **The Real Data**: We use the awesome `FastF1` library to pull actual historical telemetry and lap times from real F1 sessions.
2. **The Physics (Machine Learning)**: We train `scikit-learn` regression models to figure out exactly how much time tires lose as they get older, stripping out the effect of the car getting lighter as it burns fuel.
3. **The Track (Simulation)**: A custom-built race engine (`RaceSimulator`) handles the actual racing logic, traffic, and random safety cars.
4. **The Brain (Reinforcement Learning)**: We use `Stable-Baselines3` (specifically Proximal Policy Optimization) to train our AI. We give it points for overtaking and heavy penalties for getting passed or pitting unnecessarily. 

## Finding your way around
```text
F1_Race_Strategy_Predictor/
├── config.yaml                     # The control panel (tweak track, fuel penalty, etc.)
├── data/                           # Where the FastF1 data and trained AI brains live
├── outputs/plots/races/            # Where the cool gap-to-leader charts are saved
├── scripts/                        # The main files you'll want to run! (See below)
├── src/                            # The messy engine room (Data pipelines, models, simulator)
└── tests/                          # 28 tests to make sure the physics don't break
```

## How to play with it
You'll need Python 3.10 or newer.

**1. Set things up:**
```bash
python -m venv venv
.\venv\Scripts\activate      # Windows
source venv/bin/activate     # Mac/Linux
pip install -r requirements.txt
```

**2. Tell your computer where the code is:**
```bash
$env:PYTHONPATH="."          # Windows
export PYTHONPATH="."        # Mac/Linux
```

**3. Run the pipeline (in this order!):**
Want to train your own AI race strategist? Run these scripts one by one:
```bash
# 1. Download real F1 data and learn how tires degrade
python scripts/fetch_data.py
python scripts/fit_degradation.py

# 2. Put the AI in the simulator and let it learn by trial and error!
python scripts/train_agent.py

# 3. Test the trained AI against standard F1 strategies and generate charts
python scripts/evaluate_agent.py
python scripts/visualize_race.py

# 4. Grab some popcorn and watch a live text-commentary of the AI racing!
python scripts/demo.py
```

## The "0-Pit" Loophole (A funny quirk of AI)
If you run the evaluation, you might notice something hilarious. Real F1 rules state that a driver *must* use two different tire compounds during a race (forcing at least one pit stop). 

Because I didn't initially explicitly hardcode a massive penalty for breaking this specific rule, the AI did the math and realized a very funny truth: *Even on completely destroyed tires, it is mathematically faster to just stay out for 52 laps than to waste 25 seconds driving down the pit lane.*

It completely reward-hacked the simulation! This is exactly how Reinforcement Learning works—it ruthlessly optimizes whatever rules you give it. To fix this, you just need to add a massive negative score in the environment if the race ends and the car only used one compound!
