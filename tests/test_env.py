import sys

def test_environment():
    try:
                # pyrefly: ignore [missing-import]
        import fastf1
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        # pyrefly: ignore [missing-import]
        import gymnasium as gym
        # pyrefly: ignore [missing-import]
        import stable_baselines3 as sb3
        import torch
        import tensorboard
        import sklearn
        import joblib
        import yaml
        # pyrefly: ignore [missing-import]
        import pytest
        
        print("All critical libraries successfully imported!")
        print(f"Python version: {sys.version.split(' ')[0]}")
        print(f"FastF1 version: {fastf1.__version__}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"Stable-Baselines3 version: {sb3.__version__}")
        print(f"Gymnasium version: {gym.__version__}")
        
    except ImportError as e:
        print(f"Environment setup failed. Missing library: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_environment()
