import matplotlib.pyplot as plt
import seaborn as sns
import yaml

def setup_plot_style(config_path="config.yaml"):
    """
    DESIGN DECISION: Consolidated Matplotlib Setup
    Ensures all telemetry plots use the exact same aesthetic theme
    (dark mode, specific font sizes) without duplicating code in every script.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    theme = config.get('telemetry', {}).get('plot_theme', 'dark_background')
    plt.style.use(theme)
    
    # Custom tweaks for F1 aesthetics
    plt.rcParams.update({
        'figure.facecolor': '#111111',
        'axes.facecolor': '#111111',
        'axes.edgecolor': '#444444',
        'text.color': 'white',
        'axes.labelcolor': 'white',
        'xtick.color': 'white',
        'ytick.color': 'white',
        'grid.color': '#333333',
        'grid.alpha': 0.5,
        'font.size': 10,
        'axes.titlesize': 14,
        'axes.titleweight': 'bold'
    })
    
    # Set seaborn color palette
    sns.set_palette("bright")
