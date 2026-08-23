import os
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
from src.visualization.base_plotter import setup_plot_style

def plot_speed_map(circuit, year, driver, telemetry, circuit_info=None):
    """
    Plots the physical circuit layout colored by the driver's speed.
    """
    setup_plot_style()
    
    x = telemetry['X'].values
    y = telemetry['Y'].values
    speed = telemetry['Speed'].values
    
    # Create segments for LineCollection
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.suptitle(f"{year} {circuit} GP - {driver} Speed Map", fontsize=16, y=0.95)
    
    # We use a colormap ranging from red (slow) to green (fast)
    cmap = plt.get_cmap('RdYlGn')
    norm = plt.Normalize(speed.min(), speed.max())
    
    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=4)
    lc.set_array(speed[:-1]) # Color based on speed
    
    line = ax.add_collection(lc)
    ax.axis('off')
    
    # Auto scale because LineCollection doesn't automatically set axes limits
    ax.set_xlim(x.min() - 500, x.max() + 500)
    ax.set_ylim(y.min() - 500, y.max() + 500)
    
    # Add colorbar
    cbar = plt.colorbar(line, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Speed (km/h)', color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    cbar.ax.tick_params(colors='white')
    
    # Add Corners if available
    if circuit_info is not None and hasattr(circuit_info, 'corners'):
        for _, corner in circuit_info.corners.iterrows():
            txt = f"{corner['Number']}"
            ax.text(corner['X'], corner['Y'], txt, color='#ffffff', alpha=0.7, 
                    fontsize=12, ha='center', va='center', fontweight='bold', 
                    bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', boxstyle='circle'))
                    
    os.makedirs(os.path.join('outputs', 'plots', 'telemetry'), exist_ok=True)
    out_path = os.path.join('outputs', 'plots', 'telemetry', f'{circuit}_{year}_{driver}_SpeedMap.png')
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, facecolor='#111111')
    plt.close()
    
    return out_path
