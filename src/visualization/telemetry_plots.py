import os
import matplotlib.pyplot as plt
import numpy as np
from src.visualization.base_plotter import setup_plot_style

def plot_driver_comparison(circuit, year, driver1, driver2, dist, tel1, tel2, delta_time, s1_dist, s2_dist):
    """
    Plots a multi-panel telemetry comparison between two drivers.
    Includes Speed, Throttle, Brake, Gear, Delta Time, and Sector Shading.
    """
    setup_plot_style()
    
    fig, axes = plt.subplots(5, 1, figsize=(16, 12), sharex=True, gridspec_kw={'height_ratios': [3, 2, 1, 1, 2]})
    fig.suptitle(f"{year} {circuit} GP - Fastest Lap Comparison\n{driver1} vs {driver2}", y=0.95)
    
    max_dist = dist[-1] if len(dist) > 0 else 0
    
    # Shade Sectors on all axes
    for ax in axes:
        # Sector 1
        if s1_dist > 0:
            ax.axvspan(0, s1_dist, color='white', alpha=0.03)
        # Sector 2
        if s2_dist > s1_dist:
            ax.axvspan(s1_dist, s2_dist, color='white', alpha=0.08)
        # Sector 3
        if max_dist > s2_dist:
            ax.axvspan(s2_dist, max_dist, color='white', alpha=0.03)
    
    # Colors
    c1, c2 = '#00ffff', '#ff00ff'
    
    # 1. Speed
    axes[0].plot(dist, tel1['Speed'], label=driver1, color=c1, linewidth=2)
    axes[0].plot(dist, tel2['Speed'], label=driver2, color=c2, linewidth=2, alpha=0.8)
    axes[0].set_ylabel('Speed (km/h)')
    axes[0].legend(loc='lower right')
    
    # 2. Throttle
    axes[1].plot(dist, tel1['Throttle'], color=c1)
    axes[1].plot(dist, tel2['Throttle'], color=c2, alpha=0.8)
    axes[1].set_ylabel('Throttle %')
    
    # 3. Brake
    axes[2].plot(dist, tel1['Brake'], color=c1)
    axes[2].plot(dist, tel2['Brake'], color=c2, alpha=0.8)
    axes[2].set_ylabel('Brake')
    axes[2].set_yticks([0, 1])
    axes[2].set_yticklabels(['Off', 'On'])
    
    # 4. Gear
    axes[3].plot(dist, tel1['nGear'], color=c1)
    axes[3].plot(dist, tel2['nGear'], color=c2, alpha=0.8)
    axes[3].set_ylabel('Gear')
    
    # 5. Delta Time (Driver 1 - Driver 2)
    # Positive means D1 is slower, Negative means D1 is faster
    axes[4].plot(dist, delta_time, color='white', linewidth=2)
    axes[4].axhline(0, color='gray', linestyle='--')
    axes[4].set_ylabel(f'Delta (s)\n(+) {driver2} Faster\n(-) {driver1} Faster')
    axes[4].set_xlabel('Distance (m)')
    
    # Fill areas for visual clarity on who is gaining time
    axes[4].fill_between(dist, delta_time, 0, where=(delta_time > 0), color=c2, alpha=0.3, label=f"{driver2} gaining")
    axes[4].fill_between(dist, delta_time, 0, where=(delta_time < 0), color=c1, alpha=0.3, label=f"{driver1} gaining")
    axes[4].legend(loc='upper right')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    os.makedirs(os.path.join('outputs', 'plots', 'telemetry'), exist_ok=True)
    out_path = os.path.join('outputs', 'plots', 'telemetry', f'{circuit}_{year}_{driver1}_vs_{driver2}.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    
    return out_path
