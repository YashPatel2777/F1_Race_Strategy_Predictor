import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from src.visualization.base_plotter import setup_plot_style
import logging

logger = logging.getLogger(__name__)

def animate_fastest_lap(circuit, year, driver, telemetry, circuit_info=None):
    """
    Generates an MP4 animation of a driver's fastest lap.
    Shows a blip tracing the circuit with a live speed readout.
    """
    logger.info("Initializing animation engine...")
    setup_plot_style()
    
    # Extract telemetry arrays
    x = telemetry['X'].values
    y = telemetry['Y'].values
    speed = telemetry['Speed'].values
    
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.suptitle(f"{year} {circuit} GP - {driver} Fastest Lap", fontsize=16, y=0.95)
    
    # Plot the faint background track layout
    ax.plot(x, y, color='gray', linewidth=2, alpha=0.3)
    
    # Add Corners if available
    if circuit_info is not None and hasattr(circuit_info, 'corners'):
        for _, corner in circuit_info.corners.iterrows():
            txt = f"{corner['Number']}"
            ax.text(corner['X'], corner['Y'], txt, color='#ffffff', alpha=0.5, 
                    fontsize=10, ha='center', va='center', fontweight='bold')
    
    ax.axis('off')
    
    # The moving objects: The car (dot) and the live dashboard (text)
    car_dot, = ax.plot([], [], 'o', color='#00ffff', markersize=12, markeredgecolor='white')
    dashboard = ax.text(0.05, 0.95, '', transform=ax.transAxes, color='white', 
                        fontsize=18, fontweight='bold', fontfamily='monospace')
    
    def init():
        car_dot.set_data([], [])
        dashboard.set_text('')
        return car_dot, dashboard

    def update(frame):
        # Handle single values safely by putting them in a sequence
        car_dot.set_data([x[frame]], [y[frame]])
        # Add live data to dashboard
        dashboard.set_text(f"DRIVER: {driver}\nSPEED : {speed[frame]:.0f} km/h")
        return car_dot, dashboard
        
    # Telemetry is highly dense (usually ~15,000 points per lap).
    # To keep the MP4 generation fast and the video smooth, we sample every 4th frame.
    step = 4
    frames = np.arange(0, len(x), step)
    
    logger.info(f"Rendering {len(frames)} frames. This may take a moment...")
    ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True)
    
    os.makedirs(os.path.join('outputs', 'animations'), exist_ok=True)
    out_path = os.path.join('outputs', 'animations', f'{circuit}_{year}_{driver}_lap.mp4')
    
    try:
        # Save using FFmpeg at 30 FPS
        ani.save(out_path, writer='ffmpeg', fps=30, dpi=150)
        logger.info(f"Animation saved to {out_path}")
    except Exception as e:
        logger.error(f"FFmpeg failed to save animation. Ensure ffmpeg is installed on your system. Error: {e}")
        out_path = None
        
    plt.close()
    return out_path
