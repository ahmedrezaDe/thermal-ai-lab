"""
Thermal System Simulation Module

This module provides components for simulating thermal systems,
including heat transfer, temperature evolution, and system dynamics.
"""

import os
import numpy as np
import pandas as pd

# --- Physical Constants ---
T_AMBIENT = 22.0     # °C, cooling supply air
C_THERMAL = 50000.0  # J/K, thermal capacitance of the rack
K_NOMINAL = 200.0    # W/K, baseline heat transfer coefficient
DT = 60.0            # seconds, simulation timestep
DAYS = 7             # simulate a full week for the baseline

def generate_power_load(timesteps, dt_seconds, seed=42):
    """
    Generates a daily cyclical compute load simulating server traffic.
    Oscillates between ~1000 W (night) and ~2200 W (day), with noise.
    Uses a fixed RNG seed for scientific reproducibility.
    """
    time_s = np.arange(timesteps) * dt_seconds
    
    # 86400 seconds in a day. Mean = 1600W, Amplitude = 600W
    # -np.pi/2 shifts the sine wave so it starts at the minimum load
    p_cycle = 1600.0 + 600.0 * np.sin(2 * np.pi * time_s / 86400 - np.pi / 2)
    
    # Explicit RNG for reproducible noise generation
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 50.0, timesteps)
    
    p_total = np.clip(p_cycle + noise, 500.0, 3000.0) # Hard boundaries
    return time_s, p_total

def simulate_temperature(time_s, power_w, k_array=None):
    """
    Simulates internal rack temperature using the Euler method.
    C * dT/dt = P(t) - k * (T - T_ambient)
    """
    steps = len(time_s)
    T = np.zeros(steps)
    T[0] = 27.0  # Initialize near the steady-state baseline
    
    # Allow passing a variable k array for fault injection
    if k_array is None:
        k_array = np.full(steps, K_NOMINAL)
        
    for i in range(steps - 1):
        # The physics equation solved for dT/dt
        dTdt = (power_w[i] - k_array[i] * (T[i] - T_AMBIENT)) / C_THERMAL
        T[i+1] = T[i] + dTdt * DT
        
    return T

def main():
    print("Initializing Stage 1: Normal Baseline Simulation...")
    timesteps = int((DAYS * 24 * 3600) / DT)
    
    time_s, power_w = generate_power_load(timesteps, DT, seed=42)
    temp_c = simulate_temperature(time_s, power_w)
    
    # Calculate dT/dt (rate of change) as a feature for the ML model
    # np.gradient gives us the central difference derivative
    dTdt = np.gradient(temp_c, DT)
    
    df = pd.DataFrame({
        'time_seconds': time_s,
        'power_w': power_w,
        'temperature_c': temp_c,
        'cooling_coefficient': K_NOMINAL,
        'dTdt': dTdt,
        'label': 0  # 0 = Normal, 1 = Anomaly
    })
    
    # Save the output
    os.makedirs('results', exist_ok=True)
    out_path = 'results/normal_data.csv'
    df.to_csv(out_path, index=False)
    
    print(f"Generated {DAYS} days of reproducible normal baseline data.")
    print(f"Saved to: {out_path}")
    print(f"Temperature Range: {df['temperature_c'].min():.1f}°C to {df['temperature_c'].max():.1f}°C")

if __name__ == "__main__":
    main()