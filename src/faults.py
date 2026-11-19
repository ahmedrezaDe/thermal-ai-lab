import os
import sys
import numpy as np
import pandas as pd

# Ensure we can import from the src directory when running from the root
sys.path.append(os.path.dirname(__file__))
from simulation import generate_power_load, simulate_temperature, K_NOMINAL, DT

def generate_cooling_fault(days=3):
    """
    Fault 1: Cooling Degradation.
    The heat transfer coefficient (k) drops from 200 to 140 W/K over 24 hours,
    simulating a failing fan or fouled heat exchanger.
    """
    timesteps = int((days * 24 * 3600) / DT)
    
    # Use the EXACT same seed as the baseline for a controlled experiment
    time_s, power_w = generate_power_load(timesteps, DT, seed=42)
    
    # Initialize k array
    k_array = np.full(timesteps, K_NOMINAL)
    
    # Day 1 is normal. Degradation starts at hour 24 and ends at hour 48.
    start_idx = int(24 * 3600 / DT)
    end_idx = int(48 * 3600 / DT)
    
    # Linear drop in cooling capacity
    k_array[start_idx:end_idx] = np.linspace(K_NOMINAL, 140.0, end_idx - start_idx)
    k_array[end_idx:] = 140.0 # Stays degraded for Day 3
    
    temp_c = simulate_temperature(time_s, power_w, k_array=k_array)
    dTdt = np.gradient(temp_c, DT)
    
    # Label: 0 for normal, 1 for anomaly (ground truth)
    labels = np.zeros(timesteps, dtype=int)
    labels[start_idx:] = 1 
    
    return pd.DataFrame({
        'time_seconds': time_s,
        'power_w': power_w,
        'temperature_c': temp_c,
        'cooling_coefficient': k_array, # Fixed naming
        'dTdt': dTdt,
        'label': labels
    })

def generate_compute_fault(days=3):
    """
    Fault 2: Compute Spike.
    A sudden doubling of server load (P) lasting 15 minutes,
    simulating a runaway process or traffic spike.
    """
    timesteps = int((days * 24 * 3600) / DT)
    
    # Use the EXACT same seed as the baseline for a controlled experiment
    time_s, power_w = generate_power_load(timesteps, DT, seed=42)
    k_array = np.full(timesteps, K_NOMINAL)
    
    # Spike at hour 36 (mid-day 2) for 15 minutes
    spike_start = int(36 * 3600 / DT)
    spike_end = spike_start + 15 
    
    # Multiply the load by 2.0 (100% increase)
    power_w[spike_start:spike_end] *= 2.0
    
    temp_c = simulate_temperature(time_s, power_w, k_array=k_array)
    dTdt = np.gradient(temp_c, DT)
    
    labels = np.zeros(timesteps, dtype=int)
    labels[spike_start:spike_end] = 1
    
    return pd.DataFrame({
        'time_seconds': time_s,
        'power_w': power_w,
        'temperature_c': temp_c,
        'cooling_coefficient': k_array, # Fixed naming
        'dTdt': dTdt,
        'label': labels
    })

def main():
    print("Initializing Stage 2: Fault Injection...")
    os.makedirs('results', exist_ok=True)
    
    # 1. Generate and save cooling fault
    df_cooling = generate_cooling_fault()
    out_cooling = 'results/fault_cooling.csv'
    df_cooling.to_csv(out_cooling, index=False)
    print(f"Generated Cooling Degradation data. Max Temp: {df_cooling['temperature_c'].max():.1f}°C")
    
    # 2. Generate and save compute spike fault
    df_compute = generate_compute_fault()
    out_compute = 'results/fault_compute.csv'
    df_compute.to_csv(out_compute, index=False)
    print(f"Generated Compute Spike data. Max Temp: {df_compute['temperature_c'].max():.1f}°C")

if __name__ == "__main__":
    main()