# Thermal-AI Lab: Data Center Cooling Simulation & Anomaly Detection

A physics-based thermal simulation combined with unsupervised machine learning
to investigate anomaly detection in a simulated data-center server rack.

The project was built as a small technical experiment connecting engineering
simulation, Python, time-series data, and machine learning.

## Overview

The thermal behavior of a server rack is represented by the first-order model:

```
C · dT/dt = P(t) − k · (T − T_ambient)
```

The simulation generates normal operating behavior and two controlled fault
scenarios:

1. Cooling degradation
2. Sudden compute-power spike

An Isolation Forest is trained exclusively on normal operating data and then
evaluated on the fault scenarios.

The purpose is to investigate how a general-purpose anomaly detector behaves
under different types of abnormal system behavior.

## Physics Model

The model represents the thermal balance of a simplified server rack.

Where:

- T = rack temperature (°C)
- T_ambient = cooling supply-air temperature
- P(t) = server heat load (W)
- C = thermal capacitance (J/K)
- k = cooling effectiveness (W/K)

The simulation uses Euler integration with a one-minute timestep.

The nominal parameters are:

- T_ambient = 22°C
- C = 50,000 J/K
- k = 200 W/K

The resulting thermal time constant is approximately:

```
τ = C / k = 250 s ≈ 4.2 min
```

## Machine Learning

The Isolation Forest uses three features:

- Temperature
- Server power
- Temperature rate of change (dT/dt)

The model is trained only on normal baseline data.

It is then evaluated against two previously unseen simulated fault scenarios.

### Fault 1: Cooling Degradation

The cooling coefficient decreases gradually from:

**200 W/K → 140 W/K**

over 24 hours.

This represents gradual loss of cooling effectiveness.

The experiment shows that this type of slow system degradation is difficult for
the generic Isolation Forest to detect reliably.

### Fault 2: Compute Spike

Server power is doubled for 15 minutes while cooling effectiveness remains
nominal.

This produces a strong short-term deviation from normal operating behavior.

The Isolation Forest detects the injected compute anomaly much more reliably
than the gradual cooling degradation.

## Key Result

The contrast between the two experiments is the main finding.

The detector performs well on an abrupt abnormal event but struggles with
gradual degradation.

This demonstrates an important limitation of generic anomaly detection:
detecting unusual observations is not necessarily the same as detecting slow
changes in system behavior.

A more advanced predictive-maintenance approach could combine anomaly
detection with trend analysis, physics-based residuals, or degradation-specific
models.

## Project Structure

```
thermal-ai-lab/
├── README.md
├── requirements.txt
├── src/
│   ├── simulation.py          # Stage 1: normal baseline (7 days, 1-min steps)
│   ├── faults.py              # Stage 2: cooling-degradation and compute-spike scenarios
│   ├── anomaly_detection.py   # Stage 3: Isolation Forest trained on normal data only
│   └── analysis.py            # plots and evaluation helpers
├── notebooks/
│   └── analysis.ipynb
└── results/                   # generated CSVs
```

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the simulation:

```bash
python src/simulation.py
```

Inject the fault scenarios:

```bash
python src/faults.py
```

Train and evaluate the anomaly detector:

```bash
python src/anomaly_detection.py
```

Launch the analysis notebook:

```bash
jupyter notebook notebooks/analysis.ipynb
```

## How it was built

Designed and specified by me (Ahmed Reza). Much of the code was written with AI coding assistants, then reviewed, run and tested by me.

## Limitations

This is a small experimental prototype rather than a production simulation.

The thermal model, power profile, fault scenarios, and parameters are
synthetically defined and are not calibrated against a specific physical
cooling installation.

The experiment uses only two fault scenarios and one anomaly-detection
algorithm.

Therefore, the results demonstrate an experimental methodology and engineering
reasoning rather than production-level predictive-maintenance performance.