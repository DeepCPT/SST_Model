# Sequential Search Transformer (SST)

This repository contains the code implementation for the Monte Carlo simulations conducted to evaluate parameter recovery capabilities of the **Sequential Search Transformer (SST)**, as described in our recent study.

## Overview

Understanding and leveraging consumer behavior presents significant business opportunities. Traditional deep learning methods, despite their predictive strengths, lack interpretability and explicit modeling of consumer decision-making processes. Economic theories suggest that consumers typically follow a sequential search strategy, evaluating alternatives sequentially to find the best match for their preferences.

To address this challenge, we developed the **Sequential Search Transformer (SST)**, which integrates deep learning approaches with economic sequential search theory to accurately model consumer search and purchase decisions.

## Key Contributions

- SST combines the predictive power of deep learning with economic theory, resulting in improved interpretability and decision modeling.
- SST explicitly models consumer search and decision-making across sessions and sequentially resolves uncertainty in product utility.
- Demonstrated through Monte Carlo simulations, SST effectively recovers parameters from simulated datasets.
- Empirical evaluations indicate SST's superior performance compared to state-of-the-art deep learning and structural econometric models.

## Monte Carlo Simulations

The simulations provided in this repository serve the following purposes:

- **Parameter Recovery:** Verify SST's capability to recover known parameters from simulated datasets.
- **Model Validation:** Assess the robustness and reliability of SST under controlled conditions.


## Repository Structure

```bash
├── VAR=0.1/                    # Simulated data (simulated_sessions.7z, need to unzip the file), trained model parameters (DeepStructural_model_final.pt), and logs with variance = 0.1
├── VAR=0.5/                    # Simulated data (simulated_sessions.7z, need to unzip the file), trained model parameters, and logs with variance = 0.5
├── VAR=1.0/                    # Simulated data (simulated_sessions.7z, need to unzip the file), trained parameters, and logs with variance = 1.0
├── ru_model.py                 # Reservation Utility model
├── ru_model_parameters.pt      # Pre-trained Reservation Utility model parameters
├── simulated_sessions.pkl      # Generated simulated session data
├── Simulated_Date_PrePost_Shock.py   # Script to generate simulated session data
├── deep_structural_embedding_prepost_shock.py  # SST training script for real-world or simulated data
└── README.md
```



## Usage


### Using Pre-trained Model and Simulated Data

To evaluate SST using pre-simulated data and trained parameters:

1. Copy the files `simulated_sessions.pkl` and `DeepStructural_model_final.pt` from their respective subdirectories.
2. Run:
   ```bash
   python deep_structural_embedding_prepost_shock.py
   ```

This will load the simulated data and trained parameters to test the parameter recovery performance.

### Full Simulation and Training Process

To run the complete simulation and training process:

1. Execute the simulation script to generate data:
   ```bash
   python Simulated_Date_PrePost_Shock.py
   ```
   *(Adjust the preference shock variance parameter within the script at line 588 if needed.)*

2. After generating the simulated data, train and test SST:
   ```bash
   python deep_structural_embedding_prepost_shock.py
   ```
