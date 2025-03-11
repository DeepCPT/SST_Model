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
├── VAR=0.1/                    # Simuled Data, Learned Model Paramater, and Parameter Recovery results when VAR=0.1
├── VAR=0.5/                    # Simuled Data, Learned Model Paramater, and Parameter Recovery results when VAR=0.5
├── VAR=1.0/                    # Simuled Data, Learned Model Paramater, and Parameter Recovery results when VAR=1.0
├── ru_mapping_model.py         # Reservation Utility Estimation Network
├── model_ru_parameter.pt       # Reservation Utility Estimation Network Parameter
├── Simulated _Date_PrePost_Shock.py           # Generting Simulated Data file "simulated_sessions.pkl"
├── deep_structural_embedding_prepost_shock.py           # SST model, training via real-world data or simulated data
└── README.md
```

## Usage
