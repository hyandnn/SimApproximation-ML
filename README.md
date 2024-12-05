# SimApproximation-ML

# Machine Learning Approximation of Simulation in Re-Manufacturing

## Overview

This project leverages Machine Learning (ML) techniques to approximate discrete event simulation (DES) in remanufacturing systems. By replacing time-intensive simulation processes with an efficient ML model, the project aims to significantly reduce computational time while maintaining high accuracy.

The project incorporates a complete pipeline:
1. **Data Preprocessing**: Cleansing and transforming simulation data.
2. **Neural Architecture Search (NAS)**: Using AutoML to find optimal model architectures.
3. **Model Training and Evaluation**: Training the selected models and evaluating their performance using metrics such as MAE, R², and MAPE.
4. **Post-Processing**: Visualizing results and comparing model performance.

## Features

- **Automated Machine Learning (AutoML)**: Supports multiple tuners, including Random Search, Greedy, Bayesian Optimization, and Hyperband.
- **GUI Interface**: A user-friendly interface built with `tkinter` for managing data preprocessing, training, evaluation, and prediction workflows.
- **Data Processing**: Flexible preprocessing options with scaling techniques like `MinMaxScaler` and `StandardScaler`.
- **Visualization**: Detailed performance visualizations, including MAE and training time comparisons.

## Requirements

To replicate the environment, use the provided `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate your_env_name
```

## Installation and Usage

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/hyandnn/SimApproximation-ML.git
   cd SimApproximation-ML
   ```

2. Install dependencies:
   ```bash
   conda env create -f environment.yml
   conda activate your_env_name
   ```

### Running the GUI

To launch the graphical interface:
```bash
python main.py
```

### User Guide

Refer to the included [User_Guide.pdf](./User_Guide.pdf) for detailed instructions on using the pipeline.

### Key Modules

- **Data Preprocessing** (`module_data_preprocessing.py`): Handles JSON data loading, extraction, cleaning, and scaling.
- **AutoML** (`module_automl.py`): Manages the NAS process, model training, evaluation, and predictions.
- **Post-Processing** (`module_data_postprocessing.py`): Generates visualizations for MAE and training time comparisons.

## Outputs

- Preprocessed data saved as `.npz` files for training and testing.
- Visualization results, including MAE plots (`mae_plot_diff.png`) and MAE vs. training time comparisons (`mae_time_comparison.png`).

## Contributions

Developed by [Haoling Yang](mailto:haoling.yang@rwth-aachen.de) as part of an initiative to integrate machine learning techniques into industrial simulation workflows.
