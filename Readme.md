# Probabilistic Time Series Forecasting using Quantile Regression

## Overview

This project implements a **probabilistic forecasting pipeline** for time-series data using **LSTM-based quantile regression**.
Instead of predicting a single value, the model estimates **prediction intervals (P10, P50, P90)** to capture uncertainty in power consumption forecasting.

The system is designed with a strong focus on:

* **uncertainty modeling**
* **reproducibility**
* **evaluation of probabilistic forecasts**

---

## Key Features

* **Quantile Regression (P10, P50, P90)** using a custom pinball loss
* **LSTM-based sequence modeling** for temporal dependencies
* **End-to-end pipeline**:

  * preprocessing → training → evaluation
* **MLflow integration** for experiment tracking
* **Docker-based execution** for reproducibility
* **Evaluation using both point and probabilistic metrics**

---

## Methodology

### Dataset
**Dataset Source:- [Link](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)**</br></br>
The dataset contains household-level electricity measurements with the following features:</br>
#### Power-related features</br>

* Global_active_power</br>
* Global_intensity</br>
* Sub_metering_1</br>
* Sub_metering_2</br>
* Sub_metering_3</br>

#### Time-based features

* year</br>
* quarter</br>
* month</br>
* day</br>
* weekday</br>

### Data Processing

* Numerical feature selection
* Log transformation (`log1p`) for stability
* MinMax scaling
* Sliding window generation for time-series learning

### Model

* LSTM architecture for sequence modeling
* Dense output layer predicting multiple quantiles
* Custom **MultiQuantileLoss** (pinball loss)

### Training

* Early stopping for generalization
* Gradient clipping for stability
* MLflow logging (parameters, metrics, artifacts)

---

## Evaluation

The model is evaluated using:

* **MAE (P50)** → measures median prediction accuracy
* **Coverage (P10–P90)** → measures interval calibration

### Visualization

* Actual vs Predicted (P50)
* Prediction interval band (P10–P90)

---

## Results & Insights

* The model captures general trends but struggles with **spike prediction**
* Observed **systematic underestimation** in high-value regions
* Prediction intervals showed **calibration issues (coverage < expected)**
* Highlights limitations of:

  * deterministic models (MSE)
  * naive quantile regression under weak feature signal

This project emphasizes not just model building, but **critical evaluation of uncertainty estimates**

---

## Project Structure

```
src/
  Tensorflow/
    DataPreprocessing.py
    ModelTraining.py
    ModelEvaluation.py
  util/
    losses.py

config.yaml
requirements.txt
README.md
```

---

## How to Run

### 0. Docker Setup (Recommended)

This project is designed to run inside a Docker container for full reproducibility.

### Build Image

```bash
docker build -t quantile-forecasting .
```

### Run Container

```bash
docker run -it --gpus all -it -p 5000:5000 -v $(pwd):/workspace quantile-forecasting
```

### Inside Container

Run pipeline:

```bash
python -m src.Tensorflow.DataPreprocessing
python -m src.Tensorflow.ModelTraining
python -m src.Tensorflow.ModelEvaluation
```

---

### Environment Details

* Base Image: `tensorflow/tensorflow:2.15.0-gpu`
* Python virtual environment inside container
* Dependencies managed via `requirements.txt`

---

### Why Docker?

* Ensures consistent environment across machines
* Avoids dependency conflicts
* Reproducible ML experiments


## MLflow Tracking

* Logs:

  * training & validation loss
  * evaluation metrics (MAE, coverage)
  * model artifacts
  * source code and configuration

To launch UI:

```bash
mlflow ui --host 0.0.0.0 --port 5000
```

---

## Technologies Used

* TensorFlow / Keras
* MLflow
* NumPy, Pandas
* Matplotlib
* Docker

---

## Future Work

* Improve quantile calibration
* Explore alternative architectures (Transformers, diffusion models)
* Add richer temporal and exogenous features
* Model heteroskedastic uncertainty more effectively

---

