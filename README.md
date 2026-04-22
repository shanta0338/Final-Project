# Global Weather & Location Prediction Engine

This repository contains a complete deep learning pipeline for predicting global weather patterns, specifically focusing on temperature regression and geographical classification. Built with PyTorch and tracked via MLflow, the project processes comprehensive meteorological data to derive accurate predictions for temperature, location, and country.

Dataset: https://www.kaggle.com/datasets/nelgiriyewithana/global-weather-repository/code?datasetId=3678699&sortBy=dateRun&tab=profile&excludeNonAccessedDatasources=false

## 📌 Project Overview

The project is divided into two primary modelling tasks:
1. **Temperature Prediction (Regression):** A neural network designed to predict the current temperature (in Celsius) based on a variety of meteorological and astronomical features (e.g., wind speed, pressure, humidity, air quality, and moon phase illumination).
2. **Geographical Categorisation (Classification):** Two multi-class classification models that deduce the `location_name` and `country` solely from the surrounding weather and environmental conditions.

## 🗂 Repository Structure

* `Model.ipynb`: Contains the data pre-processing, standardisation, and training pipeline for the **Temperature Prediction** model. It features a Multi-Layer Perceptron (MLP) trained with Mean Squared Error (MSE) loss.
* `Location.ipynb`: Contains the training pipelines for the **Location** and **Country** classifiers. These models utilise deeper architectures with Batch Normalisation and Dropout to handle the high cardinality of the target variables (257 locations, 211 countries).

## 🚀 Technologies Used

* **Deep Learning:** PyTorch (with CUDA support for GPU acceleration)
* **Experiment Tracking:** MLflow
* **Data Processing:** Pandas, NumPy, Scikit-learn
* **Visualisation:** Matplotlib, Seaborn

## 📊 Model Architecture & Performance

### 1. Temperature Predictor (`Model.ipynb`)
* **Architecture:** 3-layer MLP with ReLU activations.
* **Performance Metrics:**
    * **R-squared (R²):** 0.9074
    * **Root Mean Squared Error (RMSE):** 2.9584
    * **Mean Absolute Error (MAE):** 1.8763
    * **Mean Squared Error (MSE):** 8.7522

### 2. Location & Country Classifiers (`Location.ipynb`)
* **Architecture:** PyTorch Neural Network featuring Linear layers, `BatchNorm1d`, `ReLU`, and `Dropout (0.3)` to prevent over-fitting on high-dimensional classes.
* **Performance Metrics:**
    * **Country Classification Accuracy:** 96.52%
    * **Location Classification Accuracy:** 95.10%

## 🔬 Experiment Tracking

All training runs, hyperparameters, and model artefacts are strictly versioned and logged using **MLflow**.
* **Experiments:** `Global_Weather_Temperature_Prediction` and `Global_Weather_Location_Prediction`.
* Logged parameters include epochs, batch size, learning rate, and optimiser details. Models are saved as MLflow artefacts for seamless downstream deployment.

## ⚙️ Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/shanta0338/Final-Project.git](https://github.com/shanta0338/Final-Project.git)
   cd Final-Project
