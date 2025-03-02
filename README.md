# pinn-london-traffic
This repository contains a hybrid physics-informed neural network (PINN) and long short-term memory (LSTM) model designed to predict traffic flows in an intersection in London.

## Repository Structure

### 1. `london_data/`
Contains the organized data obtained from the UTD19 dataset (https://utd19.ethz.ch/) used for training and testing the model.

### 2. `london_model.ipynb`
Jupyter Notebook: Implementation of the PINN-LSTM model architecture.

### 3. `iUROP_report_Sven_van_Loon.pdf`
Report explaining the methodology and results.

### 4. `load_data_london2.py`
Helper file for preprocessing and preparing time-series data for the LSTM.

### 5. `utd19_data_cleaning_steps/`
Contains files used for organizing the UTD19 dataset.

### 6. `utd19_data_cleaning_steps/load_utd19.ipynb`
Should bu run 1st, extracts all the rows that belong to London from the UTD19 dataset.

### 7. `utd19_data_cleaning_steps/organise_utd19.ipynb`
Should be run 2nd, it extracts all the cameras of interest out of the London cameras.

### 8. `utd19_data_cleaning_steps/convert_flows.ipynb`
Should be run 3rd, it combines flows from multiple lanes into one flow for the whole street.

### 8. `utd19_data_cleaning_steps/visualization`
Contains visualzation for the location of all the detectors in London UTD19 dataset.

### 8. `README.md`
This file. Provides an overview of the repository and its structure.
