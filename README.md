# Software Effort Estimation - ISBSG Dataset

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

This repository contains the complete code for the paper:

**"Imputation-Based Enhancement of Software Development Effort Estimation using Machine Learning: A Study on the ISBSG Dataset"**

## 📋 Overview

This repository provides the implementation of **four imputation techniques** for the ISBSG R13 dataset, followed by **machine learning model training** and **evaluation**.

### Imputation Techniques

| # | Technique | Script | Description |
|---|-----------|--------|-------------|
| 1 | **Manual Domain-Expert** | `imputation/manual_imputation.py` | Rule-based imputation guided by ISBSG field descriptions |
| 2 | **Mean Imputation** | `imputation/mean_imputation.py` | Mean for numerical, mode for categorical |
| 3 | **Multiple Imputation (MICE)** | `imputation/multiple_imputation.py` | Iterative imputation using chained equations |
| 4 | **CART Imputation** | `imputation/CART_imputation.py` | Decision tree-based imputation |

### Machine Learning Models

- Linear Regression (LR)
- Support Vector Regression (SVR)
- Random Forest (RF)
- Extra Trees (ET)
- XGBoost
- LightGBM
- CatBoost

### Evaluation Metrics

- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² (Coefficient of Determination)

## 🚀 Features

- ✅ **No Data Leakage**: All imputation performed after train-test split
- ✅ **All Evaluation Metrics**: MAE, RMSE, R²
- ✅ **Hyperparameter Optimization**: Optuna for all models
- ✅ **Reproducible**: Fixed random seed (42)
- ✅ **Comprehensive**: 7 ML models + Ensembles

## 📁 Repository Structure
