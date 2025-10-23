# Battery SOH Prediction using Linear Regression

## Team Members
| Name | Student Number |
|------|----------------|
| Hashir Rashid |  |
| Burhanuddin Mohammed | 100943760 |
| Yahya Zouhar |  |
| Farhan Shameer |  |
|  |  |

---

## Project Overview

This project applies **Linear Regression** to predict the **State of Health (SOH)** of a lithium-ion battery pack using voltage readings from individual cells (`U1`–`U21`).  
It demonstrates a complete workflow — from data loading and preprocessing to model training, evaluation, and visualization.

---

## Features

-  Loads and cleans dataset from Excel  
-  Trains a **Linear Regression** model using `scikit-learn`  
-  Evaluates performance with **R²**, **MAE**, and **RMSE**  
-  Creates detailed visualizations:
  - Actual vs Predicted SOH  
  - Residuals (Error) plot  
  - Feature Importance (coefficients)  
  - Error Distribution histogram  
-  Displays model equation and performs an example SOH prediction  

---

##  Requirements

Install all dependencies before running:

```bash
pip install pandas numpy matplotlib scikit-learn openpyxl


# Required Packages:
- pandas
- openpyxl
- matplotlib
- scikit-learn

To install, open a python terminal and run:
```python
pip install <package-name>
```
---

# How to Run:
```python
python "src/LinearRegression.py"
```
