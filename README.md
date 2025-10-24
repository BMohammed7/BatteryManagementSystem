# Battery SOH Prediction using Linear Regression

**Project:** Linear Regression model to predict battery State of Health (SOH) from cell voltages (U1–U21)  

## Team Members
| Name | Student Number |
|------|----------------|
| Hashir Rashid |  |
| Burhanuddin Mohammed | 100943760 |
| Yahya Zouhar |  |
| Farhan Shameer |  |
|  |  |

---

## Overview

This repository contains a Python script that trains a linear regression model on a `PulseBat Dataset.xlsx` Excel file (sheet: `SOC ALL`) and produces evaluation metrics and plots showing how well the model predicts SOH from individual cell voltages.

This README explains how a TA can set up the environment, run the script, and verify the results.

---

## Repository Structure

```

project-root/
├── src/
│   └── LinearRegression.py    # main script (the code you provided)
├── data/
│   └── PulseBat Dataset.xlsx  # dataset (not included in repo by default)
├── requirements.txt           # pip installable requirements
└── README.md                  # this file

````

> **Note:** Ensure `PulseBat Dataset.xlsx` is placed in `data/` relative to the script path. Adjust the path in `LinearRegression.py` if you store the dataset elsewhere.

---

## Requirements

- Python 3.9+ (3.10/3.11 are fine)  
- Python packages:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `scikit-learn`
  - `openpyxl` (Excel engine)  

Install dependencies with pip:

```bash
pip install pandas numpy matplotlib scikit-learn openpyxl
````

---

## Setup Instructions

1. **Clone the repository**

```bash
git clone <repo-url>
cd project-root
```

2. **Install Python packages**

```bash
pip install pandas numpy matplotlib scikit-learn openpyxl
```

3. **Place the dataset**

* Put `PulseBat Dataset.xlsx` into the `data/` folder.
* Confirm the file contains a sheet named `SOC ALL` and columns `SOH` and `U1..U21`.

4. **(Optional) Edit script path**

If your working directory differs, open `src/LinearRegression.py` and adjust the path at the top:

```python
df = pd.read_excel("../data/PulseBat Dataset.xlsx", sheet_name="SOC ALL")
```

Change to the correct relative or absolute path if needed.

---

## Running the Script

From the project root, run:

```bash
python LinearRegression.py
# or
python src/LinearRegression.py
```

**Expected Output:**

* Console output showing dataset shape, feature names, training/test sizes, and model metrics (R², MAE, RMSE).
* Matplotlib plots:

  * Actual vs Predicted scatter
  * Residuals plot
  * Coefficient bar chart
  * Error histogram
  * Sorted line plot comparing actual vs predicted across test samples
* Example printed model equation and a prediction using mean feature values.

> **Tip:** If running on a headless server (no GUI), either use an environment that supports plotting or modify the script to save figures:

```python
plt.savefig("figure.png")
```

---

## Troubleshooting and Common Issues

* **FileNotFoundError / Path issues:** Check the path to the Excel file and your working directory. Use an absolute path if necessary.
* **Missing columns:** Ensure `SOH` and `U1..U21` exist. Rename columns in Excel or update `feature_columns` and `y` selection in the script.
* **Engine errors when reading Excel:** Install `openpyxl` (`pip install openpyxl`) for `.xlsx` files.
* **NaN values:** Inspect with:

```python
df[feature_columns].isna().sum()
y.isna().sum()
```

Impute or drop NaNs before fitting.

* **Plots not showing in Jupyter:** Add `%matplotlib inline` or `%matplotlib notebook` at the top of the notebook.

---

**End of README**

```
