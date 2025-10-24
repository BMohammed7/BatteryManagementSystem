# Battery SOH Prediction using Linear Regression

**Project:** Linear Regression model to predict battery State of Health (SOH) from cell voltages (U1–U21) 

## Team Members
| Name | Student Number |
|------|----------------|
| Hashir Rashid | 100910330 |
| Burhanuddin Mohammed | 100943760 |
| Yahya Zouhar | 100922007 |
| Farhan Shameer | 100906062 |
| Umair Ahmed | 100826767 |

---

## Overview

This repository contains a Python script that trains a linear regression model on a `PulseBat Dataset.csv` CSV file and produces evaluation metrics and plots showing how well the model predicts SOH from individual cell voltages.

This README explains how a TA can set up the environment, run the script, and verify the results.

---

## Repository Structure

```

root/
├── src/
│   └── Main.py                # main script
│   └── LinearRegression.py    # file with all custom functions used in Main
├── data/
│   └── PulseBat Dataset.xlsx  # dataset
│   └── PulseBat Dataset.csv   # dataset in csv form (for faster reading) 
├── requirements.txt           # pip installable requirements
└── README.md                  # this file

````

> **Note:** Ensure `PulseBat Dataset.csv` is placed in `data/` relative to the script path. Adjust the path in `Main.py` if you store the dataset elsewhere.

---

## Requirements

- Python 3.9+
- Python packages:
  - `numpy`
  - `matplotlib`
  - `scikit-learn`

---

## Setup Instructions

1. **Clone the repository**

```bash
git clone github.com/hashir-rashid/BatteryManagementSystem@latest
cd project-root
```

2. **Install Python packages**

```bash
pip install numpy matplotlib scikit-learn
```

3. **Place the dataset**

* Put `PulseBat Dataset.csv` into the `data/` folder.
* Confirm the file contains columns `SOH` and `U1..U21`.

4. **(Optional) Edit script path**

If your working directory differs, open `src/Main.py` and adjust the path in the `load_data()` function:

```python
data, headers = load_data("data/PulseBat Dataset.csv")
```

Change to the correct relative or absolute path if needed.

---

## Running the Script

From the project root, run:

```bash
python Main.py # from src/
# or
python src/Main.py # from root
```

**Expected Output:**

* Console output showing dataset shape, feature names, training/test sizes, and model metrics (R², MAE, RMSE).
* Matplotlib plots:
  * Sorted line plot comparing actual vs predicted across test samples
* Example printed model equation and a prediction using mean feature values.

> **Tip:** If running on a headless server (no GUI), either use an environment that supports plotting or modify the script to save figures:

```python
plt.savefig("figure.png")
```

---

## Troubleshooting and Common Issues

* **FileNotFoundError / Path issues:** Check the path to the CSV file and your working directory. Use an absolute path if necessary.
* **Missing columns:** Ensure `SOH` and `U1..U21` exist. Rename columns in the CSV file or update `feature_columns` and `y` selection in the script.
* **NaN values:** Inspect with:

```python
df[feature_columns].isna().sum()
y.isna().sum()
```

Impute or drop NaNs before fitting.

* **Plots not showing in Jupyter:** Add `%matplotlib inline` or `%matplotlib notebook` at the top of the notebook.

---

**End of README**
