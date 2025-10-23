import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Set style for better plots
plt.style.use('default')

# Load data
print("Loading data...")
df = pd.read_excel("data/PulseBat Dataset.xlsx", sheet_name="SOC ALL")

# Display basic info about the dataset
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Features: U1 to U21
feature_columns = [f"U{i}" for i in range(1, 22)]
print(f"Using features: {feature_columns}")

# Check if all feature columns exist
missing_cols = [col for col in feature_columns if col not in df.columns]
if missing_cols:
    print(f"Warning: Missing columns: {missing_cols}")
    # Use only available columns
    feature_columns = [col for col in feature_columns if col in df.columns]

# Target: SOH
X = df[feature_columns]
y = df["SOH"]

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Target range: {y.min():.3f} to {y.max():.3f}")

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Testing set: {X_test.shape[0]} samples")

# Train Linear Regression model
print("\nTraining Linear Regression model...")
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate metrics
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n" + "="*50)
print("MODEL PERFORMANCE RESULTS")
print("="*50)
print(f"R² Score: {r2:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Number of features: {len(feature_columns)}")

# Display coefficients
coefficients = pd.DataFrame({
    'Feature': feature_columns,
    'Coefficient': model.coef_
}).sort_values('Coefficient', key=abs, ascending=False)

print("\nTop 10 Most Important Features:")
print(coefficients.head(10))

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Linear Regression: SOH Prediction Results', fontsize=16, fontweight='bold')

# Plot 1: Actual vs Predicted values
axes[0, 0].scatter(y_test, y_pred, alpha=0.6, color='blue', label='Predictions')
axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
axes[0, 0].set_xlabel('Actual SOH')
axes[0, 0].set_ylabel('Predicted SOH')
axes[0, 0].set_title(f'Actual vs Predicted SOH\nR² = {r2:.4f}')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Residuals plot
residuals = y_test - y_pred
axes[0, 1].scatter(y_pred, residuals, alpha=0.6, color='green')
axes[0, 1].axhline(y=0, color='red', linestyle='--', linewidth=2)
axes[0, 1].set_xlabel('Predicted SOH')
axes[0, 1].set_ylabel('Residuals')
axes[0, 1].set_title('Residuals Plot')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Feature importance (coefficients)
colors = ['red' if coef < 0 else 'blue' for coef in coefficients['Coefficient']]
axes[1, 0].barh(coefficients['Feature'], coefficients['Coefficient'], color=colors, alpha=0.7)
axes[1, 0].axvline(x=0, color='black', linestyle='-', linewidth=0.8)
axes[1, 0].set_xlabel('Coefficient Value')
axes[1, 0].set_ylabel('Features')
axes[1, 0].set_title('Feature Importance (Linear Regression Coefficients)')
axes[1, 0].grid(True, alpha=0.3, axis='x')

# Plot 4: Distribution of errors
axes[1, 1].hist(residuals, bins=30, alpha=0.7, color='purple', edgecolor='black')
axes[1, 1].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
axes[1, 1].set_xlabel('Prediction Error')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title(f'Distribution of Prediction Errors\nMAE = {mae:.4f}')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Additional detailed plot: Prediction comparison over samples
plt.figure(figsize=(14, 6))

# Sort for better visualization
sorted_indices = np.argsort(y_test.values)
y_test_sorted = y_test.values[sorted_indices]
y_pred_sorted = y_pred[sorted_indices]

plt.plot(range(len(y_test_sorted)), y_test_sorted, 'o-', label='Actual SOH', linewidth=2, markersize=4, color='blue')
plt.plot(range(len(y_pred_sorted)), y_pred_sorted, 's-', label='Predicted SOH', linewidth=2, markersize=4, color='red', alpha=0.7)
plt.xlabel('Test Sample Index (sorted by actual SOH)')
plt.ylabel('SOH Value')
plt.title('Actual vs Predicted SOH Across Test Samples')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Print model equation (first few terms for demonstration)
print("\n" + "="*50)
print("MODEL EQUATION (first 5 terms)")
print("="*50)
print("SOH = {:.4f}".format(model.intercept_))
for i, (feature, coef) in enumerate(zip(feature_columns, model.coef_)):
    if i < 5:  # Show first 5 terms
        print(f"     + ({coef:.4f}) * {feature}")

if len(feature_columns) > 5:
    print(f"     + ... (+ {len(feature_columns) - 5} more terms)")

# Example prediction for a new battery pack
print("\n" + "="*50)
print("EXAMPLE PREDICTION")
print("="*50)
print("Generating example prediction using average cell voltages...")

# Use average values from training set as example input
example_voltages = X_train.mean().values.reshape(1, -1)
predicted_soh = model.predict(example_voltages)[0]

print(f"Example input voltages (U1-U21 averages):")
for i, voltage in enumerate(example_voltages[0], 1):
    print(f"  U{i}: {voltage:.3f}V")

print(f"\nPredicted SOH for this battery pack: {predicted_soh:.4f}")

print("\n" + "="*50)
print("ANALYSIS COMPLETE")
print("="*50)
print("The linear regression model can predict battery pack SOH")
print("based on voltage readings from 21 individual cells.")