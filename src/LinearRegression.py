import sys, time
from pandas import read_csv
from numpy import sqrt, argsort
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Load data
print("Loading data...")

try:
    df = read_csv("data/PulseBat Dataset.csv")
except FileNotFoundError:
    print("Invalid file path!")
    sys.exit(0)

# Featured Columns: U1 to U21
feature_columns = [f"U{i}" for i in range(1, 22)]

# Target: SOH
X = df[feature_columns]
y = df["SOH"]

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
rmse = sqrt(mean_squared_error(y_test, y_pred))

print("\n" + "="*50)
print("MODEL PERFORMANCE RESULTS")
print("="*50)
print(f"R² Score: {r2:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

# Sort for better visualization
sorted_indices = argsort(y_test.values)
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