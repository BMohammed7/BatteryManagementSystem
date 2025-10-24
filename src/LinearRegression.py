from pandas import read_csv
from numpy import sqrt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sys import exit

def load_data(file_path: str):
    try:
        df = read_csv(file_path)
    except FileNotFoundError:
        print("Invalid file path!")
        exit(0)

    return df

def calculate_metrics(test_data: list[float], predicted_data: list[float]) -> tuple:
    r2 = r2_score(test_data, predicted_data)
    mae = mean_absolute_error(test_data, predicted_data)
    rmse = sqrt(mean_squared_error(test_data, predicted_data))

    return r2, mae, rmse

def train_data(X, y) -> tuple:
    # Split data (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")

    # Train Linear Regression model
    print("\nTraining Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)

    return model, [X_train, X_test, y_train, y_test]

def visualize_data(x_axis, y_axis) -> None:
    plt.plot(range(len(x_axis)), x_axis, 'o-', label='Actual SOH', linewidth=2, markersize=4, color='blue')
    plt.plot(range(len(y_axis)), y_axis, 's-', label='Predicted SOH', linewidth=2, markersize=4, color='red', alpha=0.7)
    plt.xlabel('Test Sample Index (sorted by actual SOH)')
    plt.ylabel('SOH Value')
    plt.title('Actual vs Predicted SOH Across Test Samples')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()