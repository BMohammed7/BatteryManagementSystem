from csv import reader
from numpy import sqrt, array
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sys import exit

def load_data(file_path: str) -> tuple:
    try:
        with open(file_path, 'r', newline='') as csvfile:
            read = reader(csvfile)
            headers = next(read)
            data = []
            for row in read:
                # Extract only the useful columns (Cell Voltages and SOH) 
                selected_data = row[8:30]
                data.append([float(x) for x in selected_data])
        
        # Convert to list of lists and then to numpy array
        # Assuming the last column is the target variable
        data_array = array(data, dtype=float)
        
    except FileNotFoundError:
        print("Invalid file path!")
        exit(0)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        exit(0)

    return data_array, headers

def train_data(X, y, test_size=0.2, random_state=42) -> tuple:
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")

    # Train the Linear Regression model
    print("\nTraining Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)

    return model, (X_train, X_test, y_train, y_test)

def calculate_metrics(test_data: list[float], predicted_data: list[float]) -> tuple:
    # Use scikit methods to calculate statistics
    r2 = r2_score(test_data, predicted_data)
    mae = mean_absolute_error(test_data, predicted_data)
    rmse = sqrt(mean_squared_error(test_data, predicted_data))

    return r2, mae, rmse

def visualize_data(x_axis, y_axis) -> None:
    plt.plot(range(len(x_axis)), x_axis, 'o-', label='Actual SOH', linewidth=2, markersize=4, color='blue')
    plt.plot(range(len(y_axis)), y_axis, 's-', label='Predicted SOH', linewidth=2, markersize=4, color='red', alpha=0.7)
    plt.xlabel('Test Sample Index (sorted by actual SOH)')
    plt.ylabel('SOH Value')
    plt.title('Actual vs Predicted SOH Across Test Samples')
    plt.legend()
    plt.grid(True, alpha=0.3)
    # plt.show()
    plt.savefig('LinearResults.png')

def classify_battery_health(soh_value: float) -> str:
    """
    Classifies battery health based on the project rule:
    - If SOH < 0.6 -> "The battery has a problem."
    - If SOH >= 0.6 -> "The battery is healthy."
    """
    if soh_value < 0.6:
        return "The battery has a problem."
    else:
        return "The battery is healthy."

def example_prediction(input_data, model) -> None:
    print("\n" + "="*50)
    print("EXAMPLE PREDICTION")
    print("="*50)
    print("Generating example prediction using average cell voltages...")

    # Use average values from training set as example input
    example_voltages = input_data.mean(axis=0).reshape(1, -1)
    predicted_soh = model.predict(example_voltages)[0]

    print(f"Example input voltages (U1-U21 averages):")
    for i, voltage in enumerate(example_voltages[0], 1):
        print(f"  U{i}: {voltage:.3f}V")
    
    print(f"\nPredicted SOH for this battery pack: {predicted_soh:.4f}")
    
    status = classify_battery_health(predicted_soh)
    print(f"Battery Status: {status}")


