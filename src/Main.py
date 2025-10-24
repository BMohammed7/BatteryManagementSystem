import time
from numpy import argsort
from LinearRegression import load_data, calculate_metrics, train_data, visualize_data

if __name__ == "__main__":
    # Load data
    print("Loading data...")
    data, headers = load_data("../data/PulseBat Dataset.csv")
    
    # Featured Columns: U1 to U21, Target: SOH
    X = data[:, :-1]  # All columns except last
    y = data[:, -1]   # Last column as target (SOH)

    # Train the data
    datasets = train_data(X, y)
    model = datasets[0]
    X_train, X_test, y_train, y_test = datasets[1]

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    r2, mae, rmse = calculate_metrics(y_test, y_pred)

    print("\n" + "="*50)
    print("MODEL PERFORMANCE RESULTS")
    print("="*50)
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

    # Sort for better visualization
    sorted_indices = argsort(y_test)
    y_test_sorted = y_test[sorted_indices]
    y_pred_sorted = y_pred[sorted_indices]

    visualize_data(y_test_sorted, y_pred_sorted)

    # Example prediction for a new battery pack
    print("\n" + "="*50)
    print("EXAMPLE PREDICTION")
    print("="*50)
    print("Generating example prediction using average cell voltages...")

    # Use average values from training set as example input
    example_voltages = X_train.mean(axis=0).reshape(1, -1)
    predicted_soh = model.predict(example_voltages)[0]

    print(f"Example input voltages (U1-U21 averages):")
    for i, voltage in enumerate(example_voltages[0], 1):
        print(f"  U{i}: {voltage:.3f}V")

    print(f"\nPredicted SOH for this battery pack: {predicted_soh:.4f}")

    print("\n" + "="*50)
    print("ANALYSIS COMPLETE")

    print("="*50)
