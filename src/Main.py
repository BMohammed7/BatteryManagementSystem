from numpy import argsort
from time import time
from LinearRegression import load_data, calculate_metrics, train_data, visualize_data, example_prediction

if __name__ == "__main__":
    s = time()
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
    example_prediction(X_train, model)

    print("="*50)
    print("ANALYSIS COMPLETE")
    print(time() - s)
    print("="*50)