import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from google import genai
from google.genai import types

# --- IMPORT LINEAR REGRESSION MODULE ---
# This expects your LinearRegression.py to be in the same folder
try:
    from LinearRegression import load_data, train_data
except ImportError:
    print("\n[Error] LinearRegression.py not found!")
    print("Please ensure LinearRegression.py is in this folder.\n")
    sys.exit(1)


def generate_voltage_plot(voltages, predicted_soh, threshold):
    """
    Generates a bar chart of the 21 cell voltages and saves it as an image.
    """
    plt.figure(figsize=(10, 6))

    # Create bars
    cell_indices = np.arange(1, 22)  # U1 to U21
    bars = plt.bar(cell_indices, voltages, color='skyblue', edgecolor='navy')

    # Highlight the average line
    avg_voltage = np.mean(voltages)
    plt.axhline(y=avg_voltage, color='r', linestyle='--', label=f'Avg Voltage: {avg_voltage:.2f}V')

    # Formatting
    plt.title(f"Battery Cell Voltages (U1-U21)\nPredicted SOH: {predicted_soh:.4f} (Threshold: {threshold})")
    plt.xlabel("Cell Number (U1 - U21)")
    plt.ylabel("Voltage Response")
    plt.xticks(cell_indices)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Find potential outliers to highlight in color (simple heuristic)
    # If a cell is significantly below average, mark it red
    for bar, val in zip(bars, voltages):
        if val < avg_voltage * 0.95:  # Arbitrary 5% drop trigger
            bar.set_color('salmon')

    # Save the plot
    filename = "current_plot.png"
    plt.savefig(filename)
    plt.close()
    return filename


def main():
    print("--- Battery Pack SOH Assistant (Vision Edition) ---")
    print("University Project: Linear Regression + Image Analysis")
    print("-" * 60)

    # --- 1. CONFIGURATION ---
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        api_key = input("Enter your Gemini API Key: ").strip()

    if not api_key:
        print("Error: API Key is required.")
        return

    # Set Threshold
    try:
        threshold_input = input("Set SOH Threshold (default 0.6): ").strip()
        SOH_THRESHOLD = float(threshold_input) if threshold_input else 0.6
    except ValueError:
        SOH_THRESHOLD = 0.6

    print(f" Configuration Set: Threshold = {SOH_THRESHOLD}")

    # --- 2. TRAIN MODEL ---
    print("\n[System] Training Linear Regression Model...")
    try:
        # Check standard paths for the dataset
        if os.path.exists("../data/PulseBat Dataset.csv"):
            data_path = "../data/PulseBat Dataset.csv"
        elif os.path.exists("data/PulseBat Dataset.csv"):
            data_path = "data/PulseBat Dataset.csv"
        elif os.path.exists("PulseBat Dataset.csv"):
            data_path = "PulseBat Dataset.csv"
        else:
            print("[Error] 'PulseBat Dataset.csv' not found.")
            return

        data, headers = load_data(data_path)
        X = data[:, :-1]  # Features U1-U21
        y = data[:, -1]  # Target SOH

        model, (X_train, X_test, y_train, y_test) = train_data(X, y)
        print("[System] Model Trained!")

    except Exception as e:
        print(f"[Error] Training failed: {e}")
        return

    # --- 3. INITIALIZE GEMINI CLIENT ---
    try:
        client = genai.Client(api_key=api_key)
        # We use 1.5-flash because it supports IMAGE input (Vision)
        ai_model = "gemini-2.5-flash"
    except Exception as e:
        print(f"[Error] Connection failed: {e}")
        return

    # --- 4. SYSTEM PROMPT ---
    system_instruction = f"""
    You are an expert Battery Analyst for a university project.

    YOUR ROLE:
    1. Receive an image of battery cell voltages (U1-U21).
    2. Receive the calculated SOH (State of Health) and the Threshold.
    3. Analyze the image to identify which specific cells (bars in the chart) are "weak" (lower voltage than others).
    4. Explain to the user if the battery is Healthy or Unhealthy based on the SOH vs Threshold ({SOH_THRESHOLD}).
    5. Provide a technical recommendation (e.g., "Recycle," "Reuse," "Balance Cell U14").
    6. For general questions, provide detailed, insightful, and engineering-grade answers. Avoid being too brief.

    Keep answers concise, professional, and helpful for engineering students.
    """

    print("\n" + "=" * 60)
    print(" VISUAL CHATBOT READY")
    print(" - Type 'check' or 'analyze' to simulate a battery test.")
    print(" - Type 'exit' to quit.")
    print("=" * 60 + "\n")

    chat_history = []

    # --- 5. CHAT LOOP ---
    while True:
        try:
            user_input = input("You: ").strip()

            if user_input.lower() in ['exit', 'quit']:
                print("Goodbye!")
                break

            if not user_input:
                continue

            print("Assistant: ", end="", flush=True)

            # === BRANCH A: "CHECK BATTERY" (Generate Image & Analyze) ===
            if any(kw in user_input.lower() for kw in ["check", "analyze", "test", "soh"]):

                # 1. Pick random battery sample
                idx = random.randint(0, len(X_test) - 1)
                sample_voltages = X_test[idx]
                actual_soh = y_test[idx]

                # 2. Predict SOH (Linear Regression)
                # Reshape for prediction: (1, 21)
                pred_soh = model.predict(sample_voltages.reshape(1, -1))[0]

                # 3. Generate the Image
                image_path = generate_voltage_plot(sample_voltages, pred_soh, SOH_THRESHOLD)

                # 4. Prepare Prompt for Gemini
                analysis_prompt = f"""
                Analyze this battery pack data.
                - Predicted SOH: {pred_soh:.4f}
                - Threshold: {SOH_THRESHOLD}
                - Actual SOH (Ground Truth): {actual_soh:.4f}

                Look at the attached image. Which cells are deviations? Is this battery healthy?
                """

                # 5. Send Image + Text to Gemini
                # We need to open the image as a PIL Image object or bytes
                pil_image = Image.open(image_path)

                response = client.models.generate_content(
                    model=ai_model,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction
                    ),
                    contents=[
                        analysis_prompt,
                        pil_image  # Pass the image directly
                    ]
                )

                # Stream or Print response
                print(response.text)

                # Clean up image file to keep folder clean (optional)
                # os.remove(image_path)

                # Update history (Text only to save tokens for history)
                chat_history.append({"role": "user", "content": user_input})
                chat_history.append({"role": "model", "content": response.text})

            # === BRANCH B: NORMAL CONVERSATION ===
            else:
                # Text-only interaction
                response = client.models.generate_content(
                    model=ai_model,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction
                    ),
                    contents=[user_input]
                )
                print(response.text)
                chat_history.append({"role": "user", "content": user_input})
                chat_history.append({"role": "model", "content": response.text})

            print("-" * 20)

        except Exception as e:
            print(f"\n[Error]: {e}")


if __name__ == "__main__":
    main()
