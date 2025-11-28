import streamlit as st
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from google import genai
from google.genai import types

# --- IMPORT YOUR MODULE ---
try:
    from linearRegression import load_data, train_data, calculate_metrics, visualize_data
except ImportError:
    st.error("linearRegression.py not found! Please make sure it is in the same directory.")
    st.stop()

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Battery SOH Assistant",
    page_icon="",
    layout="wide"
)


# --- CACHED FUNCTIONS (To speed up the app) ---
@st.cache_resource
def get_trained_model():
    """
    Loads data and trains the model once.
    Streamlit will remember this result so it doesn't retrain on every click.
    """
    # Try different paths for the dataset
    paths = ["PulseBat Dataset.csv", "../data/PulseBat Dataset.csv", "data/PulseBat Dataset.csv"]
    data_path = None
    for p in paths:
        if os.path.exists(p):
            data_path = p
            break

    if not data_path:
        return None, None

    data, headers = load_data(data_path)
    X = data[:, :-1]
    y = data[:, -1]

    # Train
    model, (X_train, X_test, y_train, y_test) = train_data(X, y)
    return model, (X_train, X_test, y_train, y_test)


def generate_plot_image(voltages, predicted_soh, threshold):
    """Generates the plot and returns the filename"""
    plt.figure(figsize=(10, 6))
    cell_indices = np.arange(1, 22)
    bars = plt.bar(cell_indices, voltages, color='skyblue', edgecolor='navy')

    avg_voltage = np.mean(voltages)
    plt.axhline(y=avg_voltage, color='r', linestyle='--', label=f'Avg: {avg_voltage:.2f}V')

    plt.title(f"Battery Cell Voltages (U1-U21)\nPredicted SOH: {predicted_soh:.4f} (Threshold: {threshold})")
    plt.xlabel("Cell Number")
    plt.ylabel("Voltage")
    plt.legend()

    for bar, val in zip(bars, voltages):
        if val < avg_voltage * 0.95:
            bar.set_color('salmon')

    filename = "temp_plot.png"
    plt.savefig(filename)
    plt.close()
    return filename


# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.header("Configuration")

    # API Key Handling
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        api_key = st.text_input("Enter Gemini API Key", type="password")

    # Threshold Handling
    soh_threshold = st.slider("SOH Threshold", 0.0, 1.0, 0.6, 0.05)

    st.divider()
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    st.info("University Project: Linear Regression + Chatbot Integration")

# --- MAIN APP LOGIC ---
st.title("Battery Pack SOH Assistant")

# Initialize Model
model, datasets = get_trained_model()

if not model:
    st.error("Could not find 'PulseBat Dataset.csv'. Please check your file structure.")
    st.stop()

X_train, X_test, y_train, y_test = datasets

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- TABS FOR DIFFERENT MODES ---
tab1, tab2 = st.tabs(["AI Chatbot & Analysis", "mb_chart Model Performance"])

# === TAB 1: CHATBOT ===
with tab1:
    # Display Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            # Check if the message has an image path attached
            if "image" in message:
                st.image(message["image"])
            st.markdown(message["content"])

    # Chat Input
    if prompt := st.chat_input("Ask about batteries, or type 'check' to test a pack..."):

        if not api_key:
            st.error("Please enter your API Key in the sidebar.")
            st.stop()

        # 1. Add User Message to History
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Process Response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            image_to_show = None

            try:
                client = genai.Client(api_key=api_key)
                # Use 1.5-flash (stable) or 2.0-flash-exp. 2.5 does not exist yet.
                ai_model = "gemini-2.5-flash"

                system_instruction = f"""
                You are an expert Battery Analyst. Threshold: {soh_threshold}.
                - If provided with an image of voltages, identify weak cells and determine health based on threshold.
                - If asked general questions, answer with engineering precision.
                """

                # === SCENARIO A: CHECK BATTERY REQUEST ===
                if any(kw in prompt.lower() for kw in ["check", "analyze", "test", "soh"]):

                    # Run Math
                    idx = random.randint(0, len(X_test) - 1)
                    sample_voltages = X_test[idx]
                    actual_soh = y_test[idx]
                    pred_soh = model.predict(sample_voltages.reshape(1, -1))[0]

                    # Generate Image
                    img_path = generate_plot_image(sample_voltages, pred_soh, soh_threshold)
                    pil_image = Image.open(img_path)
                    image_to_show = img_path  # Save path to display in history later

                    st.image(img_path, caption="Voltage Analysis")

                    # AI Prompt
                    ai_prompt = f"""
                    Analyze this battery data.
                    - Predicted SOH: {pred_soh:.4f}
                    - Threshold: {soh_threshold}
                    - Actual SOH: {actual_soh:.4f}
                    Look at the image. Is this healthy?
                    """

                    response = client.models.generate_content(
                        model=ai_model,
                        config=types.GenerateContentConfig(system_instruction=system_instruction),
                        contents=[ai_prompt, pil_image]
                    )
                    full_response = response.text

                # === SCENARIO B: GENERAL CHAT ===
                else:
                    response = client.models.generate_content(
                        model=ai_model,
                        config=types.GenerateContentConfig(system_instruction=system_instruction),
                        contents=[prompt]
                    )
                    full_response = response.text

                # Display Response
                message_placeholder.markdown(full_response)

                # Save to History
                msg_data = {"role": "assistant", "content": full_response}
                if image_to_show:
                    msg_data["image"] = image_to_show
                st.session_state.messages.append(msg_data)

            except Exception as e:
                st.error(f"An error occurred: {e}")

# === TAB 2: MODEL STATS ===
with tab2:
    st.header("Model Performance Metrics")

    # Calculate Metrics
    y_pred = model.predict(X_test)
    r2, mae, rmse = calculate_metrics(y_test, y_pred)

    # Display Metrics in Columns
    col1, col2, col3 = st.columns(3)
    col1.metric("R² Score", f"{r2:.4f}")
    col2.metric("RMSE", f"{rmse:.4f}")
    col3.metric("MAE", f"{mae:.4f}")

    st.divider()

    st.subheader("Actual vs Predicted SOH")
    st.info("Visualizing test set results...")

    # Generate the linear regression plot
    from numpy import argsort

    sorted_indices = argsort(y_test)
    visualize_data(y_test[sorted_indices], y_pred[sorted_indices])

    # Display the saved plot
    st.image("LinearResults.png", caption="Model Accuracy Visualization", use_container_width=True)