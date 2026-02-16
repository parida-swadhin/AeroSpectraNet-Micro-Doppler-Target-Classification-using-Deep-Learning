import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model # Assuming you still use TensorFlow/Keras
import matplotlib.pyplot as plt
import io

# --- Dummy Spectrogram Utility (if src/utils/spectrogram_utils.py is not available) ---
# This function is crucial for creating the "spectrogram_data" that your model expects.
# Adjust fs, nperseg, noverlap, nfft based on your actual data's sampling rate
# and the requirements of your real spectrogram transformation.
try:
    from src.utils.spectrogram_utils import signal_to_spectrogram
except ImportError:
    st.warning("`src/utils/spectrogram_utils.py` not found. Using a dummy spectrogram function. Please ensure the utility file is available for full functionality.")
    def signal_to_spectrogram(signal, fs=1000, nperseg=128, noverlap=64, nfft=256):
        """
        Dummy function to simulate spectrogram generation if the actual utility is missing.
        Returns a 128x128 random array, suitable as input for the dummy model.
        For a real spectrogram, 'signal' length should be sufficient for nperseg.
        """
        # Ensure the output shape matches what your model expects (e.g., 128x128)
        # In a real scenario, this would apply STFT (e.g., scipy.signal.spectrogram)
        # to the 'signal' and return the magnitude spectrogram.
        return np.random.rand(128, 128) # Default output for dummy

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="Micro-Doppler Target Classifier", layout="centered")
st.title("🎯 Micro-Doppler Target Classifier")

# --- Model Loading ---
# This attempts to load your pre-trained Keras model.
# If the model is not found, a dummy model is created for demonstration.
try:
    try:
        model = load_model("cnn_model.keras")
        st.success("✅ CNN model loaded.")
    except Exception as e:
        st.warning(f"⚠️ Model 'cnn_model.keras' not found. Creating a dummy model for demonstration. Actual predictions will not be meaningful.")
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)), # Input shape for spectrograms
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(2, activation='softmax') # Assuming 2 classes: Bird, Drone
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        st.success("✅ Dummy CNN model created.")

except Exception as e:
    st.error(f"❌ Error loading or creating model: {e}")
    st.stop() # Stop the app if model fails to load/create

# --- Class Mappings ---
# Define how numerical labels in your CSV map to human-readable names.
class_mapping_names = {0: "Bird", 1: "Drone"}

# --- File Uploader ---
st.subheader("Upload Your Data")
uploaded_file = st.file_uploader("Upload CSV file (e.g., combined_microdoppler.csv)", type="csv")

all_predictions = [] # List to store prediction results for the summary table

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        # --- Data Validation ---
        if "label" not in df.columns or "value" not in df.columns:
            st.error("❌ The uploaded CSV file must have 'label' and 'value' columns.")
            st.stop()

        # Convert 'value' column to numeric, handling non-numeric entries
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df.dropna(subset=['value'], inplace=True) # Remove rows where 'value' is not a number

        if df.empty:
            st.error("❌ No valid numerical data found in the 'value' column after cleaning. Please ensure your 'value' column contains numbers.")
            st.stop()

        # --- Aggregate Signals by Class ---
        # Collect all signal points for each class (Bird and Drone)
        signals_by_class = {}
        for label_id in df['label'].unique():
            if label_id in class_mapping_names: # Only process known labels (0 or 1)
                signals_by_class[label_id] = df[df['label'] == label_id]['value'].values

        if not signals_by_class:
            st.error("❌ No valid labels (0 for Bird, 1 for Drone) found in the 'label' column. Please ensure your data contains these labels.")
            st.stop()

        st.info("Processing random segments from aggregated 'Bird' and 'Drone' signals...")

        # --- Configuration for Segment Processing ---
        # These values determine how segments are extracted and their minimum length.
        # Adjust 'fs' if your data has a different sampling rate.
        fs = 1000 # Sample rate (Hz) - used for duration calculation and spectrogram
        segment_length = 1000 # Ideal segment length in data points (e.g., 1000 for 1 second at 1000 Hz)
        
        # min_signal_for_spectrogram: Minimum length required to produce *any* meaningful spectrogram.
        # This typically relates to the 'nperseg' parameter in your actual spectrogram function.
        # If your signals are very short (e.g., consistently 128 as per previous errors),
        # this value needs to be adjusted accordingly (e.g., 128 or less).
        # Assuming model expects 128x128 spectrogram, a minimum of 128 for signal is often used.
        min_signal_for_spectrogram = 128 

        max_samples_per_class = 2 # Number of random segments to pick per class

        processed_count = 0

        # --- Process Each Class ---
        for class_id, full_signal in signals_by_class.items():
            class_name = class_mapping_names.get(class_id, "Unknown")
            st.subheader(f"Processing Segments for {class_name}")

            # Initial check for signal length
            if len(full_signal) < min_signal_for_spectrogram:
                st.warning(f"⚠️ Full signal for {class_name} is too short ({len(full_signal)} values). Skipping analysis for this class. Need at least {min_signal_for_spectrogram} values.")
                continue

            current_segment_length_to_use = segment_length

            # Dynamic segment length adjustment if full_signal is too short for 'segment_length'
            if len(full_signal) < segment_length:
                # If full signal is shorter than 'segment_length', try to use min_signal_for_spectrogram
                if len(full_signal) < min_signal_for_spectrogram:
                     st.warning(f"⚠️ Full signal for {class_name} is too short to extract even one {min_signal_for_spectrogram}-point segment. Skipping.")
                     continue
                else:
                    st.info(f"Full signal for {class_name} is too short for {segment_length}-point segments. Will attempt to use {min_signal_for_spectrogram}-point segments.")
                    current_segment_length_to_use = min_signal_for_spectrogram
            
            # Generate potential starting points for segments
            # Ensure we don't go out of bounds: len(full_signal) - current_segment_length_to_use + 1
            # Step by 'current_segment_length_to_use' for non-overlapping segments
            segment_starts = np.arange(0, len(full_signal) - current_segment_length_to_use + 1, current_segment_length_to_use)
            
            if len(segment_starts) == 0:
                st.warning(f"No sufficient segments could be extracted for {class_name} with length {current_segment_length_to_use}. Skipping.")
                continue

            # Randomly select segments for processing
            if len(segment_starts) > max_samples_per_class:
                chosen_starts = np.random.choice(segment_starts, max_samples_per_class, replace=False)
            else:
                chosen_starts = segment_starts # Use all available segments if fewer than max_samples_per_class
            
            if len(chosen_starts) == 0:
                st.warning(f"No segments were selected for {class_name} after random choice. This might happen if 'max_samples_per_class' is too high for available segments. Skipping.")
                continue

            # --- Process Each Chosen Segment ---
            for idx, start_point in enumerate(chosen_starts):
                processed_count += 1
                segment_id = f"{class_name}_Seg{idx+1}_Start{start_point}"
                current_segment = full_signal[start_point : start_point + current_segment_length_to_use]

                # --- Time Domain Signal Visualization ---
                st.subheader(f"Time Domain Signal for {segment_id}")
                fig_time, ax_time = plt.subplots(figsize=(10, 4))
                ax_time.plot(np.arange(len(current_segment)) / fs, current_segment) # Time in seconds
                ax_time.set_xlabel("Time (seconds)")
                ax_time.set_ylabel("Amplitude")
                ax_time.set_title(f"Time Domain Signal for {segment_id}")
                ax_time.grid(True)
                st.pyplot(fig_time)
                plt.close(fig_time) # Close figure to free memory

                # --- Spectrogram Generation and Visualization ---
                spectrogram_data = signal_to_spectrogram(current_segment, fs=fs) # Pass fs to dummy/real function
                st.subheader(f"Spectrogram for {segment_id}")
                fig_spec, ax_spec = plt.subplots()
                ax_spec.imshow(spectrogram_data, aspect='auto', cmap='viridis', origin='lower')
                ax_spec.set_xlabel("Time")
                ax_spec.set_ylabel("Frequency")
                ax_spec.set_title(f"Spectrogram for {segment_id}")
                st.pyplot(fig_spec)
                plt.close(fig_spec) # Close figure to free memory

                # --- Model Prediction ---
                # Ensure spectrogram_data is 128x128 for model input
                if spectrogram_data.shape != (128, 128):
                    st.warning(f"Spectrogram for {segment_id} has shape {spectrogram_data.shape}, but model expects (128, 128). Attempting resize...")
                    # Simple resizing (this might distort actual spectrograms for real data)
                    resized_spec = np.zeros((128, 128))
                    rows_spec, cols_spec = spectrogram_data.shape
                    resized_spec[:min(rows_spec, 128), :min(cols_spec, 128)] = spectrogram_data[:min(rows_spec, 128), :min(cols_spec, 128)]
                    spectrogram_data = resized_spec

                input_data = spectrogram_data.reshape(1, 128, 128, 1) # Reshape for CNN input (batch, height, width, channels)
                prediction = model.predict(input_data, verbose=0) # Get raw prediction probabilities
                predicted_class_id = np.argmax(prediction) # Get the class with highest probability
                predicted_class_name = class_mapping_names.get(predicted_class_id, "Unknown")
                confidence = np.max(prediction) * 100 # Convert max probability to percentage

                # --- Store Results for Summary Table ---
                all_predictions.append({
                    "Sample ID": segment_id,
                    "Actual": class_name,
                    "Predicted": predicted_class_name,
                    "Confidence": f"{confidence:.2f}%", # Format as percentage string
                    "Duration (s)": round(len(current_segment) / fs, 2) # Calculate duration
                })

        # --- Final Summary Table Display ---
        if not all_predictions:
            st.error("No valid segments could be processed from the uploaded CSV for prediction. Please check data format and signal length.")
            st.stop()

        st.subheader("📊 Summary Table")
        all_predictions_df = pd.DataFrame(all_predictions)
        st.dataframe(all_predictions_df) # Display the DataFrame as a table

        # --- Optional: Display Overall Metrics ---
        st.subheader("Overall Metrics for Processed Samples")
        correct_predictions = sum(1 for p in all_predictions if p['Actual'] == p['Predicted'])
        total_predictions = len(all_predictions)
        if total_predictions > 0:
            accuracy = (correct_predictions / total_predictions) * 100
            st.write(f"Total Segments Processed: **{total_predictions}**")
            st.write(f"Correct Predictions: **{correct_predictions}**")
            st.write(f"Accuracy on Processed Segments: **{accuracy:.2f}%**")
        else:
            st.write("No predictions were made.")

    except Exception as e:
        st.error(f"❌ An unexpected error occurred while processing the uploaded file: {e}")
        st.write("Please ensure your CSV is correctly formatted with 'label' and 'value' columns and that the 'value' column contains numerical data points.")

else:
    st.info("Please upload a CSV file to get per-segment predictions and see the summary table.")