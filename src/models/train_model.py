import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data')))
from src.data.signal_generator import generate_drone_signal, generate_bird_signal
from src.utils.spectrogram_utils import signal_to_spectrogram
from src.models.cnn_model import create_cnn_model
# Generate data
X, y = [], []
for _ in range(500):
    s_drone = generate_drone_signal()
    s_bird = generate_bird_signal()
    for sig, label in [(s_drone, 1), (s_bird, 0)]:
        spec = signal_to_spectrogram(sig)
        X.append(spec)
        y.append(label)

X = np.array(X)[..., np.newaxis]  # Shape: (samples, 128, 128, 1)
y = to_categorical(y, num_classes=2)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create model
model = create_cnn_model()

# Train model
model.fit(X_train, y_train, validation_split=0.2, epochs=10, batch_size=32)

# Save model
model.save("cnn_model.keras")
print("CNN model saved as cnn_model.keras")
