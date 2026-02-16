from scipy.signal import spectrogram
import numpy as np
import cv2

def signal_to_spectrogram(signal, fs=1000):
    f, t, Sxx = spectrogram(signal, fs=fs, nperseg=128, noverlap=64)
    Sxx_dB = 10 * np.log10(Sxx + 1e-10)
    resized = cv2.resize(Sxx_dB, (128, 128))
    return resized
'''by converting raw time-domain radar signals into a visual, image-like representation (spectrogram) that highlights 
the unique micro-Doppler signatures. This spectrogram then serves as powerful input for machine learning algorithms to 
distinguish between different types of targets.'''