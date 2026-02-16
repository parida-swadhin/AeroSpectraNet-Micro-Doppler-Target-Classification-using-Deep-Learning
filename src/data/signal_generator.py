import numpy as np
def generate_drone_signal(duration=1.0, fs=1000):
    t = np.linspace(0, duration, int(fs * duration))
    carrier = np.sin(2 * np.pi * 50 * t)  # base motion
    mod = np.sin(2 * np.pi * 10 * t) * 0.5  # blade modulation
    signal = carrier * (1 + mod)
    return signal

def generate_bird_signal(duration=1.0, fs=1000):
    t = np.linspace(0, duration, int(fs * duration))
    signal = np.sin(2 * np.pi * 20 * t) + 0.5 * np.sin(2 * np.pi * 40 * t)
    return signal

if __name__ == "__main__":
    print("🛠️ Synthetic signal generator (testing functions within this file)")

    # Generate and test signals
    drone_signal = generate_drone_signal()
    bird_signal = generate_bird_signal()

    print(f"Drone signal sample: {drone_signal[:5]}")
    print(f"Bird signal sample: {bird_signal[:5]}")
