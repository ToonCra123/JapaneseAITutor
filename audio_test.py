
import sounddevice as sd
import numpy as np

print("Available audio devices:")
print(sd.query_devices())

print("\nDefault devices:", sd.default.device)

# Generate a 440 Hz test tone for 2 seconds
sample_rate = 24000
duration = 2

t = np.linspace(
    0, duration,
    int(sample_rate * duration),
    endpoint=False
)

audio = 0.2 * np.sin(2 * np.pi * 440 * t)

print("Playing test sound...")

sd.play(audio, sample_rate)
sd.wait()

print("Finished!")
