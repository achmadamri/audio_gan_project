import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import soundfile as sf
import os

# Load the saved generator model
generator = load_model('models/generator_final.h5')

# Generate random noise input
noise = tf.random.normal([1, 100])

# Generate new audio
generated_audio = generator(noise, training=False)

# Save the generated audio
os.makedirs('generated', exist_ok=True)
sf.write('generated/new_generated_audio.flac', generated_audio[0].numpy().flatten(), 16000, format='FLAC')

print("New audio generated and saved as 'new_generated_audio.flac'")
