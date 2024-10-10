import os
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import soundfile as sf

# Load preprocessed audio files
def load_audio_files(path, sr=16000):
    audio_data = []
    file_count = 0  # Track number of files
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(".flac"):
                file_path = os.path.join(root, file)
                audio, _ = librosa.load(file_path, sr=sr)  # Load FLAC and resample to 16kHz
                audio_data.append(audio[:sr])  # Use first 1 second (16k samples)
                file_count += 1
                print(f"Loaded file: {file}")
    print(f"Total files loaded: {file_count}")
    return np.array(audio_data)

# GAN Generator model
def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(256, use_bias=False, input_shape=(100,)))
    model.add(layers.LeakyReLU())
    model.add(layers.Dense(16000, use_bias=False))  # Match the discriminator's input size of 16000 samples
    model.add(layers.LeakyReLU())
    model.add(layers.Reshape((16000, 1)))  # Reshape to (16000, 1)
    return model

# GAN Discriminator model
def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv1D(64, 25, 4, 'same', input_shape=(16000, 1)))  # 16k samples, 1 channel
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv1D(128, 25, 4, 'same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.Flatten())
    model.add(layers.Dense(1))  # Binary classification (real or fake)
    return model

# Loss functions
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

# Training step
@tf.function
def train_step(audio_batch, generator, discriminator, generator_optimizer, discriminator_optimizer):
    noise = tf.random.normal([audio_batch.shape[0], 100])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_audio = generator(noise, training=True)
        real_output = discriminator(audio_batch, training=True)
        fake_output = discriminator(generated_audio, training=True)
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
    
    gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))

# Generate and save audio
def generate_and_save_audio(model, epoch, test_input, sr=16000):
    predictions = model(test_input, training=False)
    generated_audio = predictions[0].numpy().flatten()
    os.makedirs('generated', exist_ok=True)
    sf.write(f'generated/audio_epoch_{epoch}.flac', generated_audio, sr, format='FLAC')

# Training loop
def train(dataset, epochs, generator, discriminator, generator_optimizer, discriminator_optimizer):
    for epoch in range(epochs):
        print(f"Starting epoch {epoch + 1}/{epochs}")
        
        for step, audio_batch in enumerate(dataset):
            train_step(audio_batch, generator, discriminator, generator_optimizer, discriminator_optimizer)
            if step % 10 == 0:
                print(f"Step {step} in epoch {epoch + 1}")
        
        print(f"Epoch {epoch+1}/{epochs} completed.")
        
        # Generate and save audio after every 10 epochs
        if (epoch + 1) % 10 == 0:
            noise = tf.random.normal([1, 100])
            generate_and_save_audio(generator, epoch + 1, noise)
        
        # Save the generator and discriminator models after every 10 epochs
        if (epoch + 1) % 10 == 0:
            generator.save(f'models/generator_epoch_{epoch+1}.h5')
            discriminator.save(f'models/discriminator_epoch_{epoch+1}.h5')
            print(f"Models saved at epoch {epoch+1}")

    # Save the final models at the end of training
    generator.save('models/generator_final.h5')
    discriminator.save('models/discriminator_final.h5')
    print("Final models saved.")

# Main execution
if __name__ == '__main__':
    dataset_path = 'dataset/processed'  # Path to your processed FLAC files
    
    # Load the dataset
    audio_data = load_audio_files(dataset_path)
    
    # Add a check to see if data was loaded
    print(f"Total audio samples loaded: {len(audio_data)}")

    # Ensure the audio data is in the correct format
    audio_data = np.expand_dims(audio_data, -1)  # Add channel dimension if needed

    # Set batch size and prepare the dataset
    BATCH_SIZE = 32
    dataset = tf.data.Dataset.from_tensor_slices(audio_data).shuffle(10000).batch(BATCH_SIZE)

    # Initialize the generator and discriminator models
    generator = make_generator_model()
    discriminator = make_discriminator_model()

    # Optimizers
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    # Train the models
    EPOCHS = 100
    train(dataset, EPOCHS, generator, discriminator, generator_optimizer, discriminator_optimizer)
