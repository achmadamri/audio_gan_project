import os
import librosa
import numpy as np
import soundfile as sf

# Function to load FLAC audio and convert to a specified sample rate
def load_audio(file_path, sample_rate=16000):
    audio, sr = librosa.load(file_path, sr=sample_rate)
    return audio

# Function to ensure each audio clip has the same length
def preprocess_audio(audio_data, target_length=16000):
    # Truncate if longer
    if len(audio_data) > target_length:
        audio_data = audio_data[:target_length]
    # Pad with zeros if shorter
    elif len(audio_data) < target_length:
        padding = target_length - len(audio_data)
        audio_data = np.pad(audio_data, (0, padding), 'constant')
    
    return audio_data

# Function to recursively process all FLAC files in the dataset directory
def process_dataset(dataset_dir, output_dir, sample_rate=16000, target_length=16000):
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith(".flac"):
                file_path = os.path.join(root, file)
                
                # Load and preprocess the audio file
                audio_data = load_audio(file_path, sample_rate)
                processed_audio = preprocess_audio(audio_data, target_length)
                
                # Maintain directory structure for output FLAC files
                relative_path = os.path.relpath(file_path, dataset_dir)
                output_file = os.path.join(output_dir, relative_path)  # Keep .flac extension
                output_file_dir = os.path.dirname(output_file)
                os.makedirs(output_file_dir, exist_ok=True)
                
                # Save the processed audio as FLAC
                sf.write(output_file, processed_audio, sample_rate, format='FLAC')
                print(f"Processed {file_path} -> {output_file}")

if __name__ == '__main__':
    dataset_dir = 'dataset/LibriSpeech/train-clean-100'  # Path to your extracted dataset
    output_dir = 'dataset/processed'                     # Output directory for preprocessed data
    os.makedirs(output_dir, exist_ok=True)               # Create the output directory if it doesn't exist
    process_dataset(dataset_dir, output_dir)
