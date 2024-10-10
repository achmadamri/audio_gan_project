import os
import librosa
import soundfile as sf
import numpy as np

# Convert FLAC to WAV using librosa and soundfile
def flac_to_wav(input_dir, output_dir, sample_rate=16000):
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".flac"):
                flac_path = os.path.join(root, file)
                
                # Load the FLAC file
                audio_data, sr = librosa.load(flac_path, sr=sample_rate)
                
                # Define the output path for the WAV file
                relative_path = os.path.relpath(flac_path, input_dir)
                wav_path = os.path.join(output_dir, os.path.splitext(relative_path)[0] + ".wav")
                
                # Make sure the output directory exists
                os.makedirs(os.path.dirname(wav_path), exist_ok=True)
                
                # Save as WAV file
                sf.write(wav_path, audio_data, sr)
                print(f"Converted {flac_path} -> {wav_path}")

# Specify input (FLAC files) and output (WAV files) directories
input_dir = 'dataset/LibriSpeech/train-clean-100'  # Path to your FLAC files
output_dir = 'dataset/processed'                  # Path where you want to save WAV files
os.makedirs(output_dir, exist_ok=True)

# Convert the dataset
flac_to_wav(input_dir, output_dir)