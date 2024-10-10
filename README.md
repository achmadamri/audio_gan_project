# Audio GAN Project

This project implements a Generative Adversarial Network (GAN) for audio generation using **FLAC** files from the **LibriSpeech** dataset. The GAN models are built using TensorFlow/Keras and trained on the preprocessed audio files, saving the models during training, and generating audio samples.

## Features
- Load and preprocess **FLAC** audio files, maintaining the original directory structure.
- Train a GAN model to generate synthetic audio.
- Save the generator and discriminator models during training and after completion.
- Generate new audio samples using the saved models.

## Requirements
- Python 3.x
- Git
- Virtual Environment (optional)

## Setup and Usage

### Step 1: Clone the repository
Clone the repository to your local machine:

```bash
git clone https://github.com/yourusername/audio_gan_project.git
cd audio_gan_project
