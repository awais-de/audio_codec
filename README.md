# About the Project
This course project develops a low-latency neural audio coder using Transformers to enable high-quality compression at low bitrates for real-time speech applications. The architecture employs sliding-window attention in the encoder to minimize delay, with a quantizer for efficient token transmission and a decoder for waveform reconstruction. It compares performance against baselines like Opus and simpler neural models, providing insights into complexity-latency-quality trade-offs. The prototype supports real-time demos via two-PC streaming, highlighting bitrate efficiency and interactive performance.​

# Features
Streaming Transformer Pipeline: Causal attention ensures <20 ms latency for live speech processing.​

Flexible Quantization: Supports RVQ or FSQ for 8–16 kbps, producing discrete tokens for transmission.​

Quality Metrics Integration: Computes PESQ, STOI, and log-spectral distance during evaluation.​

Real-Time Capabilities: RTF <1 on CPU/GPU, with socket-based demo for two-PC teleconferencing.​

Modular and Configurable: YAML configs for hyperparameters like 8 layers and 16 heads.​

# Getting Started
To set up and run the project, follow these steps for reproducibility on a standard laptop or PC.​

# Prerequisites
Python 3.10 or higher.​

Git for repository cloning.​

Optional: NVIDIA GPU with CUDA for accelerated training.​

Audio datasets: LibriSpeech for training and VCTK for testing.​

# Installation
Clone the repository to your local machine :​

text
git clone https://github.com/yourusername/TransformerAudioCodec.git
cd TransformerAudioCodec
Create and activate a virtual environment, then install dependencies :​

text
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
pip install -r requirements.txt
Key packages include torch, torchaudio, transformers, pesq, and stois.​

Preprocess datasets by placing raw files in data/raw/ and running :​

text
python src/utils/data_preprocess.py --input_dir data/raw/ --output_dir data/processed/
This resamples to 16 kHz and segments into 20 ms frames.​

# Usage
The project provides scripts for training, inference, evaluation, and demos.​

# Training
Train the model using MSE/STFT losses on LibriSpeech :​

text
python scripts/train.py --config configs/train_config.yaml --data_dir data/processed/train/
This optimizes for 100 epochs, saving models to checkpoints/ with progressive bitrate scheduling. Use TensorBoard to track metrics like log-spectral distance.​

# Inference
Perform compression and decompression on audio files :​

text
python scripts/infer.py --input_path data/raw/sample.wav --output_path reconstructed.wav --bitrate 12 --checkpoint checkpoints/best.pth
The pipeline encodes to tokens, quantizes, and decodes back to waveform at 16 kHz.​

# Evaluation
Compare model performance with baselines :​

text
python scripts/eval.py --checkpoint checkpoints/best.pth --test_dir data/processed/test/ --baseline opus
Outputs include PESQ, STOI, bitrate, latency (<20 ms), and RTF <1. Visualize spectrograms in notebooks for quality assessment.​

# Real-Time Demo
Demonstrate live streaming between two PCs :​

On sender PC (capture and encode):

text
python demos/realtime_demo.py --mode encode --host 0.0.0.0 --port 5000
On receiver PC (decode and playback):

text
python demos/realtime_demo.py --mode decode --host sender_ip --port 5000
This transmits compressed speech with live stats on latency, bitrate, and PESQ. Refer to demos/demo_setup.md for network configuration.​​

# Architecture
The codec follows an encoder-quantizer-decoder structure :​

Encoder: Stack of 8 transformer layers with 16 heads and window size 32 for causal, streaming attention on STFT features.​

Quantizer: RVQ with multiple codebooks for 40–80 bits per frame, enabling low-bitrate discretization.​

Decoder: Symmetric transformers to reconstruct the waveform, optionally with a GAN discriminator for perceptual gains.​
Detailed diagrams are in docs/architecture_diagram.md.​

Datasets
LibriSpeech: Clean/read subsets for training at 16 kHz.​

VCTK: Diverse speakers for testing robustness.​
Preprocessing scripts handle waveform loading, STFT computation, and frame segmentation. Do not commit raw data; use download scripts.​

Evaluation Criteria
Latency: End-to-end <20 ms, measured via pipeline timing.​

Bitrate: 8–16 kbps after quantization and coding.​

Audio Quality: PESQ ≥ 3.5, STOI ≥ 0.9 on standard benchmarks.​

Runtime: Real-time factor <1 on CPU.​

Comparisons: Bitrate-quality advantages over Opus/Encodec, visualized with spectrograms.​

