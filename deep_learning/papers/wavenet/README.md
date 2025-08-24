# WaveNet Implementation

A clean, modular implementation of WaveNet for autoregressive audio generation, restructured from Jupyter notebooks into well-organized Python modules.

## Project Structure

```
wavenet/
├── main.py              # Main entry point for all operations
├── config.py            # Configuration parameters and settings
├── model.py             # WaveNet model architecture and model wrappers
├── dataset.py           # Dataset processing and loading utilities
├── trainer.py           # Training logic and progress tracking
├── train.py             # Standalone training script
├── inference.py         # Standalone inference script
├── impl.ipynb           # Original implementation notebook (unchanged)
├── datasetbuilder.ipynb # Original dataset builder notebook (unchanged)
└── README.md            # This file
```

## Quick Start

### 1. Prepare Dataset

First, prepare and tokenize the audio dataset:

```bash
python main.py prepare-dataset --data-root ./data --output-dir ./segmented_tokens
```

This will:
- Download the LJSpeech dataset if not present
- Process audio files (resample, normalize, segment)
- Convert to mu-law tokens and save in sharded format

### 2. Train Model

Train the WaveNet model on the full dataset:

```bash
python main.py train --epochs 50 --batch-size 4
```

For debugging/overfitting tests on a single sample:

```bash
python main.py train --fake --epochs 180
```

### 3. Generate Audio

Generate audio using the trained model:

```bash
# Generate from dataset sample
python main.py inference --seed-dataset-index 0 --generate-seconds 5.0

# Generate from your own audio file
python main.py inference --seed-file my_audio.wav --temperature 0.8

# Generate multiple samples with different settings
python main.py inference --batch-generate 5 --temperature 1.2 --top-k 100
```

### 4. Evaluate Model

Run evaluation tests on the trained model:

```bash
python main.py evaluate --test-all
```

## Command Reference

### Dataset Preparation

```bash
python main.py prepare-dataset [OPTIONS]
```

Options:
- `--data-root`: Directory for raw audio data (default: ./data)
- `--output-dir`: Output directory for tokenized data (default: ./segmented_tokens)
- `--shard-size`: Samples per shard (default: 10000)
- `--num-workers`: Worker processes (default: 8)
- `--force-rebuild`: Force rebuild even if dataset exists

### Training

```bash
python main.py train [OPTIONS]
```

Key options:
- `--epochs`: Number of training epochs
- `--batch-size`: Training batch size
- `--learning-rate`: Learning rate
- `--fake`: Use single sample for debugging
- `--resume/--no-resume`: Resume from checkpoint
- `--sanity-check`: Run data/model validation

Model architecture options:
- `--residual-channels`: Residual channel dimension
- `--skip-channels`: Skip connection channels
- `--n-layers`: Layers per residual block
- `--n-blocks`: Number of residual blocks

### Inference

```bash
python main.py inference [OPTIONS]
```

Key options:
- `--seed-file`: Audio file for seed
- `--seed-dataset-index`: Dataset index for seed
- `--seed-seconds`: Seed duration
- `--generate-seconds`: Generation duration
- `--temperature`: Sampling temperature (creativity)
- `--top-k`: Top-k filtering
- `--mode`: Generation mode (sample/greedy/nucleus)
- `--batch-generate`: Number of samples to generate

### Evaluation

```bash
python main.py evaluate [OPTIONS]
```

Options:
- `--test-generation`: Test generation quality
- `--test-consistency`: Test model consistency
- `--test-all`: Run all tests

## Configuration

The `config.py` file contains all hyperparameters and settings:

- **Model Architecture**: Channels, layers, blocks, kernel size
- **Training**: Batch size, learning rate, epochs, gradient clipping
- **Audio Processing**: Sample rate, window size, mu-law quantization
- **Dataset**: Paths, splits, caching options
- **Generation**: Temperature, top-k, seed length

You can override most settings via command line arguments.

## Model Architecture

The WaveNet implementation includes:

- **Causal Dilated Convolutions**: Preserve autoregressive property
- **Residual Blocks**: Enable deep networks with skip connections
- **Gated Activations**: tanh/sigmoid gating for information flow
- **Skip Connections**: Aggregate features across all layers
- **Mu-law Encoding**: 256-level quantization for audio

Key features:
- Receptive field: ~5k time steps (configurable)
- Mixed precision training (FP16)
- Gradient clipping for stability
- Checkpoint resumption
- Compiled models for inference speed

## Usage Examples

### Training Configurations

```bash
# Quick debug training (overfit single sample)
python main.py train --fake --epochs 50

# Full training with custom settings
python main.py train --epochs 100 --batch-size 8 --learning-rate 5e-4

# Resume training from checkpoint
python main.py train --resume --epochs 150

# Train with larger model
python main.py train --residual-channels 128 --skip-channels 512 --n-blocks 6
```

### Generation Variations

```bash
# Conservative generation (low temperature)
python main.py inference --temperature 0.5 --top-k 50

# Creative generation (high temperature)
python main.py inference --temperature 1.5 --mode nucleus --nucleus-p 0.9

# Long generation with custom seed
python main.py inference --seed-file voice.wav --generate-seconds 10.0

# Batch generation for diversity testing
python main.py inference --batch-generate 10 --save-plot
```

## Alternative Entry Points

You can also use the standalone scripts:

```bash
# Direct training
python train.py --epochs 50 --fake

# Direct inference
python inference.py --seed-dataset-index 5 --temperature 0.8
```

## Hardware Requirements

- **GPU**: CUDA-capable GPU recommended (tested on RTX 30xx series)
- **Memory**: 8GB+ GPU memory for training, 4GB+ for inference
- **Storage**: ~5GB for LJSpeech dataset + tokenized data

## Notes

- The original Jupyter notebooks (`impl.ipynb`, `datasetbuilder.ipynb`) remain unchanged
- All functionality has been extracted into modular Python files
- The code supports both debug mode (single sample overfitting) and full training
- Generated audio is saved as WAV files with optional waveform plots
- Training progress is automatically plotted and saved
- Checkpoints enable resumable training and easy inference
