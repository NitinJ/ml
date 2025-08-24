"""
Configuration module for WaveNet training and inference.

Contains all hyperparameters and configuration settings for the WaveNet model,
training process, and data processing pipeline.
"""

import torch
import os

# Device configuration
def get_device():
    """Get the appropriate device for computation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

# Setup global PyTorch settings
def setup_torch():
    """Configure PyTorch for optimal performance and reproducibility."""
    # Set random seeds for reproducibility
    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)
    
    # Configure CUDA settings
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        print(f"Compute Capability: {major}.{minor}")
        if major >= 8:
            print("✅ TF32 is supported (Ampere or newer).")
        else:
            print("❌ TF32 is not supported.")
        
        # Performance optimizations
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    
    # Environment variables for debugging
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['TORCH_USE_CUDA_DSA'] = '1'

# Training configuration
TRAINING_CONFIG = {
    "batch_size": 4,
    "num_workers": 2,
    "pin_memory": True,
    "learning_rate": 1e-3,
    "num_epochs": 50,
    "num_epochs_fake": 180,  # For overfitting tests
    "gradient_clip_norm": 1.0,
    "use_fake_dataset": False,  # Set to True for debugging/overfitting tests
}

# Audio processing configuration
AUDIO_CONFIG = {
    "mu": 256,
    "sr": 16000,
    "trim_silence_thresh": 1e-3,
    "window_size": 32001,  # ~2 seconds at 16kHz
    "hop_size": 16000,
}

# WaveNet model architecture configuration
MODEL_CONFIG = {
    "residual_channels": 64,
    "skip_channels": 256,
    "output_dim": 256,
    "n_layers": 10,
    "n_blocks": 5,
    "kernel_size": 2,
}

# Dataset configuration
DATASET_CONFIG = {
    "data_root": "./data",
    "segmented_tokens_dir": "./segmented_tokens",
    "manifest_file": "segmented_tokens/manifest.json",
    "shard_size": 10000,
    "cache_shards": False,
    "train_split": 0.9,
    "test_split": 0.1,
}

# Checkpoint configuration
CHECKPOINT_CONFIG = {
    "checkpoint_dir": "./wavenet_checkpoints",
    "checkpoint_dir_fake": "./wavenet_checkpoints_fake",
    "save_every_n_epochs": 1,
    "plot_every_n_epochs": 5,
}

# Generation configuration
GENERATION_CONFIG = {
    "temperature": 1.0,
    "top_k": 100,
    "seed_seconds": 0.3,
    "generate_seconds": 1.7,
    "use_fp32": True,
}

# Combined configuration
def get_config():
    """Get the complete configuration dictionary."""
    config = {}
    config.update(TRAINING_CONFIG)
    config.update(AUDIO_CONFIG)
    config.update(MODEL_CONFIG)
    config.update(DATASET_CONFIG)
    config.update(CHECKPOINT_CONFIG)
    config.update(GENERATION_CONFIG)
    return config

# Convenience function to get specific config sections
def get_training_config():
    return TRAINING_CONFIG.copy()

def get_audio_config():
    return AUDIO_CONFIG.copy()

def get_model_config():
    return MODEL_CONFIG.copy()

def get_dataset_config():
    return DATASET_CONFIG.copy()

def get_checkpoint_config():
    return CHECKPOINT_CONFIG.copy()

def get_generation_config():
    return GENERATION_CONFIG.copy()
