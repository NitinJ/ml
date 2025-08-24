#!/usr/bin/env python3
"""
Training entry point for WaveNet model.

Command-line interface for training WaveNet on audio data with configurable
hyperparameters and training modes (full dataset or debug mode).
"""

import argparse
import torch
import torch.optim as optim
import os
import warnings

from config import (
    get_config, get_device, setup_torch, 
    get_checkpoint_config, get_training_config
)
from model import Wavenet, TrainableModel
from dataset import create_datasets, create_fake_datasets
from trainer import Trainer, sanity_check_batch, test_model_output_consistency

warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")


def parse_args():
    """Parse command line arguments for training configuration."""
    parser = argparse.ArgumentParser(description='Train WaveNet model for audio generation')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs to train (default: from config)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size (default: from config)')
    parser.add_argument('--learning-rate', type=float, default=None,
                        help='Learning rate (default: from config)')
    
    # Model parameters
    parser.add_argument('--residual-channels', type=int, default=None,
                        help='Number of residual channels (default: from config)')
    parser.add_argument('--skip-channels', type=int, default=None,
                        help='Number of skip channels (default: from config)')
    parser.add_argument('--n-layers', type=int, default=None,
                        help='Number of layers per block (default: from config)')
    parser.add_argument('--n-blocks', type=int, default=None,
                        help='Number of residual blocks (default: from config)')
    
    # Training modes
    parser.add_argument('--fake', action='store_true',
                        help='Use fake dataset for debugging/overfitting tests')
    parser.add_argument('--resume', action='store_true', default=True,
                        help='Resume from checkpoint if available (default: True)')
    parser.add_argument('--no-resume', dest='resume', action='store_false',
                        help='Do not resume from checkpoint, start fresh')
    
    # Data parameters
    parser.add_argument('--rebuild-dataset', action='store_true',
                        help='Rebuild tokenized dataset instead of using existing')
    parser.add_argument('--data-root', type=str, default=None,
                        help='Root directory for raw audio data')
    
    # Output parameters
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                        help='Directory for saving checkpoints')
    parser.add_argument('--plot-every', type=int, default=None,
                        help='Plot training progress every N epochs')
    
    # Testing parameters
    parser.add_argument('--test-only', action='store_true',
                        help='Only test model without training')
    parser.add_argument('--sanity-check', action='store_true',
                        help='Run sanity checks on data and model')
    
    return parser.parse_args()


def update_config_from_args(config, args):
    """Update configuration with command line arguments."""
    # Training parameters
    if args.epochs is not None:
        if args.fake:
            config['num_epochs_fake'] = args.epochs
        else:
            config['num_epochs'] = args.epochs
    
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    
    if args.learning_rate is not None:
        config['learning_rate'] = args.learning_rate
    
    # Model parameters
    if args.residual_channels is not None:
        config['residual_channels'] = args.residual_channels
    
    if args.skip_channels is not None:
        config['skip_channels'] = args.skip_channels
    
    if args.n_layers is not None:
        config['n_layers'] = args.n_layers
    
    if args.n_blocks is not None:
        config['n_blocks'] = args.n_blocks
    
    # Data parameters
    if args.data_root is not None:
        config['data_root'] = args.data_root
    
    # Output parameters
    if args.checkpoint_dir is not None:
        if args.fake:
            config['checkpoint_dir_fake'] = args.checkpoint_dir
        else:
            config['checkpoint_dir'] = args.checkpoint_dir
    
    if args.plot_every is not None:
        config['plot_every_n_epochs'] = args.plot_every
    
    # Training modes
    config['use_fake_dataset'] = args.fake
    
    return config


def setup_model_and_optimizer(config, device):
    """Initialize model, optimizer, and scheduler."""
    print("🚀 Initializing model...")
    
    # Create base model
    base_model = Wavenet(config).to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in base_model.parameters())
    trainable_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
    print(f"📊 Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    print(f"📏 Receptive field: {base_model.get_receptive_field():,} time steps")
    
    # Create optimizer and scheduler
    learning_rate = config['learning_rate']
    optimizer = optim.AdamW(base_model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=1, threshold=1e-3
    )
    
    return base_model, optimizer, scheduler


def setup_datasets(config, args):
    """Create training and validation datasets."""
    print("📁 Setting up datasets...")
    
    if config['use_fake_dataset']:
        print("🔬 Using fake dataset for debugging/overfitting tests")
        train_dataset, test_dataset, train_loader, test_loader = create_fake_datasets(config)
        # Update batch size for fake dataset
        config['batch_size'] = 1
        config['num_workers'] = 0
    else:
        print("📊 Using full dataset for training")
        use_existing = not args.rebuild_dataset
        train_dataset, test_dataset, train_loader, test_loader = create_datasets(
            config, use_existing_tokens=use_existing
        )
    
    return train_dataset, test_dataset, train_loader, test_loader


def run_sanity_checks(train_loader, test_loader, model, device):
    """Run various sanity checks on data and model."""
    print("\n🔍 Running sanity checks...")
    
    # Check data integrity
    print("\n1. Checking data batch integrity...")
    for i, batch in enumerate(train_loader):
        sanity_check_batch(batch)
        if i >= 2:  # Check first few batches
            break
    
    # Check model output consistency
    print("\n2. Testing model output consistency...")
    test_model_output_consistency(model, test_loader, device)
    
    print("✅ All sanity checks passed!")


def main():
    """Main training function."""
    args = parse_args()
    
    # Setup PyTorch and device
    setup_torch()
    device = get_device()
    print(f"🎯 Using device: {device}")
    
    # Get and update configuration
    config = get_config()
    config = update_config_from_args(config, args)
    
    # Determine checkpoint directory and number of epochs
    if config['use_fake_dataset']:
        checkpoint_dir = config['checkpoint_dir_fake']
        num_epochs = config['num_epochs_fake']
        print(f"🔬 Debug mode: Training for {num_epochs} epochs to overfit on single sample")
    else:
        checkpoint_dir = config['checkpoint_dir']
        num_epochs = config['num_epochs']
        print(f"🎯 Training mode: Training for {num_epochs} epochs on full dataset")
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Setup datasets
    train_dataset, test_dataset, train_loader, test_loader = setup_datasets(config, args)
    
    # Setup model and optimizer
    base_model, optimizer, scheduler = setup_model_and_optimizer(config, device)
    
    # Create trainable model wrapper
    print("🔧 Setting up trainable model...")
    trainable_model = TrainableModel(
        config=config,
        checkpoint_dir=checkpoint_dir,
        base_model=base_model,
        optimizer=optimizer,
        scheduler=scheduler,
        load_from_checkpoint=args.resume,
        device=device
    )
    
    # Run sanity checks if requested
    if args.sanity_check:
        run_sanity_checks(train_loader, test_loader, base_model, device)
    
    # Check if we only want to test
    if args.test_only:
        print("🧪 Test mode: Running model tests only")
        run_sanity_checks(train_loader, test_loader, base_model, device)
        return
    
    # Check if training is needed
    if trainable_model.trained_till_epoch_index + 1 >= num_epochs:
        print(f"✅ Model is already trained to {num_epochs} epochs.")
        print(f"📊 Best validation loss: {trainable_model.training_stats.get('best_val_loss', 'N/A')}")
        return
    
    # Create trainer
    print("🏃‍♂️ Setting up trainer...")
    trainer = Trainer(
        base_model=base_model,
        trainable_model=trainable_model,
        config=config,
        learning_rate=config['learning_rate'],
        optimizer=optimizer,
        scheduler=scheduler,
        checkpoint_dir=checkpoint_dir,
        train_loader=train_loader,
        val_loader=test_loader,
        device=device
    )
    
    # Start training
    print(f"\n🚀 Starting training...")
    print(f"📊 Training samples: {len(train_dataset):,}")
    print(f"📊 Validation samples: {len(test_dataset):,}")
    print(f"📊 Training batches: {len(train_loader):,}")
    print(f"📊 Validation batches: {len(test_loader):,}")
    
    try:
        trainer.train(num_epochs)
        print("\n🎉 Training completed successfully!")
        print(f"💾 Checkpoints saved to: {checkpoint_dir}")
        print(f"📈 Best validation loss: {trainer.training_stats['best_val_loss']:.4f}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        print(f"💾 Latest checkpoint saved to: {checkpoint_dir}")
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
