#!/usr/bin/env python3
"""
Main entry point for WaveNet training, inference, and dataset preparation.

This script provides a unified interface for all WaveNet operations:
- Dataset preparation and tokenization
- Model training (full dataset or debug mode)
- Audio generation and inference
- Model evaluation and testing
"""

import argparse
import torch
import torch.optim as optim
import os
import sys
from pathlib import Path
import warnings

# Import all our modules
from config import (
    get_config, get_device, setup_torch, 
    get_checkpoint_config, get_training_config
)
from model import Wavenet, TrainableModel, EvalModel, codec
from dataset import (
    create_datasets, create_fake_datasets, 
    SegmentedTokensOnDisk, AudioProcessor,
    stream_preprocess_to_shards
)
from trainer import Trainer, sanity_check_batch, test_model_output_consistency
import torchaudio

warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")


def create_parser():
    """Create comprehensive argument parser for all operations."""
    parser = argparse.ArgumentParser(
        description='WaveNet: Training, Inference, and Dataset Preparation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prepare dataset
  python main.py prepare-dataset --data-root ./data --output-dir ./segmented_tokens

  # Train model
  python main.py train --epochs 50 --batch-size 4

  # Train in debug mode (overfit on single sample)
  python main.py train --fake --epochs 180

  # Generate audio
  python main.py inference --seed-dataset-index 0 --generate-seconds 5.0

  # Generate from audio file
  python main.py inference --seed-file my_audio.wav --temperature 0.8
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Dataset preparation command
    prepare_parser = subparsers.add_parser('prepare-dataset', help='Prepare and tokenize dataset')
    add_dataset_args(prepare_parser)
    
    # Training command
    train_parser = subparsers.add_parser('train', help='Train WaveNet model')
    add_training_args(train_parser)
    
    # Inference command
    inference_parser = subparsers.add_parser('inference', help='Generate audio using trained model')
    add_inference_args(inference_parser)
    
    # Evaluation command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate trained model')
    add_evaluation_args(eval_parser)
    
    return parser


def add_dataset_args(parser):
    """Add dataset preparation arguments."""
    parser.add_argument('--data-root', type=str, default='./data',
                        help='Root directory for raw audio data (default: ./data)')
    parser.add_argument('--output-dir', type=str, default='./segmented_tokens',
                        help='Output directory for tokenized data (default: ./segmented_tokens)')
    parser.add_argument('--shard-size', type=int, default=10000,
                        help='Number of samples per shard (default: 10000)')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='Number of worker processes (default: 8)')
    parser.add_argument('--batch-items', type=int, default=32,
                        help='Batch size for processing (default: 32)')
    parser.add_argument('--force-rebuild', action='store_true',
                        help='Force rebuild even if dataset exists')


def add_training_args(parser):
    """Add training arguments."""
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


def add_inference_args(parser):
    """Add inference arguments."""
    # Model parameters
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                        help='Directory containing trained model checkpoints')
    parser.add_argument('--checkpoint-path', type=str, default=None,
                        help='Specific checkpoint file to load')
    
    # Generation parameters
    parser.add_argument('--seed-file', type=str, default=None,
                        help='Audio file to use as seed for generation')
    parser.add_argument('--seed-dataset-index', type=int, default=None,
                        help='Index from dataset to use as seed')
    parser.add_argument('--seed-seconds', type=float, default=None,
                        help='Duration of seed audio in seconds')
    parser.add_argument('--generate-seconds', type=float, default=None,
                        help='Duration of audio to generate in seconds')
    parser.add_argument('--temperature', type=float, default=None,
                        help='Sampling temperature (1.0=normal, <1.0=conservative, >1.0=creative)')
    parser.add_argument('--top-k', type=int, default=None,
                        help='Top-k filtering (0=disabled, >0=only sample from top k tokens)')
    
    # Generation modes
    parser.add_argument('--mode', choices=['sample', 'greedy', 'nucleus'], default='sample',
                        help='Generation mode: sample, greedy, or nucleus sampling')
    parser.add_argument('--nucleus-p', type=float, default=0.9,
                        help='Nucleus sampling parameter (top-p)')
    
    # Output parameters
    parser.add_argument('--output-dir', type=str, default='./generated_audio',
                        help='Directory to save generated audio')
    parser.add_argument('--output-name', type=str, default=None,
                        help='Base name for output files (auto-generated if not specified)')
    parser.add_argument('--save-plot', action='store_true',
                        help='Save waveform plots along with audio')
    parser.add_argument('--batch-generate', type=int, default=1,
                        help='Generate multiple samples (default: 1)')


def add_evaluation_args(parser):
    """Add evaluation arguments."""
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                        help='Directory containing trained model checkpoints')
    parser.add_argument('--test-generation', action='store_true',
                        help='Run generation quality tests')
    parser.add_argument('--test-consistency', action='store_true',
                        help='Test model output consistency')
    parser.add_argument('--test-all', action='store_true',
                        help='Run all evaluation tests')


def prepare_dataset_command(args):
    """Handle dataset preparation command."""
    print("🗂️ Preparing dataset...")
    
    # Load base dataset
    print(f"📁 Loading dataset from: {args.data_root}")
    base_dataset = torchaudio.datasets.LJSPEECH(
        root=args.data_root,
        download=True
    )
    print(f"📊 Dataset loaded: {len(base_dataset)} samples")
    
    # Check if dataset already exists
    output_path = Path(args.output_dir)
    manifest_path = output_path / "manifest.json"
    
    if manifest_path.exists() and not args.force_rebuild:
        print(f"⚠️ Dataset already exists at {manifest_path}")
        print("Use --force-rebuild to rebuild anyway")
        return
    
    # Create audio processor
    audio_processor = AudioProcessor()
    
    # Build and save dataset
    print(f"🔄 Processing and tokenizing dataset...")
    print(f"   Shard size: {args.shard_size}")
    print(f"   Workers: {args.num_workers}")
    print(f"   Batch items: {args.batch_items}")
    
    manifest_path = stream_preprocess_to_shards(
        base_dataset=base_dataset,
        audio_processor=audio_processor,
        out_dir=args.output_dir,
        shard_size=args.shard_size,
        num_workers=args.num_workers,
        batch_items=args.batch_items,
    )
    
    print(f"✅ Dataset preparation completed!")
    print(f"📄 Manifest saved to: {manifest_path}")
    
    # Load and show dataset info
    dataset = SegmentedTokensOnDisk(str(manifest_path))
    print(f"📊 Final dataset: {len(dataset)} samples")
    print(f"📏 Sequence length: {dataset.T}")


def train_command(args):
    """Handle training command."""
    # Setup PyTorch and device
    setup_torch()
    device = get_device()
    print(f"🎯 Using device: {device}")
    
    # Get and update configuration
    config = get_config()
    config = update_training_config_from_args(config, args)
    
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
    train_dataset, test_dataset, train_loader, test_loader = setup_datasets_for_training(config, args)
    
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
    
    # Create trainer and start training
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
    
    print(f"\n🚀 Starting training...")
    print(f"📊 Training samples: {len(train_dataset):,}")
    print(f"📊 Validation samples: {len(test_dataset):,}")
    
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


def inference_command(args):
    """Handle inference command."""
    from inference import (
        generate_continuation, generate_greedy, 
        load_seed_audio, load_seed_from_dataset,
        save_audio_and_plot
    )
    
    # Setup PyTorch and device
    setup_torch()
    device = get_device()
    print(f"🎯 Using device: {device}")
    
    # Get and update configuration
    config = get_config()
    config = update_inference_config_from_args(config, args)
    
    # Determine checkpoint directory
    checkpoint_dir = args.checkpoint_dir or config['checkpoint_dir']
    
    # Load model
    model = load_model_for_inference(checkpoint_dir, args.checkpoint_path, config, device)
    
    # Prepare seed audio
    audio_processor = AudioProcessor()
    seed_tokens, seed_audio = prepare_seed_audio(args, audio_processor, config)
    
    # Calculate generation parameters
    seed_length = int(config['seed_seconds'] * config['sr'])
    if seed_tokens.size(1) > seed_length:
        seed_tokens = seed_tokens[:, :seed_length]
        seed_audio = seed_audio[:, :seed_length] if seed_audio.dim() > 1 else seed_audio[:seed_length].unsqueeze(0)
    
    n_steps = int(config['generate_seconds'] * config['sr'])
    
    print(f"🎯 Generation parameters:")
    print(f"   Seed length: {seed_tokens.size(1):,} tokens ({config['seed_seconds']:.1f}s)")
    print(f"   Generate length: {n_steps:,} tokens ({config['generate_seconds']:.1f}s)")
    print(f"   Temperature: {config['temperature']}")
    print(f"   Top-k: {config['top_k']}")
    print(f"   Mode: {args.mode}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate audio samples
    generate_audio_samples(model, seed_tokens, seed_audio, n_steps, args, config, output_dir)
    
    print(f"\n🎉 Generation completed!")
    print(f"💾 Output saved to: {output_dir}")


def evaluate_command(args):
    """Handle evaluation command."""
    from inference import test_generation_quality
    
    # Setup PyTorch and device
    setup_torch()
    device = get_device()
    print(f"🎯 Using device: {device}")
    
    # Get configuration
    config = get_config()
    checkpoint_dir = args.checkpoint_dir or config['checkpoint_dir']
    
    # Load model
    model = load_model_for_inference(checkpoint_dir, None, config, device)
    
    # Run requested tests
    if args.test_all or args.test_generation:
        test_generation_quality(model, None, config, device)
    
    if args.test_all or args.test_consistency:
        # Load dataset and test consistency
        dataset = SegmentedTokensOnDisk(config['manifest_file'], cache_shards=False)
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)
        test_model_output_consistency(model, test_loader, device)
    
    print("✅ Evaluation completed!")


# Helper functions
def update_training_config_from_args(config, args):
    """Update configuration with training command line arguments."""
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


def update_inference_config_from_args(config, args):
    """Update configuration with inference command line arguments."""
    if args.seed_seconds is not None:
        config['seed_seconds'] = args.seed_seconds
    
    if args.generate_seconds is not None:
        config['generate_seconds'] = args.generate_seconds
    
    if args.temperature is not None:
        config['temperature'] = args.temperature
    
    if args.top_k is not None:
        config['top_k'] = args.top_k
    
    return config


def setup_datasets_for_training(config, args):
    """Create training and validation datasets."""
    print("📁 Setting up datasets...")
    
    if config['use_fake_dataset']:
        print("🔬 Using fake dataset for debugging/overfitting tests")
        train_dataset, test_dataset, train_loader, test_loader = create_fake_datasets(config)
        config['batch_size'] = 1
        config['num_workers'] = 0
    else:
        print("📊 Using full dataset for training")
        use_existing = not args.rebuild_dataset
        train_dataset, test_dataset, train_loader, test_loader = create_datasets(
            config, use_existing_tokens=use_existing
        )
    
    return train_dataset, test_dataset, train_loader, test_loader


def setup_model_and_optimizer(config, device):
    """Initialize model, optimizer, and scheduler."""
    print("🚀 Initializing model...")
    
    base_model = Wavenet(config).to(device)
    
    total_params = sum(p.numel() for p in base_model.parameters())
    trainable_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
    print(f"📊 Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    print(f"📏 Receptive field: {base_model.get_receptive_field():,} time steps")
    
    learning_rate = config['learning_rate']
    optimizer = optim.AdamW(base_model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=1, threshold=1e-3
    )
    
    return base_model, optimizer, scheduler


def load_model_for_inference(checkpoint_dir, checkpoint_path, config, device):
    """Load trained WaveNet model for inference."""
    print("🔄 Loading trained model...")
    
    base_model = Wavenet(config).to(device)
    
    if checkpoint_path:
        print(f"📁 Loading specific checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        base_model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('trained_till_epoch_index', 0)
        print(f"✅ Model loaded from epoch {epoch + 1}")
    else:
        eval_model = EvalModel(config, checkpoint_dir, base_model, device)
        base_model = eval_model.base_model
    
    base_model.eval()
    compiled_model = torch.compile(base_model, mode="reduce-overhead")
    
    total_params = sum(p.numel() for p in base_model.parameters())
    print(f"📊 Model parameters: {total_params:,}")
    print(f"📏 Receptive field: {base_model.get_receptive_field():,} time steps")
    
    return compiled_model


def prepare_seed_audio(args, audio_processor, config):
    """Prepare seed audio from file or dataset."""
    if args.seed_file:
        print(f"🎵 Loading seed audio from: {args.seed_file}")
        audio, sr = torchaudio.load(args.seed_file)
        audio = audio_processor.resample_audio(audio, sr, config['sr'])
        audio = audio_processor.trim_silence(audio)
        audio = audio_processor.normalize(audio)
        tokens = codec.mu_law_encode(audio.squeeze()).unsqueeze(0)
        print(f"🎯 Processed audio: {audio.shape}, tokens: {tokens.shape}")
        return tokens, audio
    
    elif args.seed_dataset_index is not None:
        print(f"📚 Loading seed from dataset index: {args.seed_dataset_index}")
        dataset = SegmentedTokensOnDisk(config['manifest_file'], cache_shards=False)
        if args.seed_dataset_index >= len(dataset):
            raise ValueError(f"Dataset index {args.seed_dataset_index} out of range (dataset size: {len(dataset)})")
        tokens, _ = dataset[args.seed_dataset_index]
        tokens = tokens.unsqueeze(0)
        audio = codec.mu_law_decode(tokens).squeeze(0)
        print(f"🎯 Dataset sample: tokens {tokens.shape}, audio {audio.shape}")
        return tokens, audio
    
    else:
        print("🎲 Using default seed from dataset (index 0)")
        dataset = SegmentedTokensOnDisk(config['manifest_file'], cache_shards=False)
        tokens, _ = dataset[0]
        tokens = tokens.unsqueeze(0)
        audio = codec.mu_law_decode(tokens).squeeze(0)
        return tokens, audio


def generate_audio_samples(model, seed_tokens, seed_audio, n_steps, args, config, output_dir):
    """Generate audio samples using the trained model."""
    from inference import generate_continuation, generate_greedy, save_audio_and_plot
    
    for i in range(args.batch_generate):
        print(f"\n🚀 Generating audio sample {i+1}/{args.batch_generate}...")
        
        if args.mode == 'greedy':
            generated_tokens = generate_greedy(model, seed_tokens, n_steps, config['device'])
        elif args.mode == 'nucleus':
            generated_tokens = generate_continuation(
                model, seed_tokens, n_steps,
                temperature=config['temperature'],
                top_p=args.nucleus_p,
                device=config['device'],
                use_fp32=config.get('use_fp32', True)
            )
        else:  # sample mode
            generated_tokens = generate_continuation(
                model, seed_tokens, n_steps,
                temperature=config['temperature'],
                top_k=config['top_k'],
                device=config['device'],
                use_fp32=config.get('use_fp32', True)
            )
        
        # Convert to audio
        generated_audio = codec.mu_law_decode(generated_tokens).squeeze(0)
        
        # Create output filename
        if args.output_name:
            base_name = args.output_name
        else:
            base_name = f"generated_{args.mode}_temp{config['temperature']}"
            if config.get('top_k'):
                base_name += f"_k{config['top_k']}"
        
        if args.batch_generate > 1:
            base_name += f"_sample{i+1}"
        
        output_path = output_dir / f"{base_name}.wav"
        
        # Save audio and plot
        save_audio_and_plot(
            generated_audio, config['sr'], output_path, args.save_plot
        )
        
        # Save seed audio for reference (only for first sample)
        if i == 0:
            seed_path = output_dir / f"{base_name}_seed.wav"
            save_audio_and_plot(
                seed_audio, config['sr'], seed_path, args.save_plot
            )
        
        print(f"✅ Sample {i+1} generated successfully!")


def run_sanity_checks(train_loader, test_loader, model, device):
    """Run various sanity checks on data and model."""
    print("\n🔍 Running sanity checks...")
    
    print("\n1. Checking data batch integrity...")
    for i, batch in enumerate(train_loader):
        sanity_check_batch(batch)
        if i >= 2:
            break
    
    print("\n2. Testing model output consistency...")
    test_model_output_consistency(model, test_loader, device)
    
    print("✅ All sanity checks passed!")


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    print(f"🚀 WaveNet - {args.command.upper()} mode")
    print("="*60)
    
    if args.command == 'prepare-dataset':
        prepare_dataset_command(args)
    elif args.command == 'train':
        train_command(args)
    elif args.command == 'inference':
        inference_command(args)
    elif args.command == 'evaluate':
        evaluate_command(args)
    else:
        print(f"❌ Unknown command: {args.command}")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
