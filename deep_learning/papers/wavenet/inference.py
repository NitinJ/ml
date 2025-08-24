#!/usr/bin/env python3
"""
Inference entry point for WaveNet model.

Command-line interface for generating audio using trained WaveNet models
with various sampling strategies and quality/diversity trade-offs.
"""

import argparse
import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm.auto import tqdm
import warnings

from config import get_config, get_device, setup_torch, get_generation_config
from model import Wavenet, EvalModel, codec
from dataset import SegmentedTokensOnDisk, AudioProcessor

warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")


def parse_args():
    """Parse command line arguments for inference configuration."""
    parser = argparse.ArgumentParser(description='Generate audio using trained WaveNet model')
    
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
    
    # Quality parameters
    parser.add_argument('--sample-rate', type=int, default=None,
                        help='Sample rate for generated audio (default: from config)')
    parser.add_argument('--use-fp32', action='store_true', default=True,
                        help='Use FP32 for generation (recommended for quality)')
    parser.add_argument('--no-fp32', dest='use_fp32', action='store_false',
                        help='Use mixed precision for generation (faster but may affect quality)')
    
    # Testing parameters
    parser.add_argument('--test-generation', action='store_true',
                        help='Run generation quality tests')
    parser.add_argument('--batch-generate', type=int, default=1,
                        help='Generate multiple samples (useful for diversity testing)')
    
    return parser.parse_args()


def update_config_from_args(config, args):
    """Update configuration with command line arguments."""
    generation_config = get_generation_config()
    
    # Generation parameters
    if args.seed_seconds is not None:
        generation_config['seed_seconds'] = args.seed_seconds
    
    if args.generate_seconds is not None:
        generation_config['generate_seconds'] = args.generate_seconds
    
    if args.temperature is not None:
        generation_config['temperature'] = args.temperature
    
    if args.top_k is not None:
        generation_config['top_k'] = args.top_k
    
    if args.sample_rate is not None:
        config['sr'] = args.sample_rate
    
    generation_config['use_fp32'] = args.use_fp32
    
    # Update main config
    config.update(generation_config)
    
    return config


def load_model(checkpoint_dir, checkpoint_path, config, device):
    """Load trained WaveNet model for inference."""
    print("🔄 Loading trained model...")
    
    # Create base model
    base_model = Wavenet(config).to(device)
    
    if checkpoint_path:
        print(f"📁 Loading specific checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        base_model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('trained_till_epoch_index', 0)
        print(f"✅ Model loaded from epoch {epoch + 1}")
    else:
        # Use EvalModel to load best checkpoint
        eval_model = EvalModel(config, checkpoint_dir, base_model, device)
        base_model = eval_model.base_model
    
    # Compile for inference
    base_model.eval()
    compiled_model = torch.compile(base_model, mode="reduce-overhead")
    
    total_params = sum(p.numel() for p in base_model.parameters())
    print(f"📊 Model parameters: {total_params:,}")
    print(f"📏 Receptive field: {base_model.get_receptive_field():,} time steps")
    
    return base_model, compiled_model


def load_seed_audio(seed_file, audio_processor, config):
    """Load and process seed audio from file."""
    print(f"🎵 Loading seed audio from: {seed_file}")
    
    # Load audio file
    audio, sr = torchaudio.load(seed_file)
    print(f"📊 Original audio: {audio.shape}, sample rate: {sr}")
    
    # Process audio
    audio = audio_processor.resample_audio(audio, sr, config['sr'])
    audio = audio_processor.trim_silence(audio)
    audio = audio_processor.normalize(audio)
    
    # Convert to tokens
    tokens = codec.mu_law_encode(audio.squeeze()).unsqueeze(0)
    
    print(f"🎯 Processed audio: {audio.shape}, tokens: {tokens.shape}")
    return tokens, audio


def load_seed_from_dataset(dataset_index, config):
    """Load seed audio from dataset."""
    print(f"📚 Loading seed from dataset index: {dataset_index}")
    
    # Load dataset
    dataset = SegmentedTokensOnDisk(config['manifest_file'], cache_shards=False)
    
    if dataset_index >= len(dataset):
        raise ValueError(f"Dataset index {dataset_index} out of range (dataset size: {len(dataset)})")
    
    # Get sample
    tokens, _ = dataset[dataset_index]
    tokens = tokens.unsqueeze(0)  # Add batch dimension
    
    # Convert back to audio for reference
    audio = codec.mu_law_decode(tokens).squeeze(0)
    
    print(f"🎯 Dataset sample: tokens {tokens.shape}, audio {audio.shape}")
    return tokens, audio


@torch.no_grad()
def generate_continuation(
    model,
    seed_tokens: torch.LongTensor,
    n_steps: int,
    temperature: float = 1.0,
    top_k: int = None,
    top_p: float = None,
    device: str = "cuda",
    use_fp32: bool = True,
):
    """
    Generate audio continuation using autoregressive sampling from WaveNet.
    
    Args:
        model: WaveNet model
        seed_tokens: Initial sequence [1, seed_length]
        n_steps: Number of new tokens to generate
        temperature: Sampling temperature
        top_k: Top-k filtering
        top_p: Nucleus (top-p) sampling
        device: Computation device
        use_fp32: Whether to use FP32 for stability
        
    Returns:
        Generated sequence [1, seed_length + n_steps]
    """
    model.eval()
    seq = seed_tokens.to(device)
    B = seq.size(0)
    assert B == 1, "Use batch size 1 for generation"

    rf = model.get_receptive_field()

    for step in tqdm(range(n_steps), desc='Generating'):
        # Crop to receptive field
        ctx = seq[:, -rf:] if seq.size(1) > rf else seq

        if use_fp32:
            with torch.inference_mode():
                logits = model(ctx).float()
        else:
            with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(ctx)
                logits = logits.float()

        logits = logits[:, -1, :]  # [1, 256] - last timestep

        # Apply temperature
        if temperature != 1.0:
            logits = logits / max(temperature, 1e-6)

        # Apply top-k filtering
        if top_k is not None and top_k > 0:
            topk_vals, topk_idx = torch.topk(
                logits, k=min(top_k, logits.size(-1)), dim=-1)
            logits_filtered = torch.full_like(logits, float('-inf'))
            logits_filtered.scatter_(-1, topk_idx, topk_vals)
            logits = logits_filtered

        # Apply nucleus (top-p) sampling
        if top_p is not None and top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')

        # Sample next token
        probs = F.softmax(logits, dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1)

        # Append to sequence
        seq = torch.cat([seq, next_tok], dim=1)

        # Progress logging
        if step % 1000 == 0 and step > 0:
            prob_max = probs.max().item()
            entropy = -(probs * torch.log(probs + 1e-8)).sum().item()
            print(f"Step {step}: max_prob={prob_max:.3f}, entropy={entropy:.3f}, token={next_tok.item()}")

    return seq


@torch.no_grad()
def generate_greedy(model, seed_tokens, n_steps, device="cuda"):
    """Generate using greedy decoding (deterministic)."""
    model.eval()
    seq = seed_tokens.to(device)
    rf = model.get_receptive_field()
    
    for _ in tqdm(range(n_steps), desc='Generating (greedy)'):
        ctx = seq[:, -rf:] if seq.size(1) > rf else seq
        logits = model(ctx)[:, -1, :].float()
        next_tok = logits.argmax(dim=-1, keepdim=True)
        seq = torch.cat([seq, next_tok], dim=1)
    
    return seq


def save_audio_and_plot(audio_tensor, sample_rate, output_path, save_plot=False):
    """Save generated audio and optionally create waveform plot."""
    # Ensure audio is in correct format
    if audio_tensor.dim() == 1:
        audio_tensor = audio_tensor.unsqueeze(0)
    
    # Save audio file
    torchaudio.save(str(output_path), audio_tensor.cpu(), sample_rate)
    print(f"💾 Audio saved to: {output_path}")
    
    # Save plot if requested
    if save_plot:
        plot_path = output_path.with_suffix('.png')
        plt.figure(figsize=(14, 4))
        plt.plot(audio_tensor.squeeze().cpu().numpy())
        plt.title(f"Generated Audio Waveform")
        plt.xlabel("Sample")
        plt.ylabel("Amplitude")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"📊 Plot saved to: {plot_path}")


def test_generation_quality(model, dataset, config, device):
    """Run comprehensive generation quality tests."""
    print("\n🧪 Running generation quality tests...")
    
    # Test 1: Temperature sensitivity
    print("\n1. Testing temperature sensitivity...")
    dataset = SegmentedTokensOnDisk(config['manifest_file'], cache_shards=False)
    seed_tokens, _ = dataset[0]
    seed_tokens = seed_tokens[:8000].unsqueeze(0)  # Use first 8k tokens as seed
    
    temperatures = [0.1, 0.7, 1.0, 1.5]
    for temp in temperatures:
        print(f"   Temperature {temp}:")
        generated = generate_continuation(
            model, seed_tokens, n_steps=1000, temperature=temp, device=device
        )
        unique_tokens = generated[:, seed_tokens.size(1):].unique().numel()
        print(f"     Unique tokens in 1000 steps: {unique_tokens}")
    
    # Test 2: Top-k sensitivity
    print("\n2. Testing top-k sensitivity...")
    top_k_values = [10, 50, 100, None]
    for k in top_k_values:
        print(f"   Top-k {k if k else 'disabled'}:")
        generated = generate_continuation(
            model, seed_tokens, n_steps=1000, top_k=k, device=device
        )
        unique_tokens = generated[:, seed_tokens.size(1):].unique().numel()
        print(f"     Unique tokens in 1000 steps: {unique_tokens}")
    
    # Test 3: Mode collapse detection
    print("\n3. Testing for mode collapse...")
    generations = []
    for i in range(5):
        generated = generate_continuation(
            model, seed_tokens, n_steps=500, temperature=1.0, device=device
        )
        generations.append(generated[:, seed_tokens.size(1):])
    
    # Check diversity
    all_same = all(torch.equal(generations[0], gen) for gen in generations[1:])
    if all_same:
        print("   ⚠️  Mode collapse detected - all generations identical")
    else:
        print("   ✅ Generations show diversity")
    
    print("✅ Generation quality tests completed")


def main():
    """Main inference function."""
    args = parse_args()
    
    # Setup PyTorch and device
    setup_torch()
    device = get_device()
    print(f"🎯 Using device: {device}")
    
    # Get and update configuration
    config = get_config()
    config = update_config_from_args(config, args)
    
    # Determine checkpoint directory
    checkpoint_dir = args.checkpoint_dir or config['checkpoint_dir']
    
    # Load model
    model, compiled_model = load_model(checkpoint_dir, args.checkpoint_path, config, device)
    
    # Run tests if requested
    if args.test_generation:
        test_generation_quality(compiled_model, None, config, device)
        return
    
    # Prepare seed audio
    audio_processor = AudioProcessor()
    
    if args.seed_file:
        seed_tokens, seed_audio = load_seed_audio(args.seed_file, audio_processor, config)
    elif args.seed_dataset_index is not None:
        seed_tokens, seed_audio = load_seed_from_dataset(args.seed_dataset_index, config)
    else:
        # Use default dataset sample
        print("🎲 Using default seed from dataset (index 0)")
        seed_tokens, seed_audio = load_seed_from_dataset(0, config)
    
    # Determine seed length
    seed_length = int(config['seed_seconds'] * config['sr'])
    if seed_tokens.size(1) > seed_length:
        seed_tokens = seed_tokens[:, :seed_length]
        seed_audio = seed_audio[:, :seed_length] if seed_audio.dim() > 1 else seed_audio[:seed_length].unsqueeze(0)
    
    # Calculate generation parameters
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
    
    # Generate audio
    for i in range(args.batch_generate):
        print(f"\n🚀 Generating audio sample {i+1}/{args.batch_generate}...")
        
        if args.mode == 'greedy':
            generated_tokens = generate_greedy(compiled_model, seed_tokens, n_steps, device)
        elif args.mode == 'nucleus':
            generated_tokens = generate_continuation(
                compiled_model, seed_tokens, n_steps,
                temperature=config['temperature'],
                top_p=args.nucleus_p,
                device=device,
                use_fp32=config['use_fp32']
            )
        else:  # sample mode
            generated_tokens = generate_continuation(
                compiled_model, seed_tokens, n_steps,
                temperature=config['temperature'],
                top_k=config['top_k'],
                device=device,
                use_fp32=config['use_fp32']
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
        print(f"   Total duration: {generated_audio.size(-1) / config['sr']:.1f}s")
        print(f"   Seed duration: {seed_tokens.size(1) / config['sr']:.1f}s")
        print(f"   Generated duration: {n_steps / config['sr']:.1f}s")
    
    print(f"\n🎉 Generation completed!")
    print(f"💾 Output saved to: {output_dir}")


if __name__ == "__main__":
    main()
