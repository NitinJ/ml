"""
WaveNet model implementation.

Contains the core WaveNet architecture with causal dilated convolutions,
residual blocks, and skip connections for autoregressive audio generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import os
from config import get_model_config, get_audio_config


class MuLawEncoding:
    """
    Mu-law encoding and decoding for audio signal compression and quantization.
    
    Mu-law encoding compresses the dynamic range of audio signals by applying
    logarithmic quantization, which provides better perceptual quality for speech
    and audio. This is commonly used in telephony and audio codecs.
    
    Args:
        quantization_channels (int): Number of discrete quantization levels.
                                   Default is 256 (8-bit quantization).
    
    Attributes:
        Q (int): Number of quantization channels
        enc (torchaudio.transforms.MuLawEncoding): Encoder transform
        dec (torchaudio.transforms.MuLawDecoding): Decoder transform
    """
    def __init__(self, quantization_channels: int = 256):
        self.Q = quantization_channels
        self.enc = torchaudio.transforms.MuLawEncoding(self.Q)
        self.dec = torchaudio.transforms.MuLawDecoding(self.Q)

    @torch.no_grad()
    def mu_law_encode(self, x: torch.Tensor) -> torch.LongTensor:
        """
        Encode continuous audio waveform to discrete mu-law tokens.
        
        Converts continuous audio samples in [-1, 1] range to discrete integers
        in [0, Q-1] range using mu-law companding.
        
        Args:
            x (torch.Tensor): Input audio waveform, shape [..., T], 
                            dtype float, values in [-1, 1]
        
        Returns:
            torch.LongTensor: Encoded tokens, shape [..., T], 
                            dtype long, values in [0, Q-1]
        
        Example:
            >>> codec = MuLawEncoding(256)
            >>> audio = torch.randn(1, 16000)  # 1 second at 16kHz
            >>> tokens = codec.mu_law_encode(audio)  # shape: [1, 16000]
        """
        return self.enc(x)

    @torch.no_grad()
    def mu_law_decode(self, q: torch.Tensor) -> torch.Tensor:
        """
        Decode discrete mu-law tokens back to continuous audio waveform.
        
        Converts discrete tokens in [0, Q-1] range back to continuous audio
        samples in [-1, 1] range using inverse mu-law expansion.
        
        Args:
            q (torch.Tensor): Encoded tokens, shape [..., T], 
                            dtype long/int, values in [0, Q-1]
        
        Returns:
            torch.Tensor: Decoded audio waveform, shape [..., T], 
                        dtype float, values in [-1, 1]
        
        Example:
            >>> codec = MuLawEncoding(256)
            >>> tokens = torch.randint(0, 256, (1, 16000))
            >>> audio = codec.mu_law_decode(tokens)  # shape: [1, 16000]
        """
        return self.dec(q)


class CausalDilatedConvolution(nn.Module):
    """
    Causal dilated 1D convolution for autoregressive sequence modeling.
    
    Implements dilated convolution with causal padding to ensure that the output
    at time t only depends on inputs at times <= t. This preserves the autoregressive
    property needed for WaveNet's generative modeling.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Size of the convolving kernel. Default: 2
        dilation (int): Dilation factor for dilated convolution. Default: 1
        
    Attributes:
        kernel_size (int): Stored kernel size
        dilation (int): Stored dilation factor
        conv1d (nn.Conv1d): The underlying convolution layer
    """
    def __init__(self, in_channels, out_channels, kernel_size=2, dilation=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.conv1d = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,  # No automatic padding - we handle causality manually
            bias=True)

    def forward(self, x):
        """
        Apply causal dilated convolution to input sequence.
        
        Args:
            x (torch.Tensor): Input tensor, shape [batch_size, in_channels, time]
            
        Returns:
            torch.Tensor: Output tensor, shape [batch_size, out_channels, time]
                        Same temporal length as input due to causal padding
                        
        Note:
            Causal padding of size (kernel_size - 1) * dilation is applied to the left
            (past) side only, ensuring future information doesn't leak into past predictions.
        """
        if self.kernel_size > 1:
            # Apply causal padding: pad left side only to prevent future leakage
            pad_left = (self.kernel_size - 1) * self.dilation
            x = F.pad(x, (pad_left, 0), mode='constant', value=0)
        return self.conv1d(x)


class ResidualBlock(nn.Module):
    """
    WaveNet residual block with gated activation and skip connections.
    
    Core building block of WaveNet architecture. Applies dilated causal convolution
    followed by gated activation (tanh * sigmoid), then projects to residual and 
    skip connection outputs. The residual path enables deep networks while skip
    connections aggregate features across all layers.
    
    Args:
        C_res (int): Number of residual channels (internal feature dimension)
        C_skip (int): Number of skip connection channels
        dilation (int): Dilation factor for the causal convolution. Default: 1
        
    Attributes:
        dilated_conv (CausalDilatedConvolution): Causal dilated convolution layer
        skip_conv1x1 (nn.Conv1d): 1x1 conv for skip connection projection  
        res_conv1x1 (nn.Conv1d): 1x1 conv for residual connection projection
    """
    def __init__(self, C_res, C_skip, dilation=1):
        super().__init__()
        # Dilated conv outputs 2*C_res channels for gated activation
        self.dilated_conv = CausalDilatedConvolution(C_res, 2 * C_res, kernel_size=2, dilation=dilation)
        self.skip_conv1x1 = nn.Conv1d(C_res, C_skip, 1)
        self.res_conv1x1 = nn.Conv1d(C_res, C_res, 1)
        
    def forward(self, x):
        """
        Forward pass through residual block.
        
        Args:
            x (torch.Tensor): Input tensor, shape [batch_size, C_res, time]
            
        Returns:
            tuple: (residual_out, skip_out) where:
                - residual_out: torch.Tensor, shape [batch_size, C_res, time]
                              Residual output for next layer (x + processed_x)
                - skip_out: torch.Tensor, shape [batch_size, C_skip, time]
                          Skip connection output for final aggregation
                          
        Note:
            The gated activation is computed as: tanh(filter) * sigmoid(gate)
            where filter and gate are the two halves of the dilated conv output.
            This allows the network to learn what information to pass through.
        """
        # Apply dilated causal convolution
        output = self.dilated_conv(x)  # [B, 2*C_res, T]
        
        # Split into filter and gate components for gated activation
        filter_out, gate_out = torch.chunk(output, 2, dim=1)  # Each: [B, C_res, T]

        # Apply gated activation unit: tanh(filter) ⊙ sigmoid(gate)
        gated = torch.tanh(filter_out) * torch.sigmoid(gate_out)  # [B, C_res, T]

        # Project to residual and skip connection outputs via 1x1 convolutions
        residual = self.res_conv1x1(gated)  # [B, C_res, T]
        skip = self.skip_conv1x1(gated)     # [B, C_skip, T]

        # Return residual connection (input + processed) and skip output
        return residual + x, skip


class Wavenet(nn.Module):
    """
    WaveNet: A Generative Model for Raw Audio.
    
    Implements the WaveNet architecture for autoregressive audio generation.
    The model uses stacked dilated causal convolutions with residual and skip
    connections to efficiently model long-range dependencies in audio sequences.
    
    Architecture:
    - Input embedding layer: maps discrete tokens to continuous features
    - Stacked residual blocks: each block contains multiple layers with increasing dilations
    - Skip connections: aggregate features from all layers
    - Output head: projects skip features to output logits
    
    Args:
        config (dict): Configuration dictionary containing:
            - residual_channels (int): Number of channels in residual paths
            - skip_channels (int): Number of channels in skip connections  
            - output_dim (int): Vocabulary size (typically 256 for mu-law)
            - n_layers (int): Number of layers per residual block
            - n_blocks (int): Number of residual blocks
            - kernel_size (int): Convolution kernel size (typically 2)
            
    Attributes:
        config (dict): Stored configuration
        C_res (int): Residual channel dimension
        C_output (int): Output vocabulary size
        C_skip (int): Skip connection channel dimension
        kernel_size (int): Convolution kernel size
        _rf (int): Computed receptive field size
        embedding (nn.Embedding): Token embedding layer
        conv1d (nn.Conv1d): Initial feature processing layer
        residual_blocks (nn.ModuleList): Stack of residual blocks
        output_head (nn.Sequential): Final output projection layers
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.C_res = config['residual_channels']
        self.C_output = config['output_dim']
        self.C_skip = config['skip_channels']
        self.kernel_size = config['kernel_size']
        self._rf = self._compute_receptive_field()

        # Token embedding: converts discrete tokens [0, 255] to continuous features
        self.embedding = nn.Embedding(
            self.C_output, self.C_res, dtype=torch.float32)

        # Initial 1x1 convolution for feature processing and stability
        self.conv1d = nn.Conv1d(self.C_res, self.C_res, kernel_size=1)

        # Stack of residual blocks with exponentially increasing dilations
        # Each block contains n_layers with dilations: 1, 2, 4, 8, ..., 2^(n_layers-1)
        self.residual_blocks = nn.ModuleList([
            self._create_residual_block() for i in range(config['n_blocks'])
        ])
        
        # Output head: processes aggregated skip connections to produce logits
        self.output_head = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(self.C_skip, self.C_skip, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(self.C_skip, self.C_output, kernel_size=1))

    def _create_residual_block(self):
        """
        Create a single residual block with exponentially increasing dilations.
        
        Each block contains n_layers ResidualBlock modules with dilation factors
        of 1, 2, 4, 8, ..., 2^(n_layers-1). This exponential growth allows the
        model to efficiently capture both short and long-range dependencies.
        
        Returns:
            nn.ModuleList: List of ResidualBlock modules with increasing dilations
            
        Example:
            For n_layers=5: dilations = [1, 2, 4, 8, 16]
            For n_layers=10: dilations = [1, 2, 4, ..., 512]
        """
        return nn.ModuleList([
            ResidualBlock(self.C_res, self.C_skip, dilation=2**i)
            for i in range(self.config['n_layers'])
        ])

    def forward(self, x):
        """
        Forward pass through WaveNet model.
        
        Args:
            x (torch.Tensor): Input token sequence, shape [batch_size, time],
                            dtype long, values in [0, output_dim-1]
        
        Returns:
            torch.Tensor: Output logits, shape [batch_size, time, output_dim],
                        dtype float. These can be used with cross-entropy loss
                        or softmax for sampling.
                        
        Processing flow:
            1. Embed tokens to continuous features: [B, T] -> [B, T, C_res]
            2. Transpose for convolution: [B, T, C_res] -> [B, C_res, T]
            3. Apply initial 1x1 convolution
            4. Process through residual blocks, accumulating skip connections
            5. Apply output head to aggregated skip connections
            6. Transpose back to sequence format: [B, C_output, T] -> [B, T, C_output]
        """
        # Step 1: Embed discrete tokens to continuous features
        x_embd = self.embedding(x)  # [B, T] -> [B, T, C_res]
        
        # Step 2: Transpose for convolution operations (PyTorch conv1d expects [B, C, T])
        x_embd = x_embd.permute(0, 2, 1)  # [B, T, C_res] -> [B, C_res, T]
        skip_output = None

        # Step 3: Apply initial feature processing
        x_embd = self.conv1d(x_embd)  # [B, C_res, T]

        # Step 4: Process through all residual blocks
        for residual_block in self.residual_blocks:
            for layer in residual_block:
                x_embd, x_skip = layer(x_embd)
                # Accumulate skip connections from all layers
                skip_output = x_skip if skip_output is None else skip_output + x_skip

        # Step 5: Generate output logits from aggregated skip connections
        output = self.output_head(skip_output)  # [B, C_skip, T] -> [B, C_output, T]

        # Step 6: Transpose back to sequence format for loss computation
        return output.permute(0, 2, 1)  # [B, C_output, T] -> [B, T, C_output]

    def _compute_receptive_field(self):
        """
        Compute the theoretical receptive field of the WaveNet model.
        
        The receptive field determines how many past time steps the model
        can observe when predicting the next token. For WaveNet with
        exponentially increasing dilations, this grows exponentially with depth.
        
        Returns:
            int: Receptive field size in time steps
            
        Formula:
            For kernel_size=2: RF = 1 + n_blocks * (2^n_layers - 1)
            General case: RF = 1 + n_blocks * sum_{i=0}^{n_layers-1} (k-1)*2^i
            
        Example:
            n_blocks=5, n_layers=10, kernel_size=2:
            RF = 1 + 5 * (2^10 - 1) = 1 + 5 * 1023 = 5116 time steps
        """
        if self.kernel_size != 2:
            # General formula for arbitrary kernel size
            dilation_sum = (self.kernel_size - 1) * (2**self.config['n_layers'] - 1)
            return 1 + self.config['n_blocks'] * dilation_sum
        # Optimized formula for kernel_size=2
        return 1 + self.config['n_blocks'] * (2**self.config['n_layers'] - 1)

    def get_receptive_field(self):
        """
        Get the receptive field size of the model.
        
        Returns:
            int: Number of past time steps the model can observe
        """
        return self._rf


class Model:
    """
    Base class for WaveNet model wrappers with checkpoint management and compilation.
    
    Provides common functionality for model checkpoint handling, path management,
    and PyTorch compilation for both training and evaluation modes.
    
    Args:
        config (dict): Model configuration dictionary
        base_model (Wavenet): The underlying WaveNet model
        checkpoint_dir (str): Directory for saving/loading checkpoints
        
    Attributes:
        config (dict): Model configuration
        base_model (Wavenet): The underlying WaveNet model
        model_train (torch.fx.GraphModule): Compiled model for training (None initially)
        model_eval (torch.fx.GraphModule): Compiled model for evaluation (None initially)
        checkpoint_dir (str): Directory for checkpoints
    """
    def __init__(self, config, base_model, checkpoint_dir):
        self.config = config
        self.base_model = base_model
        self.model_train = None
        self.model_eval = None
        self.checkpoint_dir = checkpoint_dir

    def _get_checkpoint_paths(self):
        """
        Get paths to latest and best model checkpoints if they exist.
        
        Returns:
            tuple: (latest_path, best_path) where each is either a valid
                   file path string or None if the checkpoint doesn't exist
        """
        latest_path = self._get_latest_checkpoint_path()
        best_path = self._get_best_checkpoint_path()
        latest_path = latest_path if os.path.exists(latest_path) else None
        best_path = best_path if os.path.exists(best_path) else None
        return latest_path, best_path

    def _get_best_checkpoint_path(self):
        """
        Get path to the best model checkpoint (lowest validation loss).
        
        Returns:
            str: Path to best model checkpoint file
        """
        return os.path.join(self.checkpoint_dir, 'best_model.pth')

    def _get_latest_checkpoint_path(self):
        """
        Get path to the latest model checkpoint (most recent training state).
        
        Returns:
            str: Path to latest checkpoint file
        """
        return os.path.join(self.checkpoint_dir, 'latest_checkpoint.pth')

    def _compile_for_training(self):
        """
        Compile the model for optimized training performance.
        
        Sets model to training mode and applies PyTorch compilation with
        "reduce-overhead" mode for faster training.
        
        Returns:
            torch.fx.GraphModule: Compiled model optimized for training
        """
        self.base_model.train()
        return torch.compile(self.base_model, mode="reduce-overhead")

    def _compile_for_eval(self):
        """
        Compile the model for optimized evaluation/inference performance.
        
        Sets model to evaluation mode and applies PyTorch compilation with
        "reduce-overhead" mode for faster inference.
        
        Returns:
            torch.fx.GraphModule: Compiled model optimized for evaluation
        """
        self.base_model.eval()
        return torch.compile(self.base_model, mode="reduce-overhead")

    def get_receptive_field(self):
        """
        Get the receptive field size of the underlying model.
        
        Returns:
            int: Number of past time steps the model can observe
        """
        return self.base_model.get_receptive_field()


class TrainableModel(Model):
    """
    Trainable WaveNet model wrapper with training state management and checkpointing.
    
    Extends the base Model class to handle training-specific functionality including
    optimizer state, learning rate scheduling, training metrics tracking, and
    checkpoint loading/saving for resumable training.
    
    Args:
        config (dict): Model configuration dictionary
        checkpoint_dir (str): Directory for saving/loading checkpoints
        base_model (Wavenet): The underlying WaveNet model
        optimizer (torch.optim.Optimizer): Optimizer for training
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler
        load_from_checkpoint (bool): Whether to load from existing checkpoint. Default: False
        device (torch.device): Device for computation
        
    Attributes:
        base_model (Wavenet): The underlying WaveNet model
        optimizer (torch.optim.Optimizer): Training optimizer
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler
        training_stats (dict): Training metrics history containing:
            - train_losses (list): Training loss per epoch
            - val_losses (list): Validation loss per epoch  
            - train_accuracies (list): Training accuracy per epoch
            - val_accuracies (list): Validation accuracy per epoch
            - best_val_loss (float): Best validation loss achieved
        trained_till_epoch_index (int): Last completed epoch (-1 if untrained)
        compiled_model (torch.fx.GraphModule): Compiled model for training
    """
    def __init__(self, config, checkpoint_dir, base_model, optimizer, scheduler, load_from_checkpoint=False, device=None):
        super().__init__(config, base_model, checkpoint_dir)

        self.base_model = base_model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.training_stats = {
            'train_losses': [],
            'val_losses': [],
            'train_accuracies': [],
            'val_accuracies': [],
            'best_val_loss': float('inf'),
        }
        # Track training progress: -1 = untrained, 0+ = last completed epoch index
        self.trained_till_epoch_index = -1
        if load_from_checkpoint:
            self._load_from_checkpoint()

        # Compile model specifically for training optimization
        self.compiled_model = self._compile_for_training()

    def save(self, epoch, training_stats, learning_rate, is_best=False):
        """
        Save complete training checkpoint with model and optimizer state.
        
        Saves all necessary information to resume training including model weights,
        optimizer state, scheduler state, training metrics, and configuration.
        
        Args:
            epoch (int): Current epoch number (0-indexed)
            training_stats (dict): Dictionary containing training metrics:
                - train_losses, val_losses, train_accuracies, val_accuracies, best_val_loss
            learning_rate (float): Current learning rate
            is_best (bool): Whether this is the best model so far (lowest val loss)
            
        Saves:
            - Latest checkpoint: Always saved for training resumption
            - Best checkpoint: Only saved when is_best=True for evaluation
        """
        checkpoint = {
            'config': self.config,
            'model_state_dict': self.base_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),

            'train_losses': training_stats['train_losses'],
            'val_losses': training_stats['val_losses'],
            'train_accuracies': training_stats['train_accuracies'],
            'val_accuracies': training_stats['val_accuracies'],
            'best_val_loss': training_stats['best_val_loss'],

            'trained_till_epoch_index': epoch,
            'learning_rate': learning_rate
        }

        # Always save latest checkpoint for resumable training
        torch.save(checkpoint, self._get_latest_checkpoint_path())

        # Save best model checkpoint for evaluation/inference
        if is_best:
            torch.save(checkpoint, self._get_best_checkpoint_path())
            print(f"💾 New best model saved! Val Loss: {
                  training_stats['best_val_loss']:.4f}")

    def _load_from_checkpoint(self):
        """
        Load training state from the latest checkpoint for resumable training.
        
        Restores model weights, optimizer state, scheduler state, training metrics,
        and epoch counter from the most recent checkpoint. If no checkpoint exists,
        training will start from scratch.
        
        Updates:
            - base_model: Loads saved model weights
            - optimizer: Restores optimizer state (momentum, learning rates, etc.)
            - scheduler: Restores scheduler state  
            - training_stats: Loads training history metrics
            - trained_till_epoch_index: Sets last completed epoch
        """
        latest_checkpoint_path, _ = self._get_checkpoint_paths()

        if not latest_checkpoint_path:
            print("🆕 No existing checkpoints found. Starting training from scratch.")
            return

        print(f"🔄 Loading training checkpoint from: {latest_checkpoint_path}")
        checkpoint = torch.load(latest_checkpoint_path, map_location=self.device)

        # Restore model weights
        self.base_model.load_state_dict(checkpoint['model_state_dict'])
        
        # Restore optimizer state (important for momentum, learning rates, etc.)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Restore scheduler state if available
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        # Restore training progress
        self.trained_till_epoch_index = checkpoint.get('trained_till_epoch_index', -1)

        # Restore training metrics history
        self.training_stats = {
            'train_losses': checkpoint.get('train_losses', []),
            'val_losses': checkpoint.get('val_losses', []),
            'train_accuracies': checkpoint.get('train_accuracies', []),
            'val_accuracies': checkpoint.get('val_accuracies', []),
            'best_val_loss': checkpoint.get('best_val_loss', float('inf')),
        }
        print(f"✅ Training checkpoint loaded successfully. Trained for {
              self.trained_till_epoch_index + 1} epochs!")


class EvalModel(Model):
    """
    Evaluation-only WaveNet model wrapper for inference and generation.
    
    Loads the best saved model checkpoint (lowest validation loss) and compiles
    it for optimized evaluation performance. Used for inference, generation,
    and model evaluation after training.
    
    Args:
        config (dict): Model configuration dictionary
        checkpoint_dir (str): Directory containing saved checkpoints
        base_model (Wavenet): The underlying WaveNet model
        device (torch.device): Device for computation
        
    Attributes:
        base_model (Wavenet): The underlying WaveNet model
        compiled_model (torch.fx.GraphModule): Compiled model optimized for evaluation
    """
    def __init__(self, config, checkpoint_dir, base_model, device=None):
        super().__init__(config, base_model, checkpoint_dir)
        self.base_model = base_model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_from_checkpoint()

        # Compile model specifically for evaluation/inference optimization
        self.compiled_model = self._compile_for_eval()

    def _load_from_checkpoint(self):
        """
        Load the best model checkpoint for evaluation.
        
        Loads model weights from the best checkpoint (lowest validation loss)
        rather than the latest checkpoint. This ensures optimal performance
        for inference and generation tasks.
        
        Updates:
            - base_model: Loads best model weights for evaluation
        """
        _, best_checkpoint_path = self._get_checkpoint_paths()

        if not best_checkpoint_path:
            print("🆕 No existing best checkpoint found for evaluation")
            return

        print(f"🔄 Loading eval model checkpoint from: {best_checkpoint_path}")
        checkpoint = torch.load(best_checkpoint_path, map_location=self.device)
        
        # Load only model weights (no optimizer state needed for evaluation)
        self.base_model.load_state_dict(checkpoint['model_state_dict'])

        epoch = checkpoint.get('trained_till_epoch_index', 0)
        print(f"✅ Eval model checkpoint loaded successfully. Trained till epoch {
              epoch + 1}!")


# Create codec instance for global use
codec = MuLawEncoding(256)
