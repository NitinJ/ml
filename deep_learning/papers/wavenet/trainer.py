"""
Trainer module for WaveNet model.

Contains comprehensive training functionality including training loops,
validation, checkpointing, progress visualization, and model evaluation.
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import json
from tqdm.auto import tqdm

from config import get_training_config, get_checkpoint_config


class Trainer:
    """
    Comprehensive trainer for WaveNet autoregressive audio modeling.
    
    Handles the complete training pipeline including forward passes, loss computation,
    validation, checkpointing, progress tracking, and visualization. Supports
    mixed precision training, gradient clipping, and learning rate scheduling.
    
    Args:
        base_model (Wavenet): The underlying WaveNet model
        trainable_model (TrainableModel): Wrapper with training state management
        config (dict): Training configuration dictionary
        learning_rate (float): Initial learning rate
        optimizer (torch.optim.Optimizer): Optimizer for training
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler
        checkpoint_dir (str): Directory for saving checkpoints and plots
        train_loader (torch.utils.data.DataLoader): Training data loader
        val_loader (torch.utils.data.DataLoader): Validation data loader
        device (torch.device): Device for training (cuda/cpu)
        
    Attributes:
        device (torch.device): Training device
        base_model (Wavenet): Base WaveNet model
        trainable_model (TrainableModel): Training wrapper
        compiled_model (torch.fx.GraphModule): Compiled model for training
        learning_rate (float): Learning rate
        optimizer (torch.optim.Optimizer): Training optimizer
        scheduler (torch.optim.lr_scheduler._LRScheduler): LR scheduler
        checkpoint_dir (str): Checkpoint directory
        train_loader (torch.utils.data.DataLoader): Training data
        val_loader (torch.utils.data.DataLoader): Validation data
        training_stats (dict): Training metrics history
        start_epoch (int): Epoch to start/resume training from
        scaler (torch.amp.GradScaler): Gradient scaler for mixed precision
    """
    def __init__(self,
                 base_model,
                 trainable_model,
                 config,
                 learning_rate,
                 optimizer,
                 scheduler,
                 checkpoint_dir,
                 train_loader,
                 val_loader,
                 device):
        self.device = device
        self.base_model = base_model
        self.trainable_model = trainable_model
        self.compiled_model = self.trainable_model.compiled_model

        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.checkpoint_dir = checkpoint_dir
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Reference to training metrics (shared with trainable_model)
        self.training_stats = self.trainable_model.training_stats
        # Determine starting epoch (resume from checkpoint or start fresh)
        self.start_epoch = self.trainable_model.trained_till_epoch_index + 1
        
        # Mixed precision training
        self.scaler = torch.amp.GradScaler('cuda', enabled=True)
        
        # Training config
        self.training_config = get_training_config()

    def prepare_batch(self, batch, device):
        """
        Move batch tensors to the specified device for training.
        
        Args:
            batch (tuple): Batch from DataLoader containing (audio_x, audio_y)
                - audio_x: Input token sequences, shape [batch_size, seq_len]
                - audio_y: Target token sequences, shape [batch_size, seq_len]  
            device (torch.device): Target device (cuda/cpu)
            
        Returns:
            tuple: (audio_x, audio_y) moved to device
                - audio_x: torch.LongTensor, shape [batch_size, seq_len], device=device
                - audio_y: torch.LongTensor, shape [batch_size, seq_len], device=device
        """
        audio_x, audio_y = batch
        return audio_x.to(device), audio_y.to(device)

    def calculate_accuracy(self, output, target):
        """
        Calculate token-level prediction accuracy for monitoring training progress.
        
        Computes the fraction of tokens where the model's prediction (argmax of logits)
        matches the ground truth target token.
        
        Args:
            output (torch.Tensor): Model output logits, shape [batch_size * seq_len, vocab_size]
            target (torch.Tensor): Target tokens, shape [batch_size * seq_len]
            
        Returns:
            float: Accuracy as fraction of correct predictions in [0, 1]
        """
        pred = torch.argmax(output, dim=-1)  # Get predicted token indices
        correct = (pred == target).sum().item()  # Count correct predictions
        total = target.numel()  # Total number of tokens
        return correct / total

    def train_epoch(self):
        """
        Train the model for one complete epoch with mixed precision and gradient clipping.
        
        Processes all training batches, computing forward passes, losses, and gradients.
        Uses automatic mixed precision (AMP) with FP16 for memory efficiency and speed,
        gradient clipping for stability, and progress tracking.
        
        Returns:
            tuple: (average_loss, average_accuracy) for the epoch
                - average_loss (float): Mean cross-entropy loss across all batches
                - average_accuracy (float): Mean token accuracy across all batches
                
        Training pipeline per batch:
            1. Forward pass with mixed precision (FP16)
            2. Compute cross-entropy loss on flattened sequences
            3. Backward pass with gradient scaling
            4. Gradient clipping (norm=1.0) for stability
            5. Optimizer step with gradient unscaling
        """
        self.base_model.train()
        self.compiled_model.train()

        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc="Training", leave=False)

        for batch in pbar:
            try:
                # Move batch to device
                audio_x, audio_y = self.prepare_batch(batch, self.device)
                self.optimizer.zero_grad(set_to_none=True)

                # Forward pass with automatic mixed precision (FP16)
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    output = self.compiled_model(audio_x)  # [B, T, vocab_size]
                    B, T, C = output.shape
                    
                    # Flatten for cross-entropy loss computation
                    output_flat = output.reshape(-1, C)      # [B*T, vocab_size]
                    target_flat = audio_y.reshape(-1)        # [B*T]
                    
                    # Compute cross-entropy loss (supports FP16)
                    loss = F.cross_entropy(output_flat, target_flat)

                # Calculate accuracy for monitoring
                accuracy = self.calculate_accuracy(output_flat, target_flat)

                # Backward pass with gradient scaling for mixed precision
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                
                # Gradient clipping for training stability
                torch.nn.utils.clip_grad_norm_(
                    self.base_model.parameters(), 
                    self.training_config['gradient_clip_norm']
                )
                
                # Optimizer step with automatic scaling
                self.scaler.step(self.optimizer)
                self.scaler.update()

                # Accumulate metrics
                total_loss += loss.item()
                total_accuracy += accuracy
                num_batches += 1

                # Update progress bar
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{accuracy:.3f}'
                })

            except Exception as e:
                print(f"Error in training batch: {e}")
                continue

        return total_loss / max(num_batches, 1), total_accuracy / max(num_batches, 1)

    def validate_epoch(self):
        """
        Validate the model for one complete epoch without gradient computation.
        
        Evaluates model performance on validation data using the same loss and
        accuracy metrics as training. Uses mixed precision for consistency and
        efficiency, but disables gradient computation for speed and memory savings.
        
        Returns:
            tuple: (average_loss, average_accuracy) for validation epoch
                - average_loss (float): Mean cross-entropy loss across validation batches
                - average_accuracy (float): Mean token accuracy across validation batches
                
        Validation pipeline per batch:
            1. Forward pass with mixed precision (FP16) and no gradients
            2. Compute cross-entropy loss on flattened sequences
            3. Calculate accuracy metrics for monitoring
        """
        self.base_model.eval()

        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0

        with torch.no_grad():  # Disable gradient computation for efficiency
            pbar = tqdm(self.val_loader, desc="Validation", leave=False)

            for batch in pbar:
                try:
                    # Move batch to device
                    audio_x, audio_y = self.prepare_batch(batch, self.device)
                    
                    # Forward pass with mixed precision (no gradients)
                    with torch.amp.autocast('cuda', dtype=torch.float16):
                        output = self.base_model(audio_x)  # [B, T, vocab_size]
                        B, T, C = output.shape
                        
                        # Flatten for loss computation
                        output_flat = output.reshape(-1, C)      # [B*T, vocab_size]
                        target_flat = audio_y.reshape(-1)        # [B*T]
                        
                        # Compute validation loss
                        loss = F.cross_entropy(output_flat, target_flat)

                    # Calculate accuracy for monitoring
                    accuracy = self.calculate_accuracy(output_flat, target_flat)

                    # Accumulate metrics
                    total_loss += loss.item()
                    total_accuracy += accuracy
                    num_batches += 1

                    # Update progress bar
                    pbar.set_postfix({
                        'Val Loss': f'{loss.item():.4f}',
                        'Val Acc': f'{accuracy:.3f}'
                    })

                except Exception as e:
                    print(f"Error in validation batch: {e}")
                    continue

        return total_loss / max(num_batches, 1), total_accuracy / max(num_batches, 1)

    def plot_training_progress(self, epoch):
        """
        Create and save comprehensive training progress visualization.
        
        Generates a multi-panel plot showing training/validation loss curves,
        accuracy curves, and overfitting indicator. Saves plots to checkpoint
        directory for monitoring training health and performance.
        
        Args:
            epoch (int): Current epoch number (0-indexed) for plot title and filename
            
        Plots created:
            1. Loss curves: Training and validation loss over epochs
            2. Accuracy curves: Training and validation accuracy over epochs  
            3. Overfitting indicator: Difference between validation and training loss
            
        Saves:
            - PNG file: progress_epoch_{epoch+1}.png in checkpoint directory
            - Displays plot in notebook if running interactively
        """
        train_losses = self.training_stats['train_losses']
        val_losses = self.training_stats['val_losses']
        train_accuracies = self.training_stats['train_accuracies']
        val_accuracies = self.training_stats['val_accuracies']

        if len(train_losses) == 0:
            return

        plt.figure(figsize=(15, 5))

        # Loss plot
        plt.subplot(1, 3, 1)
        epochs_range = range(1, len(train_losses) + 1)
        plt.plot(epochs_range, train_losses, 'b-',
                 label='Training Loss', linewidth=2)
        plt.plot(epochs_range, val_losses, 'r-',
                 label='Validation Loss', linewidth=2)
        plt.title(f'Training Progress (Epoch {epoch+1})')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Accuracy plot
        plt.subplot(1, 3, 2)
        plt.plot(epochs_range, train_accuracies, 'b-',
                 label='Training Accuracy', linewidth=2)
        plt.plot(epochs_range, val_accuracies, 'r-',
                 label='Validation Accuracy', linewidth=2)
        plt.title(f'Accuracy Progress (Epoch {epoch+1})')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Loss difference (overfitting indicator)
        plt.subplot(1, 3, 3)
        loss_diff = [v - t for t, v in zip(train_losses, val_losses)]
        plt.plot(epochs_range, loss_diff, 'g-', linewidth=2)
        plt.title('Overfitting Indicator (Val - Train Loss)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss Difference')
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.checkpoint_dir,
                    f'progress_epoch_{epoch+1}.png'), dpi=150)
        plt.show()

    def train(self, num_epochs):
        """
        Execute the complete training loop with checkpointing and progress tracking.
        
        Runs training for the specified number of epochs, handling resumption from
        checkpoints, validation, learning rate scheduling, progress visualization,
        and automatic saving of best models.
        
        Args:
            num_epochs (int): Total number of epochs to train for. If model has
                            already been trained beyond this, training is skipped.
                            
        Training flow per epoch:
            1. Training phase: Process all training batches
            2. Validation phase: Evaluate on validation data
            3. Learning rate scheduling: Update LR based on validation loss
            4. Checkpointing: Save latest model and best model if improved
            5. Progress visualization: Plot metrics every 5 epochs
            6. Training summary: Save final results to JSON
            
        Outputs:
            - Checkpoint files: latest_checkpoint.pth, best_model.pth
            - Progress plots: progress_epoch_*.png files
            - Training summary: training_summary.json with final metrics
        """
        # Check if training is already complete
        if self.start_epoch >= num_epochs:
            print(f"Model is already trained to {num_epochs} epochs.")
            return

        print(f"\n{'='*60}")
        print(f"🚀 Training...")
        print(f"📊 Epochs: {self.start_epoch} → {num_epochs}")
        print(f"{'='*60}")

        checkpoint_config = get_checkpoint_config()
        plot_every_n_epochs = checkpoint_config['plot_every_n_epochs']

        for epoch in range(self.start_epoch, num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")

            # Training
            train_loss, train_acc = self.train_epoch()

            # Validation
            val_loss, val_acc = self.validate_epoch()

            # Update scheduler
            self.scheduler.step(val_loss)

            # Save metrics
            self.training_stats['train_losses'].append(train_loss)
            self.training_stats['val_losses'].append(val_loss)
            self.training_stats['train_accuracies'].append(train_acc)
            self.training_stats['val_accuracies'].append(val_acc)

            # Check for best model
            is_best = val_loss < self.training_stats['best_val_loss']
            if is_best:
                self.training_stats['best_val_loss'] = val_loss

            # Save checkpoint every epoch.
            self.trainable_model.save(
                epoch, self.training_stats, self.learning_rate, is_best)

            # Print epoch summary
            improvement = "🔥" if is_best else ""
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.3f}")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {
                  val_acc:.3f} {improvement}")
            print(f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")

            # Plot every N epochs or if it's the first resumed epoch
            if (epoch + 1) % plot_every_n_epochs == 0 or epoch == self.start_epoch:
                self.plot_training_progress(epoch)

        print(f"\n🎉 Training completed!")
        print(f"Best validation loss: {
              self.training_stats['best_val_loss']:.4f}")
        print(f"Total epochs trained: {
              len(self.training_stats['train_losses'])}")
        print(f"Checkpoints saved in: {self.checkpoint_dir}")

        # Final comprehensive plot.
        self.plot_training_progress(
            len(self.training_stats['train_losses']) - 1)

        self._save_training_summary()
        print(f"Training completed!")

    def _save_training_summary(self):
        """
        Save comprehensive training summary to JSON file.
        
        Creates a summary of the complete training session including final metrics,
        best performance achieved, and total training duration. Useful for comparing
        different training runs and tracking model performance.
        
        Saves:
            training_summary.json containing:
                - total_epochs: Number of epochs trained
                - best_val_loss: Best validation loss achieved  
                - final_train_loss: Training loss at end of training
                - final_val_loss: Validation loss at end of training
                - final_train_acc: Training accuracy at end of training
                - final_val_acc: Validation accuracy at end of training
        """
        final_results = {
            'total_epochs': len(self.training_stats['train_losses']),
            'best_val_loss': self.training_stats['best_val_loss'],
            'final_train_loss': self.training_stats['train_losses'][-1] if self.training_stats['train_losses'] else None,
            'final_val_loss': self.training_stats['val_losses'][-1] if self.training_stats['val_losses'] else None,
            'final_train_acc': self.training_stats['train_accuracies'][-1] if self.training_stats['train_accuracies'] else None,
            'final_val_acc': self.training_stats['val_accuracies'][-1] if self.training_stats['val_accuracies'] else None
        }
        with open(os.path.join(self.checkpoint_dir, 'training_summary.json'), 'w') as f:
            json.dump(final_results, f, indent=2)
        print(f"📋 Training summary saved to: {
              os.path.join(self.checkpoint_dir, 'training_summary.json')}")


def sanity_check_batch(batch):
    """
    Perform sanity checks on a training batch.
    
    Validates that token values are in the correct range for mu-law encoding
    and that tensors have the expected data types.
    
    Args:
        batch: Tuple of (x, y) tensors from data loader
    """
    x, y = batch
    print("x dtype:", x.dtype, "min:", int(x.min()), "max:", int(x.max()))
    print("y dtype:", y.dtype, "min:", int(y.min()), "max:", int(y.max()))
    assert x.dtype == torch.long and y.dtype == torch.long
    assert (x >= 0).all() and (x <= 255).all()
    assert (y >= 0).all() and (y <= 255).all()
    print("✅ Batch sanity check passed")


def test_model_output_consistency(model, test_loader, device, num_batches=5):
    """
    Test model output consistency and numerical stability.
    
    Args:
        model: WaveNet model to test
        test_loader: Data loader for testing
        device: Computation device
        num_batches: Number of batches to test
    """
    model.eval()
    
    print("Testing model output consistency...")
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= num_batches:
                break
                
            x, y = batch
            x, y = x.to(device), y.to(device)
            
            # Test forward pass
            try:
                output = model(x)
                print(f"Batch {i+1}: Input shape {x.shape}, Output shape {output.shape}")
                
                # Check for NaN or Inf
                if torch.isnan(output).any():
                    print(f"⚠️ Warning: NaN detected in output for batch {i+1}")
                if torch.isinf(output).any():
                    print(f"⚠️ Warning: Inf detected in output for batch {i+1}")
                    
                # Check output range (logits can be any value)
                print(f"  Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
                
            except Exception as e:
                print(f"❌ Error in batch {i+1}: {e}")
                
    print("✅ Model output consistency test completed")
