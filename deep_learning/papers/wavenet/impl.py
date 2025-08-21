import numpy as np
import torch, torchaudio
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import os

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")

# Add this at the beginning of your notebook:
import torch._inductor.config as config
config.triton.cudagraphs = False





device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability()
    print(f"Compute Capability: {major}.{minor}")
    if major >= 8:
        print("✅ TF32 is supported (Ampere or newer).")
    else:
        print("❌ TF32 is not supported.")

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'  # For better debugging
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
scaler = torch.amp.GradScaler('cuda', enabled=True)



from IPython.display import Audio, display

# Load LJSpeech dataset.
# Each audio file is a single-channel 16-bit PCM WAV with a sample rate of 22050 Hz.
dataset = torchaudio.datasets.LJSPEECH(
    root="./data",
    download=True
)

# Print dataset info.
print(f"Number of samples in dataset: {len(dataset)}")
print(f"Sample rate: {dataset[0][1]}")
print(f"Example utterance text: {dataset[0][2]}")





def display_waveform(audio, title=""):
    plt.figure(figsize=(14,4))
    plt.plot(audio.t().numpy())
    plt.title(f"Audio: {title}")
    plt.tight_layout()
    plt.show()

# Get first audio sample and plot waveform.
waveform, sample_rate, _, _ = dataset[0]

display_waveform(waveform)
Audio(waveform.numpy(), rate=sample_rate)





config = {
    "batch_size": 4,
    "num_workers": 2,  # CHANGED: Use 0 for debugging, multiprocessing can cause issues
    "pin_memory": True,
    "mu": 255,
    "sr": 16000,
    "trim_silence_thresh": 1e-3, # Length of audio segments to train on in samples (not milliseconds)
    "window_size": 32001,  # ~2 seconds at 16kHz

    # Wavenet architecture related.
    "residual_channels": 64,
    "skip_channels": 256,
    "output_dim": 256,
    "n_layers": 10,
    "n_blocks": 5,
    "kernel_size": 2,
}





class MuLawEncoding:
    def __init__(self, mu=255):
        self.mu = mu

    def mu_law_encode(self, x):
        return torchaudio.transforms.MuLawEncoding(self.mu)(x)

    def mu_law_decode(self, q, boost_factor=1.0):
        return torchaudio.transforms.MuLawDecoding(self.mu)(q)





def play_encoded_sample(encoded_sample, boost_factor=1.0):
    '''
    Play a single sample obtained from the dataset. The sample is assumed to be **encoded**
    and needs to be decoded before playback.
    '''
    mu = MuLawEncoding()
    decoded_audio = mu.mu_law_decode(encoded_sample, boost_factor=boost_factor)
    audio_for_playback = decoded_audio.squeeze().numpy()
    audio_player = Audio(audio_for_playback, rate=config['sr'])
    display(audio_player)

def play_batch(batch):
    audio, _ = batch  
    audio = audio[0] # Get first item from batch
    play_encoded_sample(audio)





class AudioProcessor:
    def __init__(self):
        self.mu_law_encoding = MuLawEncoding()
        self.resamplers = {}  # Cache resamplers for efficiency

    def normalize(self, x):
        # Ensure audio is float32 and normalize to [-1, 1]
        if x.dtype != torch.float32:
            x = x.float()

        # peak normalization
        max_val = torch.max(torch.abs(x))
        if max_val > 0:  # Avoid division by zero
            x = x / max_val

        # rms normalization (optional)
        target_rms = 0.1
        def rms(x):
            return torch.sqrt(torch.mean(x**2) + 1e-8)
        current_rms = rms(x)
        if current_rms > 0:  # Avoid division by zero
            x = x * (target_rms / current_rms)

        return x

    def resample_audio(self, audio, orig_sr, target_sr):
        if orig_sr == target_sr:
            return audio

        # Use cached resampler for efficiency
        resampler_key = f"{orig_sr}_{target_sr}"
        if resampler_key not in self.resamplers:
            self.resamplers[resampler_key] = torchaudio.transforms.Resample(
                orig_freq=orig_sr,
                new_freq=target_sr
            )

        return self.resamplers[resampler_key](audio)

    # Trim leading/trailing silence
    def trim_silence(self, sig, thresh=config['trim_silence_thresh']):
        # sig: [1,T] tensor
        # returns: [1,T'] tensor with silence trimmed
        # Calculate energy
        energy = sig.abs().squeeze()
        # Find indices where energy is above threshold
        idx = torch.where(energy > thresh)[0]
        if len(idx) == 0:
            return sig  # Return original if no samples above threshold
        # Return trimmed signal
        return sig[:, idx[0].item():idx[-1].item() + 1]

    def segment_audio(self, audio, drop_last=True, hop_size=None):
        """
        Split [1, T] into windows of size config['window_size'].
        - hop_size=None → non-overlapping
        - hop_size < window_size → overlapping
        - drop_last=True → drop incomplete tail (recommended to avoid padding loss)
        """
        window_size = config['window_size']
        hop = hop_size or window_size
        T = audio.shape[1]

        segments = []
        # all full windows
        for start in range(0, T - window_size + 1, hop):
            segments.append(audio[:, start:start + window_size])

        # optional last (padded) window
        if not drop_last and (T < window_size or (T - window_size) % hop != 0):
            last_start = max(0, T - window_size)
            tail = audio[:, last_start:]
            if tail.shape[1] < window_size:
                tail = F.pad(tail, (0, window_size - tail.shape[1]))
            segments.append(tail)

        return segments

    def collate_fn(self, batch):
        """
        Returns:
            x: LongTensor [B_total, W-1]
            y: LongTensor [B_total, W-1]
        where B_total = sum of (#segments per item in batch).
        """
        x_list, y_list = [], []

        for (audio, sr, text, normalized_text) in batch:
            # resample → (optional) trim → normalize
            audio = self.resample_audio(audio, sr, config['sr'])
            audio = self.trim_silence(audio)              # you can disable if you want
            audio = self.normalize(audio)                 # peak-normalize only

            if audio.size(0) > 1:                         # safety: downmix stereo
                audio = audio.mean(dim=0, keepdim=True)

            # Generate multiple segments
            segments = self.segment_audio(
                audio,
                drop_last=True,                           # avoid padded tails in loss
                hop_size=config.get('hop_size', 8000)     # e.g., 8000 for 50% overlap @16kHz
            )

            # convert each segment to tokens and teacher-forced pairs
            for seg in segments:
                q = self.mu_law_encoding.mu_law_encode(seg.squeeze(0))  # [W]
                x_list.append(q[:-1])
                y_list.append(q[1:])

        if len(x_list) == 0:
            return (torch.empty(0, 0, dtype=torch.long),
                    torch.empty(0, 0, dtype=torch.long))

        max_len = max(t.size(0) for t in x_list)
        x = torch.stack([F.pad(t, (0, max_len - t.size(0)), value=0) for t in x_list], dim=0)
        y = torch.stack([F.pad(t, (0, max_len - t.size(0)), value=0) for t in y_list], dim=0)
        return x.long(), y.long()





from torch.utils.data import Subset

# Split dataset into train and test sets (90-10 split)
train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(
    dataset, 
    [train_size, test_size]
)
fake_train_dataset, fake_test_dataset = Subset(dataset, [0]), Subset(dataset, [0])

print(f"Training set size: {len(train_dataset)}")
print(f"Test set size: {len(test_dataset)}")
print(f"Fake training set size: {len(fake_train_dataset)}")
print(f"Fake test set size: {len(fake_test_dataset)}")

# Audio processor.
audio_processor = AudioProcessor()

# Create dataloaders
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=config['batch_size'],
    shuffle=True,
    num_workers= max(4, os.cpu_count()//2),
    pin_memory=True,
    persistent_workers=True,         # keeps workers alive between iters
    prefetch_factor=4,
    collate_fn=lambda batch: audio_processor.collate_fn(batch),
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, 
    batch_size=config['batch_size'],
    shuffle=False,
    num_workers= max(4, os.cpu_count()//2),
    pin_memory=True,
    persistent_workers=True,         # keeps workers alive between iters
    prefetch_factor=4,
    collate_fn=lambda batch: audio_processor.collate_fn(batch),
)
print(f"Number of training batches: {len(train_loader)}")
print(f"Number of test batches: {len(test_loader)}")

def fake_collate_fn(batch):
    x, y = audio_processor.collate_fn(batch)
    return x[0].unsqueeze(0), y[0].unsqueeze(0)

fake_train_loader = torch.utils.data.DataLoader(
    fake_train_dataset,
    batch_size=1,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
    collate_fn=fake_collate_fn,
)
fake_test_loader = torch.utils.data.DataLoader(
    fake_test_dataset, 
    batch_size=1,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
    collate_fn=fake_collate_fn,
)





for batch in fake_train_loader:
    print(batch[0].shape)
    print(batch[1].shape)
    play_encoded_sample(batch[0])
    break





class CausalDilatedConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, dilation=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.conv1d = nn.Conv1d(
            in_channels, 
            out_channels,
            kernel_size=kernel_size, 
            dilation=dilation, 
            padding=0, 
            bias=True)

    def forward(self, x):
        if self.kernel_size > 1:
            x = F.pad(x, ((self.kernel_size - 1) * self.dilation, 0), mode='constant', value=0)
        return self.conv1d(x)

# Test the module.
# Get all audio samples from the batch and stack them into a single tensor.
batch = torch.stack([item[0].squeeze() for item in next(iter(train_loader))])
print(f"batch tensor shape: {batch.shape}")

module = CausalDilatedConvolution(64, 64, kernel_size=2, dilation=2).to(device)
embedding = nn.Embedding(256, 64)

batch = embedding(batch)
batch = batch.permute(0, 2, 1)
print(f"batch tensor shape after embedding: {batch.shape}")

output = module(batch.to(device))
print(f"output tensor shape: {output.shape}")
assert output.shape == batch.shape, "Output shape should be same as input shape."






class ResidualBlock(nn.Module):
    def __init__(self, C_res, C_skip, dilation=1):
        super().__init__()
        self.dilated_conv = CausalDilatedConvolution(C_res, 2* C_res, kernel_size=2, dilation=dilation)
        # self.filter_conv = CausalDilatedConvolution(C_res, C_res, kernel_size=2, dilation=dilation)
        # self.gate_conv = CausalDilatedConvolution(C_res, C_res, kernel_size=2, dilation=dilation)
        self.skip_conv1x1 = nn.Conv1d(C_res, C_skip, 1)
        self.res_conv1x1 = nn.Conv1d(C_res, C_res, 1)

    def forward(self, x):
        output = self.dilated_conv(x)
        # Split into two halves along the channel dimension (dim=1)
        # Each half has shape: [B, C_res, T]
        filter_out, gate_out = torch.chunk(output, 2, dim=1)

        # Apply the gated activation unit
        gated = torch.tanh(filter_out) * torch.sigmoid(gate_out)

        # 1x1 convolutions for residual and skip connections
        residual = self.res_conv1x1(gated)
        skip = self.skip_conv1x1(gated)

        # Return the residual connection (for the next block) and the skip connection (for the final output)
        return residual + x, skip

    # def forward(self, x):
    #     z = torch.tanh(self.filter_conv(x)) * torch.sigmoid(self.gate_conv(x))
    #     skip = self.skip_conv1x1(z)
    #     out = self.res_conv1x1(z)
    #     return out + x, skip

# Test the module.
# Get all audio samples from the batch and stack them into a single tensor.
batch = torch.stack([item[0].squeeze() for item in next(iter(train_loader))])
print(f"batch tensor shape: {batch.shape}")

module = ResidualBlock(64, 64, dilation=2).to(device)
embedding = nn.Embedding(256, 64)

batch = embedding(batch)
batch = batch.permute(0, 2, 1)
print(f"batch tensor shape after embedding: {batch.shape}")

output, skip = module(batch.to(device))
print(f"output tensor shape: {output.shape}")
print(f"skip tensor shape: {skip.shape}")
assert output.shape == batch.shape, "Output shape should be same as input shape."





class Wavenet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.C_res  = config['residual_channels']
        self.C_output = config['output_dim']
        self.C_skip = config['skip_channels']
        self.kernel_size = config['kernel_size']
        self._rf = self._compute_receptive_field()

        # Converts [B, T] to [B, T, C_res]
        self.embedding = nn.Embedding(self.C_output, self.C_res, dtype=torch.float32)

        # For stability.
        self.conv1d = nn.Conv1d(self.C_res, self.C_res, kernel_size=1)

        # Each block has N-layers with dilations = 1,2,4,8,...512.
        self.residual_blocks = nn.ModuleList([
            self._create_residual_block() for i in range(config['n_blocks'])
        ])
        self.output_head = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(self.C_skip, self.C_skip, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(self.C_skip, self.C_output, kernel_size=1))

    def _create_residual_block(self):
        # Each residual block has n-layers with dilations = 1,2,4,8,...512
        return nn.ModuleList([
            ResidualBlock(self.C_res, self.C_skip, dilation=2**i)
            for i in range(self.config['n_layers'])
        ])

    def forward(self, x):
        # [B, T] -> [B, T, C]
        x_embd = self.embedding(x)
        # [B, T, C] -> [B, C, T]
        x_embd = x_embd.permute(0, 2, 1)
        skip_output = None

        # [B, C_hidden, T] as all convolutions expect this structure.
        x_embd = self.conv1d(x_embd)

        for residual_block in self.residual_blocks:
            for layer in residual_block:
                x_embd, x_skip = layer(x_embd)
                skip_output = x_skip if skip_output is None else skip_output + x_skip

        output = self.output_head(skip_output)

        # [B, T, C_output] again.
        return output.permute(0, 2, 1)

    def _compute_receptive_field(self):
        # RF = 1 + B * (2^L - 1)  (for k=2)
        if self.kernel_size != 2:
            # general: RF = 1 + B * sum_{i=0..L-1} (k-1)*2^i
            return 1 + self.config['n_blocks'] * ((self.kernel_size - 1) * (2**self.config['n_layers'] - 1))
        return 1 + self.config['n_blocks'] * (2**self.config['n_layers'] - 1)

    def get_receptive_field(self):
        return self._rf

# Test the module.
# Get all audio samples from the batch and stack them into a single tensor.
batch = torch.stack([item[0].squeeze() for item in next(iter(train_loader))])
print(f"batch tensor shape: {batch.shape}")

module = Wavenet(config).to(device)
total_params = sum(p.numel() for p in module.parameters())
print(f"Total parameters: {total_params:,}")
print(f"Model size: {(total_params * 4) / (1024**2):.2f} MB")  # Assuming float32 (4 bytes)
output = module(batch.to(device))
print(f"output tensor shape: {output.shape}")





# Figure out dataloaders, etc. based on training requirements.
# Fake dataset setup to test if training is working and is able to overfit on 1 sample.
USE_FAKE = True
if USE_FAKE:
    config['batch_size'] = 1
    config['num_workers'] = 0

checkpoint_dir = './wavenet_checkpoints_fake' if USE_FAKE else './wavenet_checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

train_loader = fake_train_loader if USE_FAKE else train_loader
test_loader = fake_test_loader if USE_FAKE else test_loader






class Model:
    def __init__(self, config, base_model, checkpoint_dir):
        self.config = config
        self.base_model = base_model
        self.model_train = None
        self.model_eval = None
        self.checkpoint_dir = checkpoint_dir

    def _find_checkpoints(self):
        """Find the most recent checkpoint"""
        latest_path = self._get_latest_checkpoint_path()
        best_path = self._get_best_checkpoint_path()
        latest_path = latest_path if os.path.exists(latest_path) else None
        best_path = best_path if os.path.exists(best_path) else None
        return latest_path, best_path

    def _get_best_checkpoint_path(self):
        return os.path.join(self.checkpoint_dir, 'best_model.pth')

    def _get_latest_checkpoint_path(self):
        return os.path.join(self.checkpoint_dir, 'latest_checkpoint.pth')

    def _compile_for_training(self):
        self.base_model.train()
        return torch.compile(self.base_model, mode="reduce-overhead")  # no flip to eval on this

    def _compile_for_eval(self):
        self.base_model.eval()
        return torch.compile(self.base_model, mode="reduce-overhead")


class TrainableModel(Model):
    def __init__(self, config, checkpoint_dir, base_model, optimizer, scheduler, load_from_checkpoint=False):
        super().__init__(config, base_model, checkpoint_dir)

        self.base_model = base_model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.training_stats = {
            'train_losses': [],
            'val_losses': [],
            'train_accuracies': [],
            'val_accuracies': [],
            'best_val_loss': float('inf'),
        }
        # 0 indexed epoch till which the model has been trained. -1 for untrained model.
        self.trained_till_epoch_index = -1
        if load_from_checkpoint:
            self._load_from_checkpoint()

        # Model to be used only for training.
        self.compiled_model = self._compile_for_training()


    def save(self, epoch, training_stats, learning_rate, is_best=False):
        """Save checkpoint with all training state and the base model."""
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

        # Save latest checkpoint for training.
        torch.save(checkpoint, self._get_latest_checkpoint_path())

        # Save best model, for eval.
        if is_best:
            torch.save(checkpoint, self._get_best_checkpoint_path())
            print(f"💾 New best model saved! Val Loss: {training_stats['best_val_loss']:.4f}")

    def _load_from_checkpoint(self):
        latest_checkpoint_path, _ = self._find_checkpoints()

        if not latest_checkpoint_path:
            print("🆕 No existing checkpoints found. Loading base model.")
            return

        print(f"🔄 Loading model checkpoint from: {latest_checkpoint_path}")
        checkpoint = torch.load(latest_checkpoint_path, map_location=device)

        self.base_model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Load training metrics
        train_losses = checkpoint.get('train_losses', [])
        val_losses = checkpoint.get('val_losses', [])
        train_accuracies = checkpoint.get('train_accuracies', [])
        val_accuracies = checkpoint.get('val_accuracies', [])
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.trained_till_epoch_index = checkpoint.get('trained_till_epoch_index', -1)
        self.training_stats = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies,
            'best_val_loss': best_val_loss,
        }
        print(f"✅ Checkpoint loaded successfully. Trained for {self.trained_till_epoch_index + 1} epochs!")


class EvalModel(Model):
    def __init__(self, config, checkpoint_dir, base_model):
        super().__init__(config, base_model, checkpoint_dir)
        self.base_model = base_model
        self._load_from_checkpoint()

        # Model to be used only for evaluation.
        self.compiled_model = self._compile_for_eval()

    def _load_from_checkpoint(self):
        _, best_checkpoint_path = self._find_checkpoints()

        if not best_checkpoint_path:
            print("🆕 No existing checkpoints found for evaluation")
            return

        print(f"🔄 Loading eval model checkpoint from: {best_checkpoint_path}")
        checkpoint = torch.load(best_checkpoint_path, map_location=device)
        self.base_model.load_state_dict(checkpoint['model_state_dict'])

        epoch = checkpoint.get('trained_till_epoch_index', 0)
        print(f"✅ Eval model checkpoint loaded successfully. Trained till epoch {epoch + 1}!")





class Trainer:
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

        self.training_stats = {
            'train_losses': [],
            'val_losses': [],
            'train_accuracies': [], 
            'val_accuracies': [],
            'best_val_loss': float('inf'),
        }
        self.start_epoch = self.trainable_model.trained_till_epoch_index + 1

    def prepare_batch(self, batch, device):
        """Prepare batch for autoregressive training"""
        audio_x, audio_y = batch
        return audio_x.to(device), audio_y.to(device)

    def calculate_accuracy(self, output, target):
        """Calculate prediction accuracy"""
        pred = torch.argmax(output, dim=-1)
        correct = (pred == target).sum().item()
        total = target.numel()
        return correct / total

    def train_epoch(self):
        """Train for one epoch"""
        self.base_model.train()
        self.compiled_model.train()

        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc="Training", leave=False)

        for batch in pbar:
            try:
                audio_x, audio_y = self.prepare_batch(batch, self.device)

                self.optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast("cuda", dtype=torch.float16):
                    output = self.compiled_model(audio_x)                     # [B,T,256] in fp16
                    B, T, C = output.shape
                    output_flat = output.reshape(-1, C)
                    target_flat = audio_y.reshape(-1)
                    loss = F.cross_entropy(output_flat, target_flat)  # CE supports fp16

                accuracy = self.calculate_accuracy(output_flat, target_flat)

                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.base_model.parameters(), 1.0)
                scaler.step(self.optimizer)
                scaler.update()

                total_loss += loss.item()
                total_accuracy += accuracy
                num_batches += 1

                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{accuracy:.3f}'
                })

            except Exception as e:
                print(f"Error in training batch: {e}")

        return total_loss / max(num_batches, 1), total_accuracy / max(num_batches, 1)

    def validate_epoch(self):
        """Validate for one epoch"""
        self.base_model.eval()

        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0

        with torch.no_grad():
            pbar = tqdm(test_loader, desc="Validation", leave=False)

            for batch in pbar:
                try:
                    audio_x, audio_y = self.prepare_batch(batch, device)
                    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
                        output = self.base_model(audio_x)
                        B,T,C = output.shape
                        output_flat = output.reshape(-1, C)
                        target_flat = audio_y.reshape(-1)
                        loss = F.cross_entropy(output_flat, target_flat)

                    accuracy = self.calculate_accuracy(output_flat, target_flat)

                    total_loss += loss.item()
                    total_accuracy += accuracy
                    num_batches += 1

                    pbar.set_postfix({
                        'Val Loss': f'{loss.item():.4f}',
                        'Val Acc': f'{accuracy:.3f}'
                    })

                except Exception as e:
                    print(f"Error in validation batch: {e}")
                    continue

        return total_loss / max(num_batches, 1), total_accuracy / max(num_batches, 1)

    def plot_training_progress(self, epoch):
        """Plot training progress with history"""
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
        plt.plot(epochs_range, train_losses, 'b-', label='Training Loss', linewidth=2)
        plt.plot(epochs_range, val_losses, 'r-', label='Validation Loss', linewidth=2)
        plt.title(f'Training Progress (Epoch {epoch+1})')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Accuracy plot
        plt.subplot(1, 3, 2)
        plt.plot(epochs_range, train_accuracies, 'b-', label='Training Accuracy', linewidth=2)
        plt.plot(epochs_range, val_accuracies, 'r-', label='Validation Accuracy', linewidth=2)
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
        plt.savefig(os.path.join(checkpoint_dir, f'progress_epoch_{epoch+1}.png'), dpi=150)
        plt.show()

    def train(self, num_epochs):
        # Main training loop.
        if self.start_epoch >= num_epochs:
            print(f"Model is already trained to {num_epochs} epochs.")

        print(f"\n{'='*60}")
        print(f"🚀 Training...")
        print(f"📊 Epochs: {self.start_epoch} → {num_epochs}")
        print(f"{'='*60}")

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
            self.trainable_model.save(epoch, self.training_stats, self.learning_rate, is_best)

            # Print epoch summary
            improvement = "🔥" if is_best else ""
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.3f}")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.3f} {improvement}")
            print(f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")

            # Plot every 5 epochs or if it's the first resumed epoch
            if (epoch + 1) % 5 == 0 or epoch == self.start_epoch:
                self.plot_training_progress(epoch)

        print(f"\n🎉 Training completed!")
        print(f"Best validation loss: {self.training_stats['best_val_loss']:.4f}")
        print(f"Total epochs trained: {len(self.training_stats['train_losses'])}")
        print(f"Checkpoints saved in: {self.checkpoint_dir}")

        # Final comprehensive plot.
        self.plot_training_progress(len(self.training_stats['train_losses']) - 1)

        # Save final results
        final_results = {
            'total_epochs': len(self.training_stats['train_losses']),
            'best_val_loss': self.training_stats['best_val_loss'],
            'final_train_loss': self.training_stats['train_losses'][-1] if self.training_stats['train_losses'] else None,
            'final_val_loss': self.training_stats['val_losses'][-1] if self.training_stats['val_losses'] else None,
            'final_train_acc': self.training_stats['train_accuracies'][-1] if self.training_stats['train_accuracies'] else None,
            'final_val_acc': self.training_stats['val_accuracies'][-1] if self.training_stats['val_accuracies'] else None
        }

        import json
        with open(os.path.join(checkpoint_dir, 'training_summary.json'), 'w') as f:
            json.dump(final_results, f, indent=2)

        print(f"📋 Training summary saved to: {os.path.join(checkpoint_dir, 'training_summary.json')}")
        print(f"Training completed!")






# Number of epochs to train for. 
# If the saved checkpoint is < num_epochs then training will be done.
# Otherwise model will be loaded and put in eval mode and compiled.
num_epochs = 50 if not USE_FAKE else 150  # 300 is enough to overfit completely.

# Initialize base model and optimizer.
base_model = Wavenet(config).to(device)
learning_rate = 1e-3
optimizer = torch.optim.AdamW(base_model.parameters(), lr=learning_rate) #, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

# Load the latest trainable model.
trainable_model = TrainableModel(config, checkpoint_dir, base_model, optimizer, scheduler, 
                                 load_from_checkpoint=True)
print(f"Model parameters: {sum(p.numel() for p in trainable_model.base_model.parameters()):,}")

if trainable_model.trained_till_epoch_index + 1 < num_epochs:
    print(f"🚀 {'Resuming' if trainable_model.trained_till_epoch_index > -1 else 'Starting'} WaveNet Training")
    # Create the trainer.
    trainer = Trainer(base_model, trainable_model, config, learning_rate, optimizer, scheduler, checkpoint_dir, train_loader, test_loader, device)
    trainer.train(num_epochs)
else:
    print(f"Model is already trained to {num_epochs} epochs.")

# Reload model for evals and generation.
base_model = Wavenet(config).to(device)
model = EvalModel(config, checkpoint_dir, base_model)





import torch
import torch.nn.functional as F
from math import prod

@torch.no_grad()
def generate_continuation(
    model,
    seed_tokens: torch.LongTensor,   # [B, T_seed], class IDs in [0..255]
    n_steps: int,                    # how many new samples to generate
    temperature: float = 1.0,
    top_k: int | None = None,
    device: str | torch.device = "cuda",
    rf_override: int | None = None,  # optionally pass model RF
):
    model.eval()
    seq = seed_tokens.to(device)  # [B, T_seed]
    B = seq.size(0)
    assert B == 1, "start with batch size 1 for autoregressive generation"

    # try to infer RF from model config if not provided
    if rf_override is not None:
        rf = rf_override
    else:
        rf = model.get_receptive_field()

    for _ in tqdm(range(n_steps), desc='Generating'):
        # crop to receptive field context to save compute
        ctx = seq[:, -rf:] if seq.size(1) > rf else seq

        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16):
            logits = model(ctx)          # [1, T_ctx, 256]
            logits = logits[:, -1, :]    # [1, 256] last step

        # temperature / top-k sampling (avoid argmax for better naturalness)
        logits = logits / max(temperature, 1e-6)
        if top_k is not None and top_k > 0:
            topk_vals, topk_idx = torch.topk(logits, k=top_k, dim=-1)
            probs = torch.zeros_like(logits).scatter(-1, topk_idx, F.softmax(topk_vals, dim=-1))
        else:
            probs = F.softmax(logits, dim=-1)

        next_tok = torch.multinomial(probs, num_samples=1)  # [1,1] Long
        seq = torch.cat([seq, next_tok], dim=1)             # append token

    return seq  # [1, T_seed + n_steps] (tokens)


@torch.no_grad()
def generate_greedy(model, seed, n_steps, device="cuda", rf=None):
    model.eval()
    seq = seed.to(device)
    rf = rf or model.get_receptive_field()
    for _ in tqdm(range(n_steps), desc='Generating'):
        ctx = seq[:, -rf:] if seq.size(1) > rf else seq
        logits = model(ctx)[:, -1, :].float()      # force fp32 for softmax/argmax stability
        nxt = logits.argmax(dim=-1, keepdim=True)  # greedy
        seq = torch.cat([seq, nxt], 1)
    return seq





# Get one batch from test_loader; 
audio, _ = next(iter(test_loader))     # x: [B, T-1] tokens (Long)
audio = audio[np.random.randint(0, audio.shape[0]), :].unsqueeze(0).to(device)

print(audio.shape)

play_encoded_sample(audio.cpu())
display_waveform(audio.cpu(), "Original")

# Seconds to seed the generation with.
seconds_to_seed = 0.5
length_to_seed = int(16000*seconds_to_seed - 1)
seed = audio[:1, : length_to_seed]         # [1, seed_len]; ensure <= available length

# Seconds to generate.
sec_to_generate = 1.5
n_new = int(16000 * sec_to_generate)

play_encoded_sample(seed.cpu())
display_waveform(seed.cpu(), "Seed")

# audio_full = generate_continuation(
#     model.compiled_model,
#     seed_tokens=seed,
#     n_steps=n_new,
#     temperature=1.0,      # tweak 0.7–1.2
#     top_k=None,            # optional; try None or 50–200
#     device=device,
# )
audio_full = generate_greedy(model.compiled_model, seed, n_new, device=device)

play_encoded_sample(audio_full.cpu(), boost_factor=3.0)
display_waveform(audio_full.cpu(), "Full")



