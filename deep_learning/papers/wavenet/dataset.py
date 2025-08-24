"""
Dataset management for WaveNet training.

Contains audio preprocessing, dataset building, segmented token storage,
and data loading functionality for efficient WaveNet training.
"""

import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
import os
import json
import bisect
from pathlib import Path
from typing import Optional, Tuple, List
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from collections import deque
import warnings

from config import get_audio_config, get_dataset_config
from model import codec

warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")


class AudioProcessor:
    """
    Audio preprocessing utilities for WaveNet training and inference.
    
    Handles audio resampling, normalization, silence trimming, segmentation,
    and preparation of batched training data with mu-law encoding.
    
    Attributes:
        mu_law_encoding (MuLawEncoding): Codec for mu-law encoding/decoding
        resamplers (dict): Cache of resampler transforms for different sample rates
    """
    def __init__(self):
        self.mu_law_encoding = codec
        self.resamplers = {}  # Cache resamplers for efficiency
        self.config = get_audio_config()

    def normalize(self, x):
        """
        Normalize audio to [-1, 1] range using peak and RMS normalization.
        
        Applies peak normalization to ensure maximum amplitude is 1.0, followed by
        RMS normalization to standardize the energy level across different audio files.
        
        Args:
            x (torch.Tensor): Input audio waveform, shape [channels, time] or [time],
                            any dtype (will be converted to float32)
        
        Returns:
            torch.Tensor: Normalized audio, shape same as input, dtype float32,
                        values in approximately [-1, 1] range
        
        Example:
            >>> processor = AudioProcessor()
            >>> audio = torch.randn(1, 16000) * 10  # Loud audio
            >>> normalized = processor.normalize(audio)  # Peak around ±1.0
        """
        # Ensure audio is float32 and normalize to [-1, 1]
        if x.dtype != torch.float32:
            x = x.float()

        # Peak normalization: scale to maximum absolute value of 1.0
        max_val = torch.max(torch.abs(x))
        if max_val > 0:  # Avoid division by zero
            x = x / max_val

        # RMS normalization: standardize energy level
        target_rms = 0.1

        def rms(x):
            return torch.sqrt(torch.mean(x**2) + 1e-8)
        current_rms = rms(x)
        if current_rms > 0:  # Avoid division by zero
            x = x * (target_rms / current_rms)

        return x

    def resample_audio(self, audio, orig_sr, target_sr):
        """
        Resample audio to target sample rate with caching for efficiency.
        
        Uses cached resampler transforms to avoid recreating them for the same
        sample rate pairs, which significantly improves performance when processing
        many audio files.
        
        Args:
            audio (torch.Tensor): Input audio waveform, shape [channels, time]
            orig_sr (int): Original sample rate in Hz
            target_sr (int): Target sample rate in Hz
        
        Returns:
            torch.Tensor: Resampled audio, shape [channels, new_time] where
                        new_time = time * (target_sr / orig_sr)
        
        Example:
            >>> processor = AudioProcessor()
            >>> audio_44k = torch.randn(1, 44100)  # 1 second at 44.1kHz
            >>> audio_16k = processor.resample_audio(audio_44k, 44100, 16000)
            >>> print(audio_16k.shape)  # torch.Size([1, 16000])
        """
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

    def trim_silence(self, sig, thresh=None):
        """
        Remove leading and trailing silence from audio signal.
        
        Detects silence by computing energy (absolute value) and trimming 
        samples below the threshold from the beginning and end of the signal.
        
        Args:
            sig (torch.Tensor): Input audio signal, shape [1, time]
            thresh (float): Energy threshold below which samples are considered silence.
                          Default from config['trim_silence_thresh']
        
        Returns:
            torch.Tensor: Trimmed audio signal, shape [1, trimmed_time] where
                        trimmed_time <= time
        
        Example:
            >>> processor = AudioProcessor()
            >>> # Audio with silence padding
            >>> audio = torch.cat([torch.zeros(1, 1000), torch.randn(1, 8000), 
            ...                    torch.zeros(1, 1000)], dim=1)
            >>> trimmed = processor.trim_silence(audio, thresh=0.01)
            >>> print(f"Original: {audio.shape[1]}, Trimmed: {trimmed.shape[1]}")
        """
        if thresh is None:
            thresh = self.config['trim_silence_thresh']
            
        # Calculate energy (absolute value)
        energy = sig.abs().squeeze()
        # Find indices where energy is above threshold
        idx = torch.where(energy > thresh)[0]
        if len(idx) == 0:
            return sig  # Return original if no samples above threshold
        # Return trimmed signal
        return sig[:, idx[0].item():idx[-1].item() + 1]

    def segment_audio(self, audio, drop_last=True, hop_size=None):
        """
        Split audio into overlapping or non-overlapping segments for training.
        
        Divides long audio sequences into fixed-size windows that can be processed
        independently. Supports overlapping windows to increase training data.
        
        Args:
            audio (torch.Tensor): Input audio, shape [1, time]
            drop_last (bool): If True, drop incomplete segments at the end.
                            If False, pad the last segment. Default True.
            hop_size (int, optional): Step size between segments. If None,
                                    uses window_size (non-overlapping).
                                    If < window_size, creates overlapping segments.
        
        Returns:
            list[torch.Tensor]: List of audio segments, each with shape [1, window_size]
                              where window_size = config['window_size']
        
        Example:
            >>> processor = AudioProcessor()
            >>> audio = torch.randn(1, 100000)  # Long audio
            >>> segments = processor.segment_audio(audio, hop_size=16000)
            >>> print(f"Created {len(segments)} segments")
            >>> print(f"Each segment shape: {segments[0].shape}")  # [1, 32001]
        """
        window_size = self.config['window_size']
        hop = hop_size or window_size
        T = audio.shape[1]

        segments = []
        # Extract all full windows
        for start in range(0, T - window_size + 1, hop):
            segments.append(audio[:, start:start + window_size])

        # Handle incomplete tail segment
        if not drop_last and (T < window_size or (T - window_size) % hop != 0):
            last_start = max(0, T - window_size)
            tail = audio[:, last_start:]
            if tail.shape[1] < window_size:
                tail = F.pad(tail, (0, window_size - tail.shape[1]))
            segments.append(tail)

        return segments

    def collate_fn(self, batch):
        """
        Collate function for processing batches of audio data.
        
        Processes a batch of audio samples by resampling, trimming silence,
        normalizing, segmenting, and converting to mu-law tokens for training.
        
        Args:
            batch (list): List of tuples (audio, sr, text, normalized_text)
                        from the dataset
        
        Returns:
            tuple: (x, y) where:
                - x: LongTensor [B_total, W-1] input tokens
                - y: LongTensor [B_total, W-1] target tokens
                where B_total = sum of (#segments per item in batch)
                and W = window_size
        """
        x_list, y_list = [], []

        for (audio, sr, text, normalized_text) in batch:
            # resample → (optional) trim → normalize
            audio = self.resample_audio(audio, sr, self.config['sr'])
            audio = self.trim_silence(audio)              # you can disable if you want
            audio = self.normalize(audio)                 # peak-normalize only

            if audio.size(0) > 1:                         # safety: downmix stereo
                audio = audio.mean(dim=0, keepdim=True)

            # Generate multiple segments
            segments = self.segment_audio(
                audio,
                drop_last=True,                           # avoid padded tails in loss
                hop_size=self.config.get('hop_size', 16000)     # e.g., 16000 for 50% overlap @32000
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


class SegmentedTokensOnDisk(torch.utils.data.Dataset):
    """
    A PyTorch Dataset for loading audio tokens stored in sharded files on disk.
    
    This dataset efficiently loads pre-tokenized audio data from multiple shard files,
    enabling training on large audio datasets that don't fit in memory. Each sample
    consists of input tokens (x) and target tokens (y) for autoregressive training.
    
    Args:
        manifest_path (str): Path to JSON manifest file containing shard information
        root (str, optional): Root directory for shard files. Defaults to manifest directory
        cache_shards (bool): Whether to cache loaded shards in memory. Defaults to True
        
    Manifest JSON format:
        {
            "shards": ["shard_0.pt", "shard_1.pt", ...],
            "shard_sizes": [1000, 1000, ...],
            "num_samples": 2000,
            "seq_len": 32000,
            "stored_dtype": "long" or "uint8"
        }
    """
    def __init__(self, manifest_path, root=None, cache_shards=True):
        import json
        import torch
        from pathlib import Path
        mp = Path(manifest_path)
        man = json.loads(mp.read_text())
        self.root = mp.parent if root is None else Path(root)
        self.files = [self.root / s for s in man["shards"]]
        self.sizes = man["shard_sizes"]  # Number of samples per shard
        
        # Compute cumulative sizes for efficient sample location
        self.cum = []
        c = 0
        for s in self.sizes:
            c += s
            self.cum.append(c)
            
        self.N = man["num_samples"]  # Total number of samples
        self.T = man["seq_len"]      # Sequence length per sample
        self.cache = {}              # Cache for loaded shards
        self.stored_dtype = man.get("stored_dtype", "long")
        self.current_shard = -1      # Track currently loaded shard

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        
        Returns:
            int: Total number of samples across all shards
        """
        return self.N

    def _loc(self, idx):
        """
        Locate which shard contains the sample at given index and its offset within that shard.
        
        Args:
            idx (int): Global sample index
            
        Returns:
            tuple: (shard_index, offset_within_shard)
                - shard_index (int): Index of the shard containing the sample
                - offset_within_shard (int): Position of sample within the shard
        """
        import bisect
        s_idx = bisect.bisect_right(self.cum, idx)
        base = 0 if s_idx == 0 else self.cum[s_idx-1]
        off = idx - base
        return s_idx, off

    def get_shard(self, s_idx):
        """
        Load and cache a shard file containing tokenized audio data.
        
        Args:
            s_idx (int): Index of the shard to load
            
        Returns:
            dict: Loaded shard data containing:
                - 'x' or 'x_u8': Input token sequences, shape [shard_size, seq_len]
                - 'y' or 'y_u8': Target token sequences, shape [shard_size, seq_len]
        """
        # Clear cache if switching to different shard to save memory
        if self.current_shard != s_idx:
            self.cache = {}
            self.current_shard = -1

        self.current_shard = s_idx

        if s_idx in self.cache:
            shard = self.cache[s_idx]
        else:
            print(f"Loading shard {s_idx} from {self.files[s_idx]}")
            shard = torch.load(self.files[s_idx], map_location="cpu")
            self.cache[s_idx] = shard
        return shard

    def __getitem__(self, idx):
        """
        Get a sample (input, target) pair for autoregressive training.
        
        Args:
            idx (int): Sample index (0 to len(dataset)-1)
            
        Returns:
            tuple: (x, y) where:
                - x: Input tokens, shape [seq_len], dtype torch.long, values in [0, 255]
                - y: Target tokens, shape [seq_len], dtype torch.long, values in [0, 255]
                
        Note:
            Token values are clamped to [0, 255] range to ensure valid mu-law encoding.
        """
        s_idx, off = self._loc(idx)
        shard = self.get_shard(s_idx)
        
        if self.stored_dtype == "uint8":
            x = shard["x_u8"][off].to(torch.long)  # upcast once
            y = shard["y_u8"][off].to(torch.long)
        else:
            x = shard["x"][off]
            y = shard["y"][off]
            
        # Ensure tokens are in valid range for mu-law encoding
        if (x.gt(255).any() or y.gt(255).any()) or (x.lt(0).any() or y.lt(0).any()):
            x.clamp_(0, 255)
            y.clamp_(0, 255)
        return x, y


# Dataset building functionality
class _SegmentsPerItem(Dataset):
    """Wraps a base dataset so each __getitem__ returns all (x,y) segments for that item."""
    def __init__(self, base_ds, audio_processor):
        self.ds = base_ds
        self.proc = audio_processor

    def __len__(self): 
        return len(self.ds)

    def __getitem__(self, i):
        x, y = self.proc.collate_fn([self.ds[i]])  # x:[Si,T], y:[Si,T] or empty
        return x, y


def _cat_collate(batch):
    """Collate function for concatenating segment batches."""
    xs = [b[0] for b in batch if b[0].numel() > 0]
    ys = [b[1] for b in batch if b[1].numel() > 0]
    if not xs: 
        return None
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


def build_segmented_dataset_fast(base_dataset,
                                 audio_processor,
                                 num_workers: int = 8,
                                 batch_items: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build segmented dataset in memory for fast access.
    
    Args:
        base_dataset: Original audio dataset
        audio_processor: AudioProcessor instance
        num_workers: Number of worker processes
        batch_items: Batch size for processing
    
    Returns:
        tuple: (audio_x, audio_y) tensors with shape [N, T]
    """
    wrapped = _SegmentsPerItem(base_dataset, audio_processor)
    loader = DataLoader(
        wrapped,
        batch_size=batch_items,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        pin_memory=False,
        prefetch_factor=(4 if num_workers > 0 else None),
        collate_fn=_cat_collate,
    )
    xs, ys = [], []
    for out in tqdm(loader, total=(len(base_dataset)+batch_items-1)//batch_items, desc="Pre-segmenting"):
        if out is None: continue
        x, y = out
        xs.append(x.contiguous())
        ys.append(y.contiguous())
    if not xs:
        return (torch.empty(0, 0, dtype=torch.long), torch.empty(0, 0, dtype=torch.long))
    audio_x = torch.cat(xs, dim=0)
    audio_y = torch.cat(ys, dim=0)
    return audio_x, audio_y


def save_segmented_dataset_sharded(audio_x: torch.Tensor,
                                         audio_y: torch.Tensor,
                                         out_dir: str,
                                         shard_size: int = 512):
    """
    Save segmented dataset to sharded files on disk.
    
    Saves shards as uint8 to cut memory/disk by 8×.
    audio_x/audio_y are Long in [0,255]; we store as uint8 and upcast on load.
    
    Args:
        audio_x: Input token tensor [N, T]
        audio_y: Target token tensor [N, T]
        out_dir: Output directory for shards
        shard_size: Number of samples per shard
        
    Returns:
        Path: Path to manifest.json file
    """
    from pathlib import Path
    import json, torch

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    N, T = audio_x.size(0), audio_x.size(1)
    assert audio_y.size(0) == N and audio_y.size(1) == T

    shard_files, shard_sizes = [], []
    for s in range(0, N, shard_size):
        e = min(s + shard_size, N)
        x_u8 = audio_x[s:e].to(torch.uint8).contiguous()
        y_u8 = audio_y[s:e].to(torch.uint8).contiguous()
        f = f"shard_{s//shard_size:05d}.pt"
        torch.save({"x_u8": x_u8, "y_u8": y_u8}, out / f)
        shard_files.append(f)
        shard_sizes.append(e - s)

    manifest = {
        "version": 2,
        "num_samples": N,
        "seq_len": T,
        "stored_dtype": "uint8",
        "target_dtype": "long",     # what the model expects
        "shard_size": shard_size,
        "shards": shard_files,
        "shard_sizes": shard_sizes,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return out / "manifest.json"


def stream_preprocess_to_shards(base_dataset,
                                audio_processor,
                                out_dir: str,
                                shard_size: int = 10000,
                                num_workers: int = 8,
                                batch_items: int = 32):
    """
    Build dataset and stream to shards for low memory usage.
    
    Builds (x,y) token segments in parallel but writes shards incrementally to keep RAM low.
    Each shard has exactly <= shard_size samples. Handles partial consumption of a batch.
    
    Args:
        base_dataset: Original audio dataset
        audio_processor: AudioProcessor instance
        out_dir: Output directory for shards
        shard_size: Maximum samples per shard
        num_workers: Number of worker processes
        batch_items: Batch size for processing
        
    Returns:
        Path: Path to manifest.json file
    """
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    wrapped = _SegmentsPerItem(base_dataset, audio_processor)
    loader = DataLoader(
        wrapped,
        batch_size=batch_items,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        pin_memory=False,
        prefetch_factor=(4 if num_workers > 0 else None),
        collate_fn=_cat_collate,
    )

    # Buffers of tensors to consume from (FIFO)
    buf_x, buf_y = deque(), deque()
    buf_count = 0                       # total samples in buffers
    shard_idx = 0
    shard_files, shard_sizes = [], []
    total = 0
    seq_len = None
    dtype = None

    def _push_batch(x, y):
        nonlocal buf_count, seq_len, dtype
        if x is None: return
        assert x.size(0) == y.size(0)
        if seq_len is None:
            seq_len = x.size(1)
            dtype = str(x.dtype).replace("torch.", "")
        else:
            # sanity: all segments must share same length
            assert x.size(1) == seq_len, "Inconsistent segment length T across batches"
        buf_x.append(x)
        buf_y.append(y)
        buf_count += x.size(0)

    def _pop_exact_n(n):
        """Pop exactly n samples from buffers, returning cat(x_parts), cat(y_parts).
           Supports splitting the front tensor if needed.
        """
        nonlocal buf_count
        parts_x, parts_y = [], []
        need = n
        while need > 0:
            assert buf_x, "Buffer underflow"
            x0, y0 = buf_x[0], buf_y[0]
            m = x0.size(0)
            if m <= need:
                # take whole front tensor
                parts_x.append(x0)
                parts_y.append(y0)
                buf_x.popleft(); buf_y.popleft()
                buf_count -= m
                need -= m
            else:
                # take a slice and put leftovers back
                parts_x.append(x0[:need])
                parts_y.append(y0[:need])
                buf_x[0] = x0[need:]
                buf_y[0] = y0[need:]
                buf_count -= need
                need = 0
        X = torch.cat(parts_x, dim=0)
        Y = torch.cat(parts_y, dim=0)
        return X, Y

    def _flush_if_ready(force=False):
        """Write shards while we have >= shard_size, or if force=True write remaining."""
        nonlocal shard_idx, total
        while buf_count >= shard_size or (force and buf_count > 0):
            take = shard_size if buf_count >= shard_size else buf_count
            X, Y = _pop_exact_n(take)
            f = f"shard_{shard_idx:05d}.pt"
            torch.save({"x": X, "y": Y}, out / f)
            shard_files.append(f)
            shard_sizes.append(X.size(0))
            total += X.size(0)
            shard_idx += 1

    # main loop
    for out_batch in tqdm(loader, total=(len(base_dataset) + batch_items - 1)//batch_items, desc="Streaming pre-seg"):
        if out_batch is None:
            continue
        x, y = out_batch
        _push_batch(x, y)
        _flush_if_ready(force=False)

    # final flush
    _flush_if_ready(force=True)

    # write manifest
    manifest = {
        "version": 1,
        "num_samples": total,
        "seq_len": seq_len if seq_len is not None else 0,
        "dtype": dtype if dtype is not None else "long",
        "shard_size": shard_size,
        "shards": shard_files,
        "shard_sizes": shard_sizes,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return out / "manifest.json"


def create_datasets(config, use_existing_tokens=True):
    """
    Create train and test datasets for WaveNet training.
    
    Args:
        config: Configuration dictionary
        use_existing_tokens: Whether to use existing tokenized data or rebuild
        
    Returns:
        tuple: (train_dataset, test_dataset, train_loader, test_loader)
    """
    if use_existing_tokens and os.path.exists(config['manifest_file']):
        print(f"Loading existing tokenized dataset from {config['manifest_file']}")
        dataset = SegmentedTokensOnDisk(
            config['manifest_file'], 
            cache_shards=config['cache_shards']
        )
    else:
        print("Building new tokenized dataset...")
        # Load base dataset (LJSpeech)
        base_dataset = torchaudio.datasets.LJSPEECH(
            root=config['data_root'],
            download=True
        )
        
        # Create audio processor and build tokenized dataset
        audio_processor = AudioProcessor()
        manifest_path = stream_preprocess_to_shards(
            base_dataset=base_dataset,
            audio_processor=audio_processor,
            out_dir=config['segmented_tokens_dir'],
            shard_size=config['shard_size'],
            num_workers=8,
            batch_items=32,
        )
        print(f"Dataset built and saved to: {manifest_path}")
        
        dataset = SegmentedTokensOnDisk(
            str(manifest_path), 
            cache_shards=config['cache_shards']
        )
    
    print(f"Number of samples in dataset: {len(dataset)}")
    
    # Split dataset into train and test sets
    train_size = int(config['train_split'] * len(dataset))
    test_size = len(dataset) - train_size
    train_indices = list(range(0, train_size))
    test_indices = list(range(train_size, len(dataset)))
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    print(f"Training set size: {len(train_dataset)}")
    print(f"Test set size: {len(test_dataset)}")

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        num_workers=config.get('num_workers', 0),
        persistent_workers=False,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        num_workers=config.get('num_workers', 0),
        persistent_workers=False,
        shuffle=False,
    )
    
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of test batches: {len(test_loader)}")
    
    return train_dataset, test_dataset, train_loader, test_loader


def create_fake_datasets(config):
    """
    Create minimal datasets for debugging/overfitting tests.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        tuple: (fake_train_dataset, fake_test_dataset, fake_train_loader, fake_test_loader)
    """
    # Load the full dataset first
    dataset = SegmentedTokensOnDisk(
        config['manifest_file'], 
        cache_shards=config['cache_shards']
    )
    
    # Create single-sample datasets for overfitting tests
    fake_train_dataset = torch.utils.data.Subset(dataset, [0])
    fake_test_dataset = torch.utils.data.Subset(dataset, [0])

    print(f"Fake training set size: {len(fake_train_dataset)}")
    print(f"Fake test set size: {len(fake_test_dataset)}")

    # Create fake dataloaders
    fake_train_loader = torch.utils.data.DataLoader(
        fake_train_dataset,
        batch_size=1,
        num_workers=0,
        persistent_workers=False,
    )
    fake_test_loader = torch.utils.data.DataLoader(
        fake_test_dataset,
        batch_size=1,
        num_workers=0,
        persistent_workers=False,
    )
    
    return fake_train_dataset, fake_test_dataset, fake_train_loader, fake_test_loader
