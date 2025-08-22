import numpy as np
import torch, torchaudio
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import os
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import torch, os, bisect, json
from pathlib import Path

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





class MuLawEncoding:
    def __init__(self, quantization_channels: int = 256):
        self.Q = quantization_channels
        self.enc = torchaudio.transforms.MuLawEncoding(self.Q)
        self.dec = torchaudio.transforms.MuLawDecoding(self.Q)

    @torch.no_grad()
    def mu_law_encode(self, x: torch.Tensor) -> torch.LongTensor:
        # x: [..., T] float in [-1, 1]
        return self.enc(x)

    @torch.no_grad()
    def mu_law_decode(self, q: torch.Tensor) -> torch.Tensor:
        # q: [..., T] long/int in [0, Q-1]
        return self.dec(q)

# Save codec once and use everywhere.
codec = MuLawEncoding(256)





config = {
    "batch_size": 4,
    "num_workers": 2,  # CHANGED: Use 0 for debugging, multiprocessing can cause issues
    "pin_memory": True,
    "mu": 255,
    "sr": 16000,
    "trim_silence_thresh": 1e-3,
    "window_size": 32001,  # ~2 seconds at 16kHz

    # Wavenet architecture related.
    "residual_channels": 64,
    "skip_channels": 256,
    "output_dim": 256,
    "n_layers": 10,
    "n_blocks": 5,
    "kernel_size": 2,
    'hop_size': 16000,
}





class AudioProcessor:
    def __init__(self):
        self.mu_law_encoding = codec
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
                hop_size=config.get('hop_size', 16000)     # e.g., 16000 for 50% overlap @32000
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











# segmented_tokens_io.py

import os, json, bisect
from pathlib import Path
from typing import Optional, Tuple, List

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from collections import deque
from pathlib import Path
import json
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# ------------------------------------------------------------
# REQUIREMENT: You already defined AudioProcessor elsewhere.
# It must expose: collate_fn([dataset[i]]) -> (x_segments, y_segments)
# where each is Long tensor [Si, T] (Si = #segments from item i).
# ------------------------------------------------------------

# ---------- Builder (parallel, RAM; fast if dataset fits in RAM) ----------

class _SegmentsPerItem(Dataset):
    """Wraps a base dataset so each __getitem__ returns all (x,y) segments for that item."""
    def __init__(self, base_ds, audio_processor):
        self.ds = base_ds
        self.proc = audio_processor

    def __len__(self): return len(self.ds)

    def __getitem__(self, i):
        x, y = self.proc.collate_fn([self.ds[i]])  # x:[Si,T], y:[Si,T] or empty
        return x, y

def _cat_collate(batch):
    xs = [b[0] for b in batch if b[0].numel() > 0]
    ys = [b[1] for b in batch if b[1].numel() > 0]
    if not xs: return None
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)

def build_segmented_dataset_fast(base_dataset,
                                 audio_processor,
                                 num_workers: int = 8,
                                 batch_items: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      audio_x: [N, T] Long
      audio_y: [N, T] Long
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

# ---------- Saver (sharded) & Manifest ----------

def save_segmented_dataset_sharded(audio_x: torch.Tensor,
                                         audio_y: torch.Tensor,
                                         out_dir: str,
                                         shard_size: int = 512):
    """
    Saves shards as uint8 to cut memory/disk by 8×.
    audio_x/audio_y are Long in [0,255]; we store as uint8 and upcast on load.
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


# ---------- Streaming writer (low-RAM; writes shards on the fly) ----------

def stream_preprocess_to_shards(base_dataset,
                                audio_processor,
                                out_dir: str,
                                shard_size: int = 10000,
                                num_workers: int = 8,
                                batch_items: int = 32):
    """
    Builds (x,y) token segments in parallel but writes shards incrementally to keep RAM low.
    Each shard has exactly <= shard_size samples. Handles partial consumption of a batch.
    Returns path to manifest.json.
    """
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    # --- dataset wrappers (same as before) ---
    class _SegmentsPerItem(torch.utils.data.Dataset):
        def __init__(self, base_ds, proc):
            self.ds = base_ds
            self.proc = proc
        def __len__(self): return len(self.ds)
        def __getitem__(self, i):
            return self.proc.collate_fn([self.ds[i]])  # (x:[Si,T], y:[Si,T])

    def _cat_collate(batch):
        xs = [b[0] for b in batch if b[0].numel() > 0]
        ys = [b[1] for b in batch if b[1].numel() > 0]
        if not xs:
            return None
        x = torch.cat(xs, dim=0).contiguous()
        y = torch.cat(ys, dim=0).contiguous()
        return x, y

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
# ---------- Loader (direct use in training) ----------

class SegmentedTokensOnDisk(torch.utils.data.Dataset):
    def __init__(self, manifest_path, root=None, cache_shards=False):
        import json, torch
        from pathlib import Path
        mp = Path(manifest_path)
        man = json.loads(mp.read_text())
        self.root = mp.parent if root is None else Path(root)
        self.files = [self.root / s for s in man["shards"]]
        self.sizes = man["shard_sizes"]
        self.cum = []
        c = 0
        for s in self.sizes:
            c += s
            self.cum.append(c)
        self.N = man["num_samples"]
        self.T = man["seq_len"]
        self.cache = {} if cache_shards else None
        self.stored_dtype = man.get("stored_dtype", "long")

    def __len__(self): return self.N

    def _loc(self, idx):
        import bisect
        s_idx = bisect.bisect_right(self.cum, idx)
        base = 0 if s_idx == 0 else self.cum[s_idx-1]
        off = idx - base
        return s_idx, off

    def __getitem__(self, idx):
        s_idx, off = self._loc(idx)
        if self.cache is not None and s_idx in self.cache:
            shard = self.cache[s_idx]
        else:
            shard = torch.load(self.files[s_idx], map_location="cpu")
            if self.cache is not None:
                self.cache[s_idx] = shard

        if self.stored_dtype == "uint8":
            x = shard["x_u8"][off].to(torch.long)  # upcast once
            y = shard["y_u8"][off].to(torch.long)
        else:
            x = shard["x"][off]
            y = shard["y"][off]
        return x, y

# ---------- Example usage ----------







ap = AudioProcessor()
out_dir = "./segmented_tokens"          # where to save shards
shard_size = 10000

manifest = stream_preprocess_to_shards(
    base_dataset=dataset,
    audio_processor=ap,
    out_dir=out_dir,
    shard_size=shard_size,
    num_workers=8,
    batch_items=32,
)
print("Manifest written to:", manifest)





# Load later for training:
ds = SegmentedTokensOnDisk("segmented_tokens/manifest.json", cache_shards=False)

# Start conservative:
dataset_loader = torch.utils.data.DataLoader(
    ds,
    batch_size=4,
    shuffle=True,
    num_workers=0,          # first run; ensure it works
    pin_memory=False,       # you can flip to True later if needed
)





for batch in dataset_loader:
    print(f"x: {batch[0].shape}, y: {batch[1].shape}")
    break





import json, torch
from pathlib import Path

def scan_for_oob(manifest_path):
    mp = Path(manifest_path)
    man = json.loads(mp.read_text())
    root = mp.parent
    bads = []

    for sid, fname in enumerate(man["shards"]):
        obj = torch.load(root / fname, map_location="cpu")
        # support both uint8-v2 and long-v1
        if "x_u8" in obj:
            X = obj["x_u8"].to(torch.int16)  # cheap
            Y = obj["y_u8"].to(torch.int16)
        else:
            X = obj["x"]; Y = obj["y"]

        bad_x = (X < 0) | (X > 255)
        bad_y = (Y < 0) | (Y > 255)
        if bad_x.any() or bad_y.any():
            ix = bad_x.nonzero(as_tuple=False)
            iy = bad_y.nonzero(as_tuple=False)
            print(f"[shard {sid}] bad X: {ix.shape[0]}, bad Y: {iy.shape[0]}")
            # record first few
            for n in ix[:5]:
                r,c = n.tolist()
                print("  X oob at sample", r, "pos", c, "val", int(X[r,c]))
                bads.append(("x", sid, r, c, int(X[r,c])))
            for n in iy[:5]:
                r,c = n.tolist()
                print("  Y oob at sample", r, "pos", c, "val", int(Y[r,c]))
                bads.append(("y", sid, r, c, int(Y[r,c])))
    return bads

bads = scan_for_oob("segmented_tokens/manifest.json")
print("Total OOB entries:", len(bads))



