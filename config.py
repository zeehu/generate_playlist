
import os
from dataclasses import dataclass, field
from typing import List

@dataclass
class DataConfig:
    """Configuration for data paths and processing."""
    # --- Paths to be filled by user ---
    song_info_file: str = "./data/gen_song_info.csv"
    playlist_info_file: str = "./data/gen_playlist_info.csv"
    playlist_songs_file: str = "./data/gen_playlist_song.csv"
    
    # Generated files (no need to change)
    semantic_ids_file: str = "outputs/song_semantic_ids.jsonl"
    
    # Splitting ratio
    train_split_ratio: float = 0.98
    val_split_ratio: float = 0.01
    # Test split is implicitly 1.0 - train - val


@dataclass
class SongRQVAEConfig:
    """Configuration for Song RQ-VAE training."""
    # --- Path to be filled by user ---
    song_vector_file: str = "./data/song_vectors.csv"
    
    # Model parameters
    input_dim: int = 100
    vocab_size: int = 1024
    levels: int = 2
    dim: int = 256
    hidden_dim: int = 512
    num_layers: int = 4
    dropout: float = 0.1
    commitment_cost: float = 0.25
    
    # Training parameters
    epochs: int = 50
    batch_size: int = 128
    learning_rate: float = 1e-3

@dataclass
class PlaylistTIGERConfig:
    """Configuration for the TIGER model for playlist generation."""
    model_name: str = "t5-small"
    max_input_length: int = 128  # Max length for playlist title/tags
    max_target_length: int = 256 # Max length for song sequence (e.g., 128 songs * 2 tokens/song)
    
    # Training parameters
    num_train_epochs: int = 10
    per_device_train_batch_size: int = 32
    per_device_eval_batch_size: int = 8
    learning_rate: float = 5e-4
    warmup_steps: int = 500
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    fp16: bool = True


@dataclass
class EvalConfig:
    """Configuration for offline evaluation."""
    # --- Path for offline metric scripts ---
    local_metrics_path: str = "./local_metrics"


@dataclass
class SongRQKMeansConfig:
    """Configuration for Song RQ-KMeans training."""
    # Inherits song_vector_file from DataConfig
    input_dim: int = 100
    vocab_size: int = 1024  # This is 'k' in k-means
    levels: int = 2         # Number of residual levels
    seed: int = 42


@dataclass
class Config:
    """Main configuration for the project."""
    # Sub-configurations
    data: DataConfig = field(default_factory=DataConfig)
    rqvae: SongRQVAEConfig = field(default_factory=SongRQVAEConfig)
    rqkmeans: SongRQKMeansConfig = field(default_factory=SongRQKMeansConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    tiger: PlaylistTIGERConfig = field(default_factory=PlaylistTIGERConfig)
    
    # Common paths
    output_dir: str = "outputs"
    model_dir: str = "models"
    log_dir: str = "logs"
    
    # System settings
    device: str = "cuda"
    seed: int = 42
    num_workers: int = 2
    
    def __post_init__(self):
        # Create directories if they don't exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
