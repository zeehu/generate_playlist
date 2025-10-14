


import os
import sys
import numpy as np
import json
import logging
from tqdm import tqdm
import pandas as pd

try:
    import faiss
except ImportError:
    print("Faiss library not found. Please install it first.")
    print("CPU version: pip install faiss-cpu")
    print("GPU version: pip install faiss-gpu")
    sys.exit(1)

# Adjust path to import from playlist_src
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from config import Config
from utils import setup_logging

logger = logging.getLogger(__name__)

class KMeansTrainer:
    """Orchestrates the RQ-KMeans training and semantic ID generation."""

    def __init__(self, config: Config):
        self.config = config
        self.kmeans_config = config.rqkmeans
        self.data_config = config.data
        self.use_gpu = torch.cuda.is_available() and hasattr(faiss, 'StandardGpuResources')
        if self.use_gpu:
            logger.info("Faiss GPU support detected. Using GPU for K-Means.")
        else:
            logger.info("Using CPU for K-Means. This might be slow for large datasets.")

    def run(self):
        logger.info("--- Starting Phase 1 (Alternative): RQ-KMeans Training ---")

        # 1. Load data
        song_ids, vectors = self._load_data()

        # 2. Train RQ-KMeans and get indices
        indices_per_level, codebooks = self._train_rq_kmeans(vectors)

        # 3. Save results
        self._save_results(song_ids, indices_per_level, codebooks)

        logger.info("--- RQ-KMeans Phase 1 Completed Successfully ---")

    def _load_data(self) -> tuple[list, np.ndarray]:
        vector_file = self.config.rqvae.song_vector_file # Path is in rqvae config
        logger.info(f"Loading song vectors from {vector_file}...")
        try:
            # Using pandas is generally faster for large CSVs
            df = pd.read_csv(vector_file, header=None, dtype={0: str})
            song_ids = df[0].tolist()
            vectors = df.iloc[:, 1:].to_numpy(dtype='float32')
            logger.info(f"Successfully loaded {len(song_ids)} song vectors.")
            return song_ids, vectors
        except FileNotFoundError:
            logger.error(f"FATAL: Song vector file not found at {vector_file}")
            logger.error("Please update the path in 'playlist_src/config.py'.")
            sys.exit(1)

    def _train_rq_kmeans(self, vectors: np.ndarray) -> tuple[np.ndarray, list]:
        logger.info("Starting residual quantization training with K-Means...")
        
        residuals = vectors.copy()
        all_indices = []
        all_codebooks = []
        
        for level in range(self.kmeans_config.levels):
            logger.info(f"--- Training Level {level + 1}/{self.kmeans_config.levels} ---")
            
            d = self.kmeans_config.input_dim
            k = self.kmeans_config.vocab_size
            seed = self.kmeans_config.seed + level # Use different seed for each level

            # Setup K-Means. Use GPU if available.
            kmeans = faiss.Kmeans(d, k, niter=20, verbose=True, seed=seed, gpu=self.use_gpu)
            
            # Train on the current residuals
            kmeans.train(residuals)
            all_codebooks.append(kmeans.centroids)

            # Assign indices to the residuals
            _D, I = kmeans.index.search(residuals, 1)
            all_indices.append(I.flatten())

            # Update residuals for the next level
            # new_residuals = residuals - assigned_centroids
            assigned_centroids = kmeans.centroids[I.flatten()]
            residuals = residuals - assigned_centroids

        # Shape: (num_levels, num_songs) -> (num_songs, num_levels)
        final_indices = np.stack(all_indices, axis=1)
        return final_indices, all_codebooks

    def _save_results(self, song_ids: list, indices: np.ndarray, codebooks: list):
        logger.info("Saving semantic IDs and codebooks...")

        # 1. Save the semantic IDs to a JSONL file
        # This will overwrite the file from the VAE method, which is intended
        output_path = os.path.join(self.config.output_dir, "song_semantic_ids.jsonl")
        with open(output_path, 'w') as f:
            for i, song_id in enumerate(tqdm(song_ids, desc="Saving Semantic IDs")):
                item = {
                    'song_id': song_id,
                    'semantic_ids': indices[i].tolist()
                }
                f.write(json.dumps(item) + '\n')
        logger.info(f"Semantic IDs saved to {output_path}")

        # 2. Save the learned codebooks
        codebooks_path = os.path.join(self.config.model_dir, "rq_kmeans_codebooks.npy")
        np.save(codebooks_path, np.array(codebooks))
        logger.info(f"Codebooks saved to {codebooks_path}")

if __name__ == "__main__":
    # Import torch here just for the cuda check, to keep dependencies minimal if not using torch
    import torch
    config = Config()
    log_file_path = os.path.join(config.log_dir, "phase1_train_rq_kmeans.log")
    logger = setup_logging(log_file_path)

    # Check if the song vector file path is configured
    if config.rqvae.song_vector_file == "path/to/your/song_vectors.csv":
        logger.error("="*80)
        logger.error("FATAL: Please edit 'playlist_src/config.py' and set the")
        logger.error("'song_vector_file' path.")
        logger.error("="*80)
        sys.exit(1)

    trainer = KMeansTrainer(config)
    trainer.run()
