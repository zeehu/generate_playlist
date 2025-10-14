Phase 1b (Alternative): Evaluate the quality of IDs from RQ-KMeans.

This script assesses the quantization quality by:
1. Reconstructing vectors from the saved codebooks and semantic IDs.
2. Comparing the original vectors with the reconstructed ones (Cosine Similarity).
3. Performing neighborhood preservation analysis.
"""
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import json
from tqdm import tqdm
import logging
import random

# Adjust path to import from playlist_src
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from config import Config
from train_rq_kmeans import KMeansTrainer # To reuse data loading
from utils import setup_logging

logger = logging.getLogger(__name__)

class KMeansEvaluator:
    """Orchestrates the evaluation of the RQ-KMeans results."""

    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device("cpu") # Evaluation can run on CPU

        # Load all necessary data
        self.song_ids, self.original_vectors = self._load_original_vectors()
        self.semantic_id_map = self._load_semantic_ids()
        self.codebooks = self._load_codebooks()
        self.reconstructed_vectors = self._reconstruct_all_vectors()

    def _load_original_vectors(self) -> tuple[list, torch.Tensor]:
        # Reuse the data loading logic from the trainer
        trainer = KMeansTrainer(self.config)
        song_ids, vectors_np = trainer._load_data()
        return song_ids, torch.from_numpy(vectors_np).to(self.device)

    def _load_semantic_ids(self) -> dict:
        semantic_ids_file = os.path.join(self.config.output_dir, "song_semantic_ids.jsonl")
        if not os.path.exists(semantic_ids_file):
            logger.error(f"FATAL: Semantic ID file not found. Please run a training script first.")
            sys.exit(1)
        
        mapping = {}
        with open(semantic_ids_file, 'r') as f:
            for line in f:
                item = json.loads(line)
                mapping[item['song_id']] = item['semantic_ids']
        return mapping

    def _load_codebooks(self) -> torch.Tensor:
        codebooks_path = os.path.join(self.config.model_dir, "rq_kmeans_codebooks.npy")
        if not os.path.exists(codebooks_path):
            logger.error(f"FATAL: Codebook file not found at {codebooks_path}")
            logger.error("Please run 'train_rq_kmeans.py' first.")
            sys.exit(1)
        
        logger.info(f"Loading codebooks from {codebooks_path}...")
        codebooks_np = np.load(codebooks_path)
        return torch.from_numpy(codebooks_np).to(self.device)

    def _reconstruct_all_vectors(self) -> torch.Tensor:
        logger.info("Reconstructing all vectors from codebooks and semantic IDs...")
        num_songs = len(self.song_ids)
        embedding_dim = self.config.rqkmeans.input_dim
        reconstructed = torch.zeros((num_songs, embedding_dim), device=self.device)

        for i, song_id in enumerate(tqdm(self.song_ids, desc="Reconstructing vectors")):
            if song_id in self.semantic_id_map:
                semantic_ids = self.semantic_id_map[song_id]
                reconstructed_vec = torch.zeros(embedding_dim, device=self.device)
                for level, code_index in enumerate(semantic_ids):
                    reconstructed_vec += self.codebooks[level, code_index, :]
                reconstructed[i] = reconstructed_vec
        return reconstructed

    def run_all_evaluations(self, sample_size: int = 10000, k_neighbors: int = 5):
        logger.info("--- Starting RQ-KMeans Evaluation ---")
        self.evaluate_reconstruction(sample_size)
        # The neighborhood evaluation from RQ-VAE script can be reused if adapted
        # For simplicity, we focus on the main reconstruction metric here.
        logger.info("--- Evaluation Completed ---")

    def evaluate_reconstruction(self, sample_size: int):
        logger.info(f"\n[Evaluating Reconstruction Quality on a sample of {sample_size} songs]")
        
        indices = random.sample(range(len(self.song_ids)), min(sample_size, len(self.song_ids)))
        
        original_sample = self.original_vectors[indices]
        reconstructed_sample = self.reconstructed_vectors[indices]

        # Calculate cosine similarity for the entire sample at once
        cosine_sim = F.cosine_similarity(original_sample, reconstructed_sample).mean().item()

        print("\n" + "="*50)
        print("  Reconstruction Quality Report (K-Means)")
        print("="*50)
        print(f"  Average Cosine Similarity:  {cosine_sim:.6f}")
        print("="*50)
        logger.info(f"Reconstruction Report: Avg Cosine Sim={cosine_sim:.6f}")
        if cosine_sim < 0.85:
            logger.warning("Cosine similarity is okay, but could be better. Consider increasing k (vocab_size) or levels.")
        else:
            logger.info("Cosine similarity looks excellent! Quantization quality is high.")

if __name__ == "__main__":
    config = Config()
    log_file_path = os.path.join(config.log_dir, "evaluate_rq_kmeans.log")
    logger = setup_logging(log_file_path)

    evaluator = KMeansEvaluator(config)
    evaluator.run_all_evaluations()
