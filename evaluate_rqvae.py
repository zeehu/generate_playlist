Phase 1b: Evaluate the quality of generated Song Semantic IDs.

This script assesses the trained RQ-VAE model by measuring:
1. Reconstruction Quality: How well can we reconstruct the original song vector from the semantic ID?
   - Metrics: Mean Squared Error (MSE) and Cosine Similarity.
2. Neighborhood Preservation: Do songs that are neighbors in the original vector space
   remain neighbors after quantization?

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
from rqvae import RQVAE
from train_rqvae import SongVectorDataset # Reuse the dataset class
from utils import setup_logging

logger = logging.getLogger(__name__)

class Evaluator:
    """Orchestrates the evaluation of the trained RQ-VAE model."""

    def __init__(self, config: Config):
        self.config = config
        self.rqvae_config = config.rqvae
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model()
        self.dataset = SongVectorDataset(self.rqvae_config.song_vector_file, self.rqvae_config.input_dim)

    def _load_model(self) -> RQVAE:
        model_path = os.path.join(self.config.model_dir, "song_rqvae_best.pt")
        if not os.path.exists(model_path):
            logger.error(f"FATAL: Model file not found at {model_path}")
            logger.error("Please run 'playlist_src/train_rqvae.py' first.")
            sys.exit(1)
        
        logger.info(f"Loading model from {model_path}...")
        model = RQVAE(
            input_dim=self.rqvae_config.input_dim,
            num_embeddings=self.rqvae_config.vocab_size,
            embedding_dim=self.rqvae_config.dim,
            hidden_dim=self.rqvae_config.hidden_dim,
            num_levels=self.rqvae_config.levels,
            num_layers=self.rqvae_config.num_layers
        ).to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        return model

    def run_all_evaluations(self, sample_size: int = 10000, k_neighbors: int = 5):
        logger.info("--- Starting RQ-VAE Evaluation ---")
        
        # 1. Reconstruction Quality Evaluation
        self.evaluate_reconstruction(sample_size)
        
        # 2. Neighborhood Preservation Evaluation
        self.evaluate_neighborhood(sample_size, k_neighbors)
        
        logger.info("--- Evaluation Completed ---")

    def evaluate_reconstruction(self, sample_size: int):
        logger.info(f"\n[1. Evaluating Reconstruction Quality on a sample of {sample_size} songs]")
        
        total_mse = 0
        total_cosine_sim = 0
        count = 0

        indices = random.sample(range(len(self.dataset)), min(sample_size, len(self.dataset)))
        
        with torch.no_grad():
            for i in tqdm(indices, desc="Calculating Reconstruction Metrics"):
                data = self.dataset[i]
                original_vector = data['features'].unsqueeze(0).to(self.device)
                
                # Get the reconstructed vector
                reconstructed_vector, _, _ = self.model(original_vector)
                
                # Calculate metrics
                total_mse += F.mse_loss(reconstructed_vector, original_vector).item()
                total_cosine_sim += F.cosine_similarity(reconstructed_vector, original_vector).item()
                count += 1

        avg_mse = total_mse / count
        avg_cosine_sim = total_cosine_sim / count

        print("\n" + "="*50)
        print("  Reconstruction Quality Report")
        print("="*50)
        print(f"  Average MSE:              {avg_mse:.6f}")
        print(f"  Average Cosine Similarity:  {avg_cosine_sim:.6f}")
        print("="*50)
        logger.info(f"Reconstruction Report: Avg MSE={avg_mse:.6f}, Avg Cosine Sim={avg_cosine_sim:.6f}")
        if avg_cosine_sim < 0.7:
            logger.warning("Cosine similarity is low. The semantic IDs may not retain enough information.")
        else:
            logger.info("Cosine similarity looks good. Semantic IDs seem to retain information well.")

    def evaluate_neighborhood(self, sample_size: int, k: int):
        logger.info(f"\n[2. Evaluating Neighborhood Preservation on a sample of {sample_size} songs]")

        # Create a random subset of the data for feasible nearest neighbor search
        subset_indices = random.sample(range(len(self.dataset)), min(sample_size, len(self.dataset)))
        
        original_vectors = torch.stack([self.dataset[i]['features'] for i in subset_indices]).to(self.device)
        song_ids = [self.dataset[i]['song_id'] for i in subset_indices]

        # Get quantized vectors for the subset
        with torch.no_grad():
            encoded_vectors = self.model.encoder(original_vectors)
            quantized_vectors, _, _ = self.model.quantizer(encoded_vectors)

        # Pick a few random songs from the subset to be our "seed" songs
        seed_indices = random.sample(range(len(subset_indices)), 3)

        print("\n" + "="*70)
        print("  Neighborhood Preservation Report")
        print("="*70)
        
        for seed_idx in seed_indices:
            seed_song_id = song_ids[seed_idx]
            print(f"\n--- Analysis for Seed Song: {seed_song_id} ---")

            # Find neighbors in original vector space
            seed_vector_orig = original_vectors[seed_idx].unsqueeze(0)
            sim_orig = F.cosine_similarity(seed_vector_orig, original_vectors)
            top_k_orig_indices = torch.topk(sim_orig, k + 1).indices[1:] # +1 and [1:] to exclude the song itself
            
            print(f"  Top {k} Neighbors in ORIGINAL space:")
            for idx in top_k_orig_indices:
                neighbor_id = song_ids[idx.item()]
                similarity = sim_orig[idx.item()].item()
                print(f"    - {neighbor_id} (Similarity: {similarity:.4f})")

            # Find neighbors in quantized vector space
            seed_vector_quant = quantized_vectors[seed_idx].unsqueeze(0)
            sim_quant = F.cosine_similarity(seed_vector_quant, quantized_vectors)
            top_k_quant_indices = torch.topk(sim_quant, k + 1).indices[1:]
            
            print(f"\n  Top {k} Neighbors in QUANTIZED space:")
            for idx in top_k_quant_indices:
                neighbor_id = song_ids[idx.item()]
                similarity = sim_quant[idx.item()].item()
                print(f"    - {neighbor_id} (Similarity: {similarity:.4f})")
            
            # Calculate overlap
            orig_set = set(top_k_orig_indices.cpu().numpy())
            quant_set = set(top_k_quant_indices.cpu().numpy())
            overlap = len(orig_set.intersection(quant_set))
            print(f"\n  => Overlap between two neighbor lists: {overlap}/{k}")
            print("-"*70)

if __name__ == "__main__":
    config = Config()
    log_file_path = os.path.join(config.log_dir, "phase1b_evaluate_rqvae.log")
    logger = setup_logging(log_file_path)

    evaluator = Evaluator(config)
    evaluator.run_all_evaluations()
