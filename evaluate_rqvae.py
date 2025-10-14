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
import csv
from typing import Dict, List, Tuple

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
        self.song_info_map = self._load_song_info()

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

    def _load_song_info(self) -> Dict[str, Dict[str, str]]:
        song_info_file = self.config.data.song_info_file
        if not os.path.exists(song_info_file):
            logger.warning(f"Song info file not found at {song_info_file}. Song names will not be displayed.")
            return {}

        logger.info(f"Loading song info from {song_info_file}...")
        mapping = {}
        try:
            with open(song_info_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                # Skip header if exists
                first_row = next(reader)
                if 'mixsongid' not in first_row[0]:
                    self._process_song_info_row(first_row, mapping)
                
                for row in reader:
                    self._process_song_info_row(row, mapping)
            logger.info(f"Loaded info for {len(mapping)} songs.")
        except Exception as e:
            logger.error(f"Error reading song info file: {e}")
            return {}
        return mapping

    def _process_song_info_row(self, row: List[str], mapping: Dict):
        if len(row) >= 3:
            song_id, song_name, singer_name = row[0], row[1], row[2]
            mapping[song_id] = {"name": song_name, "singer": singer_name}

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
                
                reconstructed_vector, _, _ = self.model(original_vector)
                
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

        subset_indices = random.sample(range(len(self.dataset)), min(sample_size, len(self.dataset)))
        original_vectors = torch.stack([self.dataset[i]['features'] for i in subset_indices]).to(self.device)
        song_ids = [self.dataset[i]['song_id'] for i in subset_indices]

        with torch.no_grad():
            encoded_vectors = self.model.encoder(original_vectors)
            quantized_vectors, _, _ = self.model.quantizer(encoded_vectors)

        seed_indices = random.sample(range(len(subset_indices)), 3)

        print("\n" + "="*70)
        print("  Neighborhood Preservation Report")
        print("="*70)
        
        for seed_idx in seed_indices:
            seed_song_id = song_ids[seed_idx]
            seed_info = self.song_info_map.get(seed_song_id)
            seed_display = f"{seed_info['name']} - {seed_info['singer']}" if seed_info else seed_song_id
            print(f"\n--- Analysis for Seed Song: {seed_display} ---")

            # Find neighbors in original vector space
            seed_vector_orig = original_vectors[seed_idx].unsqueeze(0)
            sim_orig = F.cosine_similarity(seed_vector_orig, original_vectors)
            top_k_orig_indices = torch.topk(sim_orig, k + 1).indices[1:]
            
            print(f"  Top {k} Neighbors in ORIGINAL space:")
            for idx in top_k_orig_indices:
                neighbor_id = song_ids[idx.item()]
                info = self.song_info_map.get(neighbor_id)
                display_name = f"{info['name']} - {info['singer']}" if info else neighbor_id
                similarity = sim_orig[idx.item()].item()
                print(f"    - {display_name} (Similarity: {similarity:.4f})")

            # Find neighbors in quantized vector space
            seed_vector_quant = quantized_vectors[seed_idx].unsqueeze(0)
            sim_quant = F.cosine_similarity(seed_vector_quant, quantized_vectors)
            top_k_quant_indices = torch.topk(sim_quant, k + 1).indices[1:]
            
            print(f"\n  Top {k} Neighbors in QUANTIZED space:")
            for idx in top_k_quant_indices:
                neighbor_id = song_ids[idx.item()]
                info = self.song_info_map.get(neighbor_id)
                display_name = f"{info['name']} - {info['singer']}" if info else neighbor_id
                similarity = sim_quant[idx.item()].item()
                print(f"    - {display_name} (Similarity: {similarity:.4f})")
            
            orig_set = set(top_k_orig_indices.cpu().numpy())
            quant_set = set(top_k_quant_indices.cpu().numpy())
            overlap = len(orig_set.intersection(quant_set))
            print(f"\n  => Overlap between two neighbor lists: {overlap}/{k}")
            print("-"*70)

if __name__ == "__main__":
    config = Config()
    log_file_path = os.path.join(config.log_dir, "phase1b_evaluate_rqvae.log")
    logger = setup_logging(log_file_path)

    if config.data.song_info_file == "path/to/your/gen_song_info.csv":
        logger.warning("="*80)
        logger.warning("提示: 您还未在 'playlist_src/config.py' 中配置 'song_info_file' 的路径。")
        logger.warning("歌曲的详细信息（歌名、歌手）将不会被显示。")
        logger.warning("="*80)

    evaluator = Evaluator(config)
    evaluator.run_all_evaluations()
