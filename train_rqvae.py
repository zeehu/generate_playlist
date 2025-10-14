"""
Phase 1: Train RQ-VAE for Song Semantic ID Generation.
"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from tqdm import tqdm
import logging
import csv

# Adjust path to import from playlist_src
# This allows running the script from the root directory
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from config import Config
from rqvae import RQVAE
from utils import set_seed, setup_logging

logger = logging.getLogger(__name__)

class SongVectorDataset(Dataset):
    """Dataset for loading song vectors from a CSV file."""
    
    def __init__(self, csv_path: str, vector_dim: int):
        self.song_vectors = []
        self.song_ids = []
        logger.info(f"Loading song vectors from {csv_path}...")
        
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for i, row in enumerate(tqdm(reader, desc="Loading song data")):
                    if not row: continue
                    self.song_ids.append(row[0])
                    vector = np.array(row[1:], dtype=np.float32)
                    
                    if len(vector) != vector_dim:
                        logger.warning(f"Skipping row {i+1} due to incorrect vector dimension. Expected {vector_dim}, got {len(vector)}.")
                        self.song_ids.pop()
                        continue
                        
                    self.song_vectors.append(vector)
        except FileNotFoundError:
            logger.error(f"FATAL: Song vector file not found at {csv_path}")
            logger.error("Please update the 'song_vector_file' path in 'playlist_src/config.py'.")
            sys.exit(1)

        if not self.song_vectors:
            logger.error(f"FATAL: No data loaded from {csv_path}. Please check the file format and path.")
            sys.exit(1)

        logger.info(f"Successfully loaded {len(self.song_vectors)} song vectors.")

    def __len__(self):
        return len(self.song_vectors)
    
    def __getitem__(self, idx):
        return {
            'song_id': self.song_ids[idx],
            'features': torch.FloatTensor(self.song_vectors[idx])
        }

class Trainer:
    """Orchestrates the RQ-VAE training and semantic ID generation for songs."""
    
    def __init__(self, config: Config):
        self.config = config.rqvae
        self.system_config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        set_seed(config.seed)

    def run(self):
        """Main entry point for the training pipeline."""
        logger.info("--- Starting Phase 1: Song RQ-VAE Training ---")
        
        # 1. Prepare Data
        dataloader = self._prepare_data()
        
        # 2. Create Model
        model = self._create_model()
        
        # 3. Optimizer and Scheduler
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.epochs)
        
        # 4. Training Loop
        self._train_loop(model, dataloader, optimizer, scheduler)
        
        # 5. Generate and Save Semantic IDs
        model_path = os.path.join(self.system_config.model_dir, "song_rqvae_best.pt")
        self._generate_semantic_ids(model_path)
        
        logger.info("--- Phase 1 Completed Successfully ---")

    def _prepare_data(self) -> DataLoader:
        dataset = SongVectorDataset(self.config.song_vector_file, self.config.input_dim)
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.system_config.num_workers,
            pin_memory=True
        )

    def _create_model(self) -> RQVAE:
        logger.info("Creating RQ-VAE model...")
        return RQVAE(
            input_dim=self.config.input_dim,
            num_embeddings=self.config.vocab_size,
            embedding_dim=self.config.dim,
            hidden_dim=self.config.hidden_dim,
            num_levels=self.config.levels,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
            commitment_cost=self.config.commitment_cost
        ).to(self.device)

    def _train_loop(self, model, dataloader, optimizer, scheduler):
        best_loss = float('inf')
        for epoch in range(1, self.config.epochs + 1):
            model.train()
            total_loss = 0
            pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{self.config.epochs}")
            
            for batch in pbar:
                features = batch['features'].to(self.device)
                reconstructed, vq_loss, _ = model(features)
                recon_loss = nn.MSELoss()(reconstructed, features)
                loss = recon_loss + vq_loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix({'Loss': f"{loss.item():.4f}"})
            
            avg_loss = total_loss / len(dataloader)
            logger.info(f"Epoch {epoch} Avg Loss: {avg_loss:.4f}")
            scheduler.step()
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                model_path = os.path.join(self.system_config.model_dir, "song_rqvae_best.pt")
                torch.save(model.state_dict(), model_path)
                logger.info(f"Saved best model with loss {avg_loss:.4f}")

    def _generate_semantic_ids(self, model_path: str):
        logger.info("Generating semantic IDs...")
        model = self._create_model()
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        
        dataset = SongVectorDataset(self.config.song_vector_file, self.config.input_dim)
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False)

        semantic_ids_data = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Generating IDs"):
                features = batch['features'].to(self.device)
                song_ids = batch['song_id']
                indices = model.get_semantic_ids(features)
                indices_per_song = torch.stack(indices).T

                for i, song_id in enumerate(song_ids):
                    semantic_ids_data.append({
                        'song_id': song_id,
                        'semantic_ids': indices_per_song[i].cpu().tolist()
                    })

        output_path = os.path.join(self.system_config.output_dir, "song_semantic_ids.jsonl")
        with open(output_path, 'w') as f:
            for item in semantic_ids_data:
                f.write(json.dumps(item) + '\n')
        
        logger.info(f"Semantic IDs for {len(semantic_ids_data)} songs saved to {output_path}")

if __name__ == "__main__":
    config = Config()
    log_file_path = os.path.join(config.log_dir, "phase1_train_rqvae.log")
    logger = setup_logging(log_file_path)
    
    if config.rqvae.song_vector_file == "path/to/your/song_vectors.csv":
        logger.error("="*80)
        logger.error("FATAL: Please edit 'playlist_src/config.py' and set the")
        logger.error("'song_vector_file' path in the SongRQVAEConfig class.")
        logger.error("="*80)
        sys.exit(1)

    trainer = Trainer(config)
    trainer.run()
