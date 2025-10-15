"""
Phase 4 (Advanced): Evaluate the TIGER model using cluster-aware metrics.
This script expands predicted semantic IDs into sets of songs and calculates
set-based Precision, Recall, and F1-score against the ground truth.
"""
import os
import sys
import torch
from tqdm import tqdm
import logging
import numpy as np
import json
from typing import Dict, List, Tuple

from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator

# Adjust path to import from playlist_src
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from config import Config
from tiger_model import TIGERModel, TIGERTokenizer
from utils import setup_logging

logger = logging.getLogger(__name__)

class TestDataset(Dataset):
    """Dataset for loading the test portion of the corpus."""
    def __init__(self, data_path: str, tokenizer: TIGERTokenizer, max_input_len: int):
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.data = []
        logger.info(f"Loading test data from {data_path}...")
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 3:
                    glid, input_text, target_sequence = parts
                    self.data.append({"input_text": input_text, "target_sequence": target_sequence})
        logger.info(f"Loaded {len(self.data)} test samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_encoding = self.tokenizer.base_tokenizer(item["input_text"], max_length=self.max_input_len, truncation=True)
        return input_encoding

class ClusterEvaluator:
    """Orchestrates cluster-aware evaluation using Accelerate."""

    def __init__(self, config: Config):
        self.config = config
        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        self.model = self._load_model()
        self.semantic_to_song_map = self._create_reverse_map()

    def run_evaluation(self):
        logger.info("--- Starting Phase 4: Cluster-Aware Evaluation ---")
        tokenizer = self.model.tokenizer
        test_dataset = TestDataset(
            data_path=os.path.join(self.config.output_dir, "test.tsv"),
            tokenizer=tokenizer,
            max_input_len=self.config.tiger.max_input_length
        )

        def collate_fn(examples):
            return tokenizer.base_tokenizer.pad(examples, padding="longest", return_tensors="pt")

        test_dataloader = DataLoader(test_dataset, shuffle=False, collate_fn=collate_fn, batch_size=self.config.tiger.per_device_eval_batch_size)
        self.model, test_dataloader = self.accelerator.prepare(self.model, test_dataloader)

        all_predicted_token_ids = []
        logger.info("Starting generation on test set...")
        for batch in tqdm(test_dataloader, desc="Generating Predictions"):
            with torch.no_grad():
                unwrapped_model = self.accelerator.unwrap_model(self.model)
                generated_tokens = unwrapped_model.model.generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    max_new_tokens=self.config.tiger.max_target_length,
                    num_beams=3
                )
            padded_predictions = self.accelerator.pad_across_processes(generated_tokens, dim=1, pad_index=tokenizer.pad_token_id)
            gathered_predictions = self.accelerator.gather(padded_predictions)
            all_predicted_token_ids.extend(gathered_predictions.cpu().numpy())

        if self.accelerator.is_main_process:
            all_predicted_token_ids = all_predicted_token_ids[:len(test_dataset)]
            self._compute_cluster_metrics(test_dataset.data, all_predicted_token_ids)

        logger.info("--- Evaluation Completed ---")

    def _compute_cluster_metrics(self, test_data: List[Dict], predicted_token_ids: np.ndarray):
        logger.info("Computing cluster-aware evaluation metrics...")
        
        str_predictions = self.model.tokenizer.base_tokenizer.batch_decode(predicted_token_ids, skip_special_tokens=True)

        total_precision, total_recall, total_f1 = [], [], []

        for i, item in enumerate(tqdm(test_data, desc="Calculating Metrics")):
            # 1. Get the ground truth set of song IDs
            reference_song_ids = set(self._decode_semantic_string(item["target_sequence"]))
            if not reference_song_ids:
                continue

            # 2. Get the predicted set of song IDs by expanding clusters
            predicted_semantic_ids = self._get_semantic_tuples(str_predictions[i])
            predicted_song_ids = set()
            for sem_id_tuple in predicted_semantic_ids:
                if sem_id_tuple in self.semantic_to_song_map:
                    # Add all songs from the cluster to the set
                    predicted_song_ids.update(self.semantic_to_song_map[sem_id_tuple])
            
            # 3. Calculate set-based metrics
            intersection = len(predicted_song_ids.intersection(reference_song_ids))
            precision = intersection / len(predicted_song_ids) if predicted_song_ids else 0
            recall = intersection / len(reference_song_ids) if reference_song_ids else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            total_precision.append(precision)
            total_recall.append(recall)
            total_f1.append(f1)

        # --- Print Report ---
        print("\n" + "="*50)
        print("  TIGER Model Cluster-Aware Evaluation Summary")
        print("="*50)
        print(f"  Avg. Precision:     {np.mean(total_precision):.4f}")
        print(f"  Avg. Recall:        {np.mean(total_recall):.4f}")
        print(f"  Avg. F1-Score:      {np.mean(total_f1):.4f}")
        print("="*50)

    def _get_semantic_tuples(self, semantic_str: str) -> List[Tuple[int, ...]]:
        numerical_ids = [int(token[4:-1]) for token in semantic_str.split() if token.startswith("<id_") and token.endswith(">")]
        chunk_size = self.config.rqvae.levels
        semantic_tuples = []
        for i in range(0, len(numerical_ids) - chunk_size + 1, chunk_size):
            semantic_tuples.append(tuple(numerical_ids[i : i + chunk_size]))
        return list(dict.fromkeys(semantic_tuples)) # Return unique semantic ID tuples

    def _decode_semantic_string(self, semantic_str: str) -> List[str]:
        # This is now only used for decoding the ground truth
        tuples = self._get_semantic_tuples(semantic_str)
        song_ids = [self.semantic_to_song_map.get(t, [None])[0] for t in tuples]
        return [sid for sid in song_ids if sid is not None]

    def _load_model(self): return TIGERModel.from_pretrained(os.path.join(self.config.model_dir, "tiger_final"))
    def _create_reverse_map(self) -> Dict[Tuple[int, ...], List[str]]:
        # Maps a semantic ID tuple to a LIST of song IDs
        mapping = defaultdict(list)
        try:
            with open(os.path.join(self.config.output_dir, "song_semantic_ids.jsonl"), 'r') as f:
                for line in f: 
                    item = json.loads(line)
                    mapping[tuple(item['semantic_ids'])].append(item['song_id'])
        except FileNotFoundError: 
            logger.error("FATAL: song_semantic_ids.jsonl not found.")
            sys.exit(1)
        return mapping

if __name__ == "__main__":
    from collections import defaultdict
    config = Config()
    logger = setup_logging(os.path.join(config.log_dir, "phase4_evaluate_tiger.log"))
    evaluator = ClusterEvaluator(config)
    evaluator.run_evaluation()