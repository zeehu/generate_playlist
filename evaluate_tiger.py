"""
Phase 4 (Advanced): Evaluate the TIGER model with Multi-GPU support and detailed file output.
"""
import os
import sys
import torch
import evaluate
from tqdm import tqdm
import logging
import numpy as np
import json
from typing import Dict, List, Tuple

from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq

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
    def __init__(self, data_path: str, tokenizer: TIGERTokenizer, max_input_len: int, max_target_len: int):
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len
        self.data = []

        logger.info(f"Loading test data from {data_path}...")
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 3:
                    glid, input_text, target_sequence = parts
                    self.data.append({"glid": glid, "input_text": input_text, "target": target_sequence})
        logger.info(f"Loaded {len(self.data)} test samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_encoding = self.tokenizer.base_tokenizer(
            item["input_text"],
            max_length=self.max_input_len,
            truncation=True,
        )
        target_encoding = self.tokenizer.base_tokenizer(
            item["target"],
            max_length=self.max_target_len,
            truncation=True,
        )
        
        input_encoding["labels"] = target_encoding.input_ids
        return input_encoding

class ModelEvaluator:
    """Orchestrates multi-GPU evaluation and detailed result logging."""

    def __init__(self, config: Config):
        self.config = config
        self.model = self._load_model()
        self.song_info_map = self._load_song_info()
        self.semantic_to_song_map = self._create_reverse_map()

    def run_evaluation(self):
        logger.info("--- Starting Phase 4: Final Model Evaluation (Multi-GPU) ---")

        # 1. Prepare test dataset
        test_dataset = TestDataset(
            data_path=os.path.join(self.config.output_dir, "test.tsv"),
            tokenizer=self.model.tokenizer,
            max_input_len=self.config.tiger.max_input_length,
            max_target_len=self.config.tiger.max_target_length
        )

        # 2. Use Trainer for multi-GPU prediction
        training_args = TrainingArguments(
            output_dir=os.path.join(self.config.model_dir, "eval_temp"),
            per_device_eval_batch_size=self.config.tiger.per_device_eval_batch_size,
            dataloader_num_workers=self.config.num_workers,
            fp16=self.config.tiger.fp16,
            report_to="none",
        )

        trainer = Trainer(
            model=self.model.model, 
            args=training_args,
            data_collator=DataCollatorForSeq2Seq(tokenizer=self.model.tokenizer.base_tokenizer, model=self.model.model)
        )
        logger.info("Starting prediction on test set (will use all available GPUs)...")
        predictions = trainer.predict(test_dataset)
        
        # predictions.predictions is a numpy array of token IDs
        predicted_token_ids = predictions.predictions

        # 3. Decode, format, and save results
        self._process_and_save_results(test_dataset.data, predicted_token_ids)

        logger.info("--- Phase 4 Completed Successfully ---")

    def _process_and_save_results(self, test_data: List[Dict], predicted_token_ids: np.ndarray):
        logger.info("Decoding predictions and saving detailed results to file...")
        
        # For metric calculation
        str_predictions = []
        str_references = []

        output_file_path = os.path.join(self.config.output_dir, "evaluation_results.txt")
        with open(output_file_path, 'w', encoding='utf-8') as f:
            for i, item in enumerate(tqdm(test_data, desc="Processing and Saving Results")):
                # Decode Prediction
                pred_ids = [token_id for token_id in predicted_token_ids[i] if token_id != -100]
                pred_semantic_str = self.model.tokenizer.base_tokenizer.decode(pred_ids, skip_special_tokens=True)
                predicted_song_ids = self._decode_semantic_string(pred_semantic_str)
                str_predictions.append(pred_semantic_str)

                # Decode Reference
                ref_semantic_str = item["target"].replace(" <eos>", "")
                reference_song_ids = self._decode_semantic_string(ref_semantic_str)
                str_references.append(ref_semantic_str)

                # Write detailed comparison to file
                f.write("="*80 + "\n")
                f.write(f"Playlist ID: {item['glid']}\n")
                f.write(f"Input Text:  {item['input_text']}\n")
                f.write("-"*80 + "\n")
                
                f.write("--- Ground Truth --- ({} songs)\n".format(len(reference_song_ids)))
                for j, song_id in enumerate(reference_song_ids, 1):
                    info = self.song_info_map.get(song_id, {"name": "N/A", "singer": "N/A"})
                    f.write(f"  {j}. {info['name']} - {info['singer']} (ID: {song_id})\n")
                
                f.write("\n--- Model Prediction --- ({} songs)\n".format(len(predicted_song_ids)))
                for j, song_id in enumerate(predicted_song_ids, 1):
                    info = self.song_info_map.get(song_id, {"name": "N/A", "singer": "N/A"})
                    f.write(f"  {j}. {info['name']} - {info['singer']} (ID: {song_id})\n")
                f.write("\n\n")

        logger.info(f"Detailed comparison saved to {output_file_path}")
        # 4. Compute and print summary metrics
        self._compute_summary_metrics(str_predictions, str_references)

    def _decode_semantic_string(self, semantic_str: str) -> List[str]:
        numerical_ids = []
        for token in semantic_str.split():
            if token.startswith("<id_") and token.endswith(">"):
                try:
                    numerical_ids.append(int(token[4:-1]))
                except ValueError:
                    continue
        
        chunk_size = self.config.rqvae.levels
        reconstructed_song_ids = []
        for i in range(0, len(numerical_ids) - chunk_size + 1, chunk_size):
            id_chunk = tuple(numerical_ids[i : i + chunk_size])
            if id_chunk in self.semantic_to_song_map:
                reconstructed_song_ids.append(self.semantic_to_song_map[id_chunk])
        return reconstructed_song_ids

    def _compute_summary_metrics(self, predictions: list, references: list):
        # (This is the same logic as the previous version of the script)
        rouge_metric = evaluate.load(os.path.join(self.config.eval.local_metrics_path, 'rouge'))
        bleu_metric = evaluate.load(os.path.join(self.config.eval.local_metrics_path, 'bleu'))
        rouge_results = rouge_metric.compute(predictions=predictions, references=references)
        bleu_results = bleu_metric.compute(predictions=predictions, references=[[r] for r in references])

        total_f1 = 0
        for pred, ref in zip(predictions, references):
            pred_songs, ref_songs = set(pred.split()), set(ref.split())
            if not ref_songs: continue
            intersection = len(pred_songs.intersection(ref_songs))
            precision = intersection / len(pred_songs) if pred_songs else 0
            recall = intersection / len(ref_songs) if ref_songs else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            total_f1 += f1
        avg_f1 = total_f1 / len(references)

        print("\n" + "="*50)
        print("  TIGER Model Evaluation Summary")
        print("="*50)
        print(f"  ROUGE-L (F1-Score): {rouge_results['rougeL']:.4f}")
        print(f"  BLEU-4 Score:       {bleu_results['bleu']:.4f}")
        print(f"  Avg. Song F1-Score: {avg_f1:.4f}")
        print("="*50)

    # Helper methods to load necessary data
    def _load_model(self): return TIGERModel.from_pretrained(os.path.join(self.config.model_dir, "tiger_final"))
    def _load_song_info(self): 
        # (This logic can be copied from generate_playlist.py)
        import csv
        mapping = {}
        try:
            with open(self.config.data.song_info_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader, None) # Skip header
                for row in reader:
                    if len(row) >= 3: mapping[row[0]] = {"name": row[1], "singer": row[2]}
        except FileNotFoundError:
            logger.warning("Song info file not found. Names will not be displayed.")
        return mapping
    def _create_reverse_map(self):
        mapping = {}
        try:
            with open(os.path.join(self.config.output_dir, "song_semantic_ids.jsonl"), 'r') as f:
                for line in f: 
                    item = json.loads(line)
                    mapping[tuple(item['semantic_ids'])] = item['song_id']
        except FileNotFoundError:
            logger.error("FATAL: song_semantic_ids.jsonl not found. Run Phase 1 first.")
            sys.exit(1)
        return mapping

if __name__ == "__main__":
    config = Config()
    logger = setup_logging(os.path.join(config.log_dir, "phase4_evaluate_tiger.log"))
    evaluator = ModelEvaluator(config)
    evaluator.run_evaluation()