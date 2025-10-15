
"""
Phase 4 (Advanced): Evaluate the TIGER model with Multi-GPU support and detailed file output.
This version uses Accelerate for a memory-efficient, manual inference loop.
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
from accelerate import Accelerator, DistributedType

# Dependencies for offline metrics
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

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
            padding=False
        )
        # Return only model inputs
        return {"input_ids": input_encoding.input_ids, "attention_mask": input_encoding.attention_mask}

class ModelEvaluator:
    """Orchestrates multi-GPU evaluation using Accelerate."""

    def __init__(self, config: Config):
        self.config = config
        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        self.model = self._load_model()
        self.song_info_map = self._load_song_info()
        self.semantic_to_song_map = self._create_reverse_map()

    def run_evaluation(self):
        logger.info("--- Starting Phase 4: Final Model Evaluation (Accelerate) ---")

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

        all_predictions = []

        logger.info("Starting generation on test set (will use all available GPUs)...")
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
            all_predictions.extend(gathered_predictions.cpu().numpy())

        if self.accelerator.is_main_process:
            all_predictions = all_predictions[:len(test_dataset)]
            self._process_and_save_results(test_dataset.data, all_predictions)

        logger.info("--- Phase 4 Completed Successfully ---")

    def _process_and_save_results(self, test_data: List[Dict], predicted_token_ids: np.ndarray):
        logger.info("Decoding predictions and saving detailed results to file...")
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        str_predictions = unwrapped_model.tokenizer.base_tokenizer.batch_decode(predicted_token_ids, skip_special_tokens=True)
        
        reference_texts = [item["target"] for item in test_data]
        str_references = [ref.replace(" <eos>", "") for ref in reference_texts]

        output_file_path = os.path.join(self.config.output_dir, "evaluation_results.txt")
        with open(output_file_path, 'w', encoding='utf-8') as f:
            for i, item in enumerate(tqdm(test_data, desc="Processing and Saving Results")):
                predicted_song_ids = self._decode_semantic_string(str_predictions[i])
                reference_song_ids = self._decode_semantic_string(str_references[i])

                f.write("="*80 + "\n")
                f.write(f"Playlist ID: {item['glid']}\n")
                f.write(f"Input Text:  {item['input_text']}\n")
                f.write("-"*80 + "\n")
                f.write(f"--- Ground Truth --- ({len(reference_song_ids)} songs)\n")
                for j, song_id in enumerate(reference_song_ids, 1): f.write(f"  {j}. {self.song_info_map.get(song_id, {}).get('name', 'N/A')} - {self.song_info_map.get(song_id, {}).get('singer', 'N/A')} (ID: {song_id})\n")
                f.write(f"\n--- Model Prediction --- ({len(predicted_song_ids)} songs)\n")
                for j, song_id in enumerate(predicted_song_ids, 1): f.write(f"  {j}. {self.song_info_map.get(song_id, {}).get('name', 'N/A')} - {self.song_info_map.get(song_id, {}).get('singer', 'N/A')} (ID: {song_id})\n")
                f.write("\n\n")

        logger.info(f"Detailed comparison saved to {output_file_path}")
        self._compute_summary_metrics(str_predictions, str_references)

    def _decode_semantic_string(self, semantic_str: str) -> List[str]:
        numerical_ids = [int(token[4:-1]) for token in semantic_str.split() if token.startswith("<id_") and token.endswith(">")]
        chunk_size = self.config.rqvae.levels
        reconstructed_song_ids = []
        for i in range(0, len(numerical_ids) - chunk_size + 1, chunk_size):
            id_chunk = tuple(numerical_ids[i : i + chunk_size])
            if id_chunk in self.semantic_to_song_map: reconstructed_song_ids.append(self.semantic_to_song_map[id_chunk])
        return reconstructed_song_ids

    def _compute_summary_metrics(self, predictions: list, references: list):
        logger.info("Computing evaluation metrics using local implementations...")

        # 1. ROUGE Score
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        rouge_l_f1 = np.mean([scorer.score(ref, pred)['rougeL'].fmeasure for ref, pred in zip(references, predictions)])

        # 2. BLEU Score
        chencherry = SmoothingFunction()
        bleu_scores = []
        for ref, pred in zip(references, predictions):
            ref_tokens = [ref.split()]
            pred_tokens = pred.split()
            bleu_scores.append(sentence_bleu(ref_tokens, pred_tokens, smoothing_function=chencherry.method1))
        avg_bleu = np.mean(bleu_scores)

        # 3. Set-based F1 Score
        avg_f1 = np.mean([self._calculate_f1(p, r) for p, r in zip(predictions, references)])

        # --- Print Report ---
        print("\n" + "="*50)
        print("  TIGER Model Evaluation Summary")
        print("="*50)
        print(f"  ROUGE-L (F1-Score): {rouge_l_f1:.4f}")
        print(f"  BLEU-4 Score:       {avg_bleu:.4f}")
        print(f"  Avg. Song F1-Score: {avg_f1:.4f}")
        print("="*50)

    def _calculate_f1(self, pred, ref):
        pred_songs, ref_songs = set(pred.split()), set(ref.split())
        if not ref_songs: return 0
        intersection = len(pred_songs.intersection(ref_songs))
        precision = intersection / len(pred_songs) if pred_songs else 0
        recall = intersection / len(ref_songs) if ref_songs else 0
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    def _load_model(self): return TIGERModel.from_pretrained(os.path.join(self.config.model_dir, "tiger_final"))
    def _load_song_info(self): 
        import csv
        mapping = {}
        try:
            with open(self.config.data.song_info_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader, None)
                for row in reader:
                    if len(row) >= 3: mapping[row[0]] = {"name": row[1], "singer": row[2]}
        except FileNotFoundError: logger.warning("Song info file not found.")
        return mapping
    def _create_reverse_map(self):
        mapping = {}
        try:
            with open(os.path.join(self.config.output_dir, "song_semantic_ids.jsonl"), 'r') as f:
                for line in f: 
                    item = json.loads(line)
                    mapping[tuple(item['semantic_ids'])] = item['song_id']
        except FileNotFoundError: 
            logger.error("FATAL: song_semantic_ids.jsonl not found.")
            sys.exit(1)
        return mapping

if __name__ == "__main__":
    config = Config()
    logger = setup_logging(os.path.join(config.log_dir, "phase4_evaluate_tiger.log"))
    evaluator = ModelEvaluator(config)
    evaluator.run_evaluation()
