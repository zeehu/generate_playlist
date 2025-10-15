
"""
Phase 4 (Final): Evaluate the TIGER model with comprehensive, user-defined metrics.

This script:
1. Calculates metrics (ROUGE, BLEU) on the raw semantic ID sequences.
2. Calculates a cluster-aware F1-score by expanding predicted clusters.
3. Produces a detailed side-by-side comparison file showing songs and their semantic IDs.
"""
import os
import sys
import torch
import random
from tqdm import tqdm
import logging
import numpy as np
import json
from typing import Dict, List, Tuple
from collections import defaultdict

from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator

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
                    self.data.append({"glid": glid, "input_text": input_text, "target_sequence": target_sequence})
        logger.info(f"Loaded {len(self.data)} test samples.")

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_encoding = self.tokenizer.base_tokenizer(item["input_text"], max_length=self.max_input_len, truncation=True)
        return input_encoding

class ModelEvaluator:
    def __init__(self, config: Config):
        self.config = config
        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        self.model = self._load_model()
        self.song_to_sem_id_map, self.sem_id_to_songs_map = self._create_mappings()
        self.song_info_map = self._load_song_info()

    def run_evaluation(self):
        logger.info("--- Starting Phase 4: Final Model Evaluation ---")
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
                    num_beams=3,
                    repetition_penalty=1.2
                )
            padded_predictions = self.accelerator.pad_across_processes(generated_tokens, dim=1, pad_index=tokenizer.pad_token_id)
            gathered_predictions = self.accelerator.gather(padded_predictions)
            all_predicted_token_ids.extend(gathered_predictions.cpu().numpy())

        if self.accelerator.is_main_process:
            all_predicted_token_ids = all_predicted_token_ids[:len(test_dataset)]
            self._process_and_save_results(test_dataset.data, all_predicted_token_ids)

        logger.info("--- Evaluation Completed ---")

    def _process_and_save_results(self, test_data: List[Dict], predicted_token_ids: np.ndarray):
        logger.info("Decoding predictions and saving detailed results to file...")
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        str_predictions = unwrapped_model.tokenizer.base_tokenizer.batch_decode(predicted_token_ids, skip_special_tokens=True)
        str_references = [item["target_sequence"].replace(" <eos>", "") for item in test_data]

        output_file_path = os.path.join(self.config.output_dir, "evaluation_results.txt")
        with open(output_file_path, 'w', encoding='utf-8') as f:
            for i, item in enumerate(tqdm(test_data, desc="Processing and Saving Results")):
                gt_sem_ids = self._get_semantic_tuples(str_references[i])
                gt_song_ids = [self.sem_id_to_songs_map.get(t, [None])[0] for t in gt_sem_ids]
                gt_song_ids = [sid for sid in gt_song_ids if sid is not None]

                pred_sem_ids = self._get_semantic_tuples(str_predictions[i])
                pred_song_ids = [random.choice(self.sem_id_to_songs_map.get(sem_id, ["N/A"])) for sem_id in pred_sem_ids]

                f.write("="*90 + "\n")
                f.write(f"Playlist ID: {item['glid']}\nInput Text:  {item['input_text']}\n")
                f.write("-"*90 + "\n")
                f.write("{:<45} | {:<45}\n".format("--- Ground Truth ---", "--- Model Prediction ---"))
                f.write("-"*90 + "\n")

                max_len = max(len(gt_song_ids), len(pred_song_ids))
                for row_idx in range(max_len):
                    gt_part, pred_part = "", ""
                    if row_idx < len(gt_song_ids):
                        song_id = gt_song_ids[row_idx]
                        info = self.song_info_map.get(song_id, {"name": "N/A", "singer": "N/A"})
                        sem_id = self.song_to_sem_id_map.get(song_id, "N/A")
                        gt_part = f"{info['name']} - {info['singer']} {sem_id}"

                    if row_idx < len(pred_song_ids):
                        song_id = pred_song_ids[row_idx]
                        info = self.song_info_map.get(song_id, {"name": "N/A", "singer": "N/A"})
                        sem_id = self.song_to_sem_id_map.get(song_id, "N/A")
                        pred_part = f"{info['name']} - {info['singer']} {sem_id}"

                    f.write("{:<45} | {:<45}\n".format(gt_part[:44], pred_part[:44]))
                f.write("\n\n")

        logger.info(f"Detailed comparison saved to {output_file_path}")
        self._compute_summary_metrics(str_predictions, str_references)

    def _get_semantic_tuples(self, semantic_str: str) -> List[Tuple[int, ...]]:
        numerical_ids = [int(token[4:-1]) for token in semantic_str.split() if token.startswith("<id_")]
        chunk_size = self.config.rqvae.levels
        return list(dict.fromkeys([tuple(numerical_ids[i:i+chunk_size]) for i in range(0, len(numerical_ids) - chunk_size + 1, chunk_size)]))

    def _compute_summary_metrics(self, predictions: list, references: list):
        # Metric 1 & 2: ROUGE and BLEU on semantic ID strings
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        rouge_l_f1 = np.mean([scorer.score(ref, pred)['rougeL'].fmeasure for ref, pred in zip(references, predictions)])
        chencherry = SmoothingFunction()
        bleu_scores = [sentence_bleu([ref.split()], pred.split(), smoothing_function=chencherry.method1) for ref, pred in zip(references, predictions)]
        avg_bleu = np.mean(bleu_scores)

        # Metric 3: Cluster-Aware Song F1-Score
        f1_scores = []
        for pred_str, ref_str in zip(predictions, references):
            ref_sem_ids = self._get_semantic_tuples(ref_str)
            ref_song_ids = set([self.sem_id_to_songs_map.get(t, [None])[0] for t in ref_sem_ids if t in self.sem_id_to_songs_map])

            pred_sem_ids = self._get_semantic_tuples(pred_str)
            predicted_song_ids = set()
            for sem_id_tuple in pred_sem_ids:
                predicted_song_ids.update(self.sem_id_to_songs_map.get(sem_id_tuple, []))
            
            f1 = self._calculate_f1_from_sets(predicted_song_ids, ref_song_ids)
            f1_scores.append(f1)
        avg_f1 = np.mean(f1_scores)

        print("\n" + "="*60)
        print("  TIGER Model Evaluation Summary")
        print("="*60)
        print("  [Semantic ID Sequence Metrics]")
        print(f"  ROUGE-L (F1-Score): {rouge_l_f1:.4f}")
        print(f"  BLEU-4 Score:       {avg_bleu:.4f}")
        print("\n  [Expanded Song Set Metrics]")
        print(f"  Avg. Song F1-Score: {avg_f1:.4f}")
        print("="*60)

    def _calculate_f1_from_sets(self, pred_set, ref_set):
        if not ref_set: return 0
        intersection = len(pred_set.intersection(ref_set))
        precision = intersection / len(pred_set) if pred_set else 0
        recall = intersection / len(ref_set) if ref_set else 0
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
    def _create_mappings(self) -> Tuple[Dict[str, Tuple], Dict[Tuple, List[str]]]:
        song_to_sem_id = {}
        sem_id_to_songs = defaultdict(list)
        try:
            with open(os.path.join(self.config.output_dir, "song_semantic_ids.jsonl"), 'r') as f:
                for line in f: 
                    item = json.loads(line)
                    song_id = item['song_id']
                    sem_id = tuple(item['semantic_ids'])
                    song_to_sem_id[song_id] = sem_id
                    sem_id_to_songs[sem_id].append(song_id)
        except FileNotFoundError: 
            logger.error("FATAL: song_semantic_ids.jsonl not found.")
            sys.exit(1)
        return dict(song_to_sem_id), dict(sem_id_to_songs)

if __name__ == "__main__":
    config = Config()
    logger = setup_logging(os.path.join(config.log_dir, "phase4_evaluate_tiger.log"))
    evaluator = ModelEvaluator(config)
    evaluator.run_evaluation()
