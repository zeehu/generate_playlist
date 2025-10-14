
"""
import os
import sys
import torch
import evaluate # Hugging Face's evaluation library
from tqdm import tqdm
import logging
import numpy as np

# Adjust path to import from playlist_src
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from config import Config
from tiger_model import TIGERModel
from utils import setup_logging

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Orchestrates the evaluation of the final TIGER model."""

    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model()

    def _load_model(self) -> TIGERModel:
        model_path = os.path.join(self.config.model_dir, "tiger_final")
        if not os.path.exists(model_path):
            logger.error(f"FATAL: Final model not found at {model_path}")
            logger.error("Please run Phase 3 (train_tiger.py) first.")
            sys.exit(1)
        
        logger.info(f"Loading model from {model_path}...")
        model = TIGERModel.from_pretrained(model_path)
        model.model.to(self.device)
        model.eval()
        return model

    def run_evaluation(self):
        logger.info("--- Starting Phase 4: Final Model Evaluation ---")

        # 1. Load test data
        test_data_path = os.path.join(self.config.output_dir, "test.tsv")
        if not os.path.exists(test_data_path):
            logger.error(f"FATAL: test.tsv not found. Please run Phase 2 first.")
            sys.exit(1)
        
        with open(test_data_path, 'r', encoding='utf-8') as f:
            test_data = [line.strip().split('\t', 1) for line in f if '\t' in line]

        # 2. Generate predictions for the test set
        predictions, references = self._generate_predictions(test_data)

        # 3. Compute and print metrics
        self._compute_metrics(predictions, references)

        logger.info("--- Phase 4 Completed Successfully ---")

    def _generate_predictions(self, test_data: list) -> tuple[list, list]:
        predictions, references = [], []
        
        logger.info(f"Generating predictions for {len(test_data)} test samples...")
        for input_text, reference_sequence in tqdm(test_data, desc="Generating Predictions"):
            input_ids = self.model.tokenizer.base_tokenizer(
                input_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.tiger.max_input_length
            ).input_ids.to(self.device)

            with torch.no_grad():
                generated_ids = self.model.model.generate(
                    input_ids,
                    max_new_tokens=self.config.tiger.max_target_length,
                    num_beams=3, # Use beam search for higher quality evaluation
                    early_stopping=True
                )
            
            # Decode the generated IDs to a string
            prediction = self.model.tokenizer.base_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            
            predictions.append(prediction)
            # The reference already has <eos> removed by this point if we use skip_special_tokens
            references.append(reference_sequence.replace(" <eos>", ""))

        return predictions, references

    def _compute_metrics(self, predictions: list, references: list):
        logger.info("Computing evaluation metrics...")

        # --- Text-based Metrics (ROUGE, BLEU) ---
        rouge_metric = evaluate.load('rouge')
        bleu_metric = evaluate.load('bleu')

        rouge_results = rouge_metric.compute(predictions=predictions, references=references)
        # For BLEU, references must be a list of lists
        bleu_results = bleu_metric.compute(predictions=predictions, references=[[r] for r in references])

        # --- Set-based Metrics (Song F1-Score) ---
        total_precision, total_recall, total_f1 = 0, 0, 0
        for pred, ref in zip(predictions, references):
            pred_songs = set(pred.split())
            ref_songs = set(ref.split())
            
            if not ref_songs: continue

            intersection = len(pred_songs.intersection(ref_songs))
            precision = intersection / len(pred_songs) if pred_songs else 0
            recall = intersection / len(ref_songs) if ref_songs else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            total_precision += precision
            total_recall += recall
            total_f1 += f1

        avg_precision = total_precision / len(references)
        avg_recall = total_recall / len(references)
        avg_f1 = total_f1 / len(references)

        # --- Print Report ---
        print("\n" + "="*50)
        print("  TIGER Model Evaluation Report")
        print("="*50)
        print("\n[Text Generation Metrics]")
        print(f"  ROUGE-L (F1-Score): {rouge_results['rougeL']:.4f}")
        print(f"  BLEU-4 Score:       {bleu_results['bleu']:.4f}")
        print("\n[Song Set Metrics]")
        print(f"  Avg. Precision:     {avg_precision:.4f}")
        print(f"  Avg. Recall:        {avg_recall:.4f}")
        print(f"  Avg. F1-Score:      {avg_f1:.4f}")
        print("="*50)

if __name__ == "__main__":
    config = Config()
    log_file_path = os.path.join(config.log_dir, "phase4_evaluate_tiger.log")
    logger = setup_logging(log_file_path)

    evaluator = ModelEvaluator(config)
    evaluator.run_evaluation()
