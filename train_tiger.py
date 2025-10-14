
import os
import sys
from dataclasses import dataclass, field

import torch
from torch.utils.data import Dataset
from transformers (
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
import logging

# Adjust path to import from playlist_src
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from config import Config
from tiger_model import TIGERModel, TIGERTokenizer
from utils import set_seed, setup_logging

logger = logging.getLogger(__name__)

class PlaylistCorpusDataset(Dataset):
    """Dataset for loading the playlist corpus (input_text, output_sequence)."""

    def __init__(self, data_path: str, tokenizer: TIGERTokenizer, max_input_len: int, max_target_len: int):
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len
        self.data = []

        logger.info(f"Loading data from {data_path}...")
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if '\t' in line:
                    input_text, target_text = line.strip().split('\t', 1)
                    self.data.append((input_text, target_text))
        logger.info(f"Loaded {len(self.data)} samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_text, target_text = self.data[idx]

        input_encoding = self.tokenizer.base_tokenizer(
            input_text,
            max_length=self.max_input_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        target_encoding = self.tokenizer.base_tokenizer(
            target_text,
            max_length=self.max_target_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        labels = target_encoding.input_ids
        # Replace padding token id with -100 so it's ignored in the loss calculation
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_encoding.input_ids.flatten(),
            "attention_mask": input_encoding.attention_mask.flatten(),
            "labels": labels.flatten()
        }

class TigerTrainer:
    """Orchestrates the training of the TIGER model."""

    def __init__(self, config: Config):
        self.config = config
        set_seed(config.seed)

    def run(self):
        logger.info("--- Starting Phase 3: TIGER Model Training ---")

        # 1. Initialize Model and Tokenizer
        # The TIGERModel will create its own tokenizer internally, which is crucial
        # because it adds all the special <id_xxx> tokens and resizes the model's embeddings.
        logger.info("Initializing TIGER model and tokenizer...")
        model = TIGERModel(
            base_model=self.config.tiger.model_name,
            vocab_size=self.config.rqvae.vocab_size
        )
        # We then get the tokenizer from the model instance to use for data processing.
        tokenizer = model.tokenizer

        # 2. Prepare Datasets
        train_dataset = self._prepare_dataset("train", tokenizer)
        val_dataset = self._prepare_dataset("val", tokenizer)

        # 3. Configure Training Arguments
        training_args = self._get_training_args()

        # 4. Initialize Trainer
        trainer = Trainer(
            model=model.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer.base_tokenizer, model=model.model),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )

        # 5. Start Training
        logger.info("Starting training... This may take a while.")
        trainer.train()

        # 6. Save the final model
        final_model_path = os.path.join(self.config.model_dir, "tiger_final")
        model.save_pretrained(final_model_path)
        logger.info(f"Training complete. Final model saved to {final_model_path}")
        logger.info("--- Phase 3 Completed Successfully ---")

    def _prepare_dataset(self, split: str, tokenizer: TIGERTokenizer) -> PlaylistCorpusDataset:
        data_path = os.path.join(self.config.output_dir, f"{split}.tsv")
        if not os.path.exists(data_path):
            logger.error(f"FATAL: {split}.tsv not found. Please run Phase 2 (prepare_corpus.py) first.")
            sys.exit(1)
        
        return PlaylistCorpusDataset(
            data_path=data_path,
            tokenizer=tokenizer,
            max_input_len=self.config.tiger.max_input_length,
            max_target_len=self.config.tiger.max_target_length
        )

    def _get_training_args(self) -> TrainingArguments:
        return TrainingArguments(
            output_dir=os.path.join(self.config.model_dir, "tiger_checkpoints"),
            num_train_epochs=self.config.tiger.num_train_epochs,
            per_device_train_batch_size=self.config.tiger.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.tiger.per_device_eval_batch_size,
            learning_rate=self.config.tiger.learning_rate,
            warmup_steps=self.config.tiger.warmup_steps,
            weight_decay=self.config.tiger.weight_decay,
            
            # Performance and multi-GPU optimizations
            fp16=self.config.tiger.fp16,
            gradient_checkpointing=True, # Saves memory
            dataloader_num_workers=self.config.num_workers,

            # Evaluation and saving strategy
            evaluation_strategy="steps",
            eval_steps=1000,
            save_strategy="steps",
            save_steps=1000,
            save_total_limit=2, # Only keep the last 2 checkpoints
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,

            # Logging
            logging_dir=os.path.join(self.config.log_dir, "tiger_logs"),
            logging_steps=100,
            report_to="none",  # Disable all online reporting for offline environment
            
            # Required for multi-GPU
            remove_unused_columns=False,
        )

if __name__ == "__main__":
    config = Config()
    log_file_path = os.path.join(config.log_dir, "phase3_train_tiger.log")
    logger = setup_logging(log_file_path)

    trainer = TigerTrainer(config)
    trainer.run()
