import torch
import torch.nn as nn
from transformers import (
    T5ForConditionalGeneration, 
    T5Tokenizer, 
    T5Config
)
from typing import List, Dict, Optional, Tuple
import json
import os

class TIGERTokenizer:
    """Custom tokenizer for TIGER model with semantic IDs"""
    
    def __init__(self, base_model: str = "t5-small", vocab_size: int = 16384):
        self.base_tokenizer = T5Tokenizer.from_pretrained(base_model)
        self.vocab_size = vocab_size
        
        semantic_tokens = [f"<id_{i}>" for i in range(vocab_size)]
        special_tokens = ["<user>", "<eos>", "<unk>", "<pad>", "<mask>"]
        
        new_tokens = semantic_tokens + special_tokens
        self.base_tokenizer.add_tokens(new_tokens)
        
        self.semantic_id_to_token = {i: f"<id_{i}>" for i in range(vocab_size)}
        self.token_to_semantic_id = {f"<id_{i}>": i for i in range(vocab_size)}
        
    def __len__(self):
        return len(self.base_tokenizer)
    
    @property
    def pad_token_id(self):
        return self.base_tokenizer.pad_token_id

    def save_pretrained(self, save_directory: str):
        self.base_tokenizer.save_pretrained(save_directory)
        mappings = {
            'vocab_size': self.vocab_size
        }
        with open(os.path.join(save_directory, 'custom_tokenizer_config.json'), 'w') as f:
            json.dump(mappings, f)
    
    @classmethod
    def from_pretrained(cls, load_directory: str):
        base_tokenizer = T5Tokenizer.from_pretrained(load_directory)
        mappings_path = os.path.join(load_directory, 'custom_tokenizer_config.json')
        with open(mappings_path, 'r') as f:
            mappings = json.load(f)
        
        tokenizer = cls.__new__(cls)
        tokenizer.base_tokenizer = base_tokenizer
        tokenizer.vocab_size = mappings['vocab_size']
        tokenizer.semantic_id_to_token = {i: f"<id_{i}>" for i in range(mappings['vocab_size'])}
        tokenizer.token_to_semantic_id = {f"<id_{i}>": i for i in range(mappings['vocab_size'])}
        return tokenizer

class TIGERModel(nn.Module):
    """TIGER: T5-based Generative Recommendation Model"""
    
    def __init__(self, base_model: str = "t5-small", vocab_size: int = 16384):
        super().__init__()
        # Store the base model path for robust saving
        self.base_model_path = base_model
        self.config = T5Config.from_pretrained(base_model)
        self.model = T5ForConditionalGeneration.from_pretrained(base_model)
        self.tokenizer = TIGERTokenizer(base_model, vocab_size)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.vocab_size = vocab_size
        
    def forward(self, **kwargs):
        return self.model(**kwargs)
    
    def save_pretrained(self, save_directory: str):
        os.makedirs(save_directory, exist_ok=True)
        self.model.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)
        config_dict = {
            'vocab_size': self.vocab_size,
            'base_model': self.base_model_path
        }
        with open(os.path.join(save_directory, 'tiger_config.json'), 'w') as f:
            json.dump(config_dict, f)
    
    @classmethod
    def from_pretrained(cls, load_directory: str):
        config_path = os.path.join(load_directory, 'tiger_config.json')
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        model = cls(
            base_model=config_dict['base_model'],
            vocab_size=config_dict['vocab_size']
        )
        
        model.model = T5ForConditionalGeneration.from_pretrained(load_directory)
        model.tokenizer = TIGERTokenizer.from_pretrained(load_directory)
        return model