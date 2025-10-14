"""
TIGER: T5-based Generative Recommendation Model
"""
import torch
import torch.nn as nn
from transformers import (
    T5ForConditionalGeneration, 
    T5Tokenizer, 
    T5Config,
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)
from typing import List, Dict, Optional, Tuple
import json
import os

class TIGERTokenizer:
    """Custom tokenizer for TIGER model with semantic IDs"""
    
    def __init__(self, base_model: str = "t5-small", vocab_size: int = 16384):
        self.base_tokenizer = T5Tokenizer.from_pretrained(base_model)
        self.vocab_size = vocab_size
        
        # Add semantic ID tokens
        semantic_tokens = [f"<id_{i}>" for i in range(vocab_size)]
        special_tokens = ["<user>", "<eos>", "<unk>", "<pad>", "<mask>"]
        
        new_tokens = semantic_tokens + special_tokens
        self.base_tokenizer.add_tokens(new_tokens)
        
        # Create token mappings
        self.semantic_id_to_token = {i: f"<id_{i}>" for i in range(vocab_size)}
        self.token_to_semantic_id = {f"<id_{i}>": i for i in range(vocab_size)}
        
    def encode(self, text: str, **kwargs) -> List[int]:
        """Encode text to token IDs"""
        return self.base_tokenizer.encode(text, **kwargs)
    
    def decode(self, token_ids: List[int], **kwargs) -> str:
        """Decode token IDs to text"""
        return self.base_tokenizer.decode(token_ids, **kwargs)
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text"""
        return self.base_tokenizer.tokenize(text)
    
    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """Convert tokens to IDs"""
        return self.base_tokenizer.convert_tokens_to_ids(tokens)
    
    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        """Convert IDs to tokens"""
        return self.base_tokenizer.convert_ids_to_tokens(ids)
    
    def __len__(self):
        return len(self.base_tokenizer)
    
    @property
    def pad_token_id(self):
        return self.base_tokenizer.pad_token_id
    
    @property
    def eos_token_id(self):
        return self.base_tokenizer.eos_token_id
    
    def save_pretrained(self, save_directory: str):
        """Save tokenizer"""
        self.base_tokenizer.save_pretrained(save_directory)
        
        # Save custom mappings
        mappings = {
            'semantic_id_to_token': self.semantic_id_to_token,
            'token_to_semantic_id': self.token_to_semantic_id,
            'vocab_size': self.vocab_size
        }
        
        with open(os.path.join(save_directory, 'custom_mappings.json'), 'w') as f:
            json.dump(mappings, f)
    
    @classmethod
    def from_pretrained(cls, load_directory: str):
        """Load tokenizer"""
        # Load base tokenizer
        base_tokenizer = T5Tokenizer.from_pretrained(load_directory)
        
        # Load custom mappings
        mappings_path = os.path.join(load_directory, 'custom_mappings.json')
        with open(mappings_path, 'r') as f:
            mappings = json.load(f)
        
        # Create instance
        tokenizer = cls.__new__(cls)
        tokenizer.base_tokenizer = base_tokenizer
        tokenizer.vocab_size = mappings['vocab_size']
        tokenizer.semantic_id_to_token = mappings['semantic_id_to_token']
        tokenizer.token_to_semantic_id = mappings['token_to_semantic_id']
        
        return tokenizer

class TIGERModel(nn.Module):
    """TIGER: T5-based Generative Recommendation Model"""
    
    def __init__(self, base_model: str = "t5-small", vocab_size: int = 16384):
        super().__init__()
        
        # Load base T5 model
        self.config = T5Config.from_pretrained(base_model)
        self.model = T5ForConditionalGeneration.from_pretrained(base_model)
        
        # Resize embeddings for new tokens
        self.tokenizer = TIGERTokenizer(base_model, vocab_size)
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Store vocab size
        self.vocab_size = vocab_size
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None,
                labels: torch.Tensor = None, **kwargs):
        """Forward pass"""
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )
    
    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None,
                max_new_tokens: int = 10, num_beams: int = 5, 
                num_return_sequences: int = 1, **kwargs):
        """Generate recommendations"""
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            **kwargs
        )
    
    def recommend(self, user_sequence: List[str], num_recommendations: int = 10,
                 num_beams: int = 10) -> List[List[int]]:
        """Generate recommendations for a user sequence"""
        self.eval()
        
        # Prepare input
        input_text = " ".join(user_sequence)
        input_ids = torch.tensor([self.tokenizer.encode(input_text)]).to(next(self.parameters()).device)
        
        # Generate
        with torch.no_grad():
            outputs = self.generate(
                input_ids=input_ids,
                max_new_tokens=2,  # Generate 2 tokens (one semantic ID)
                num_beams=num_beams,
                num_return_sequences=num_recommendations
            )
        
        # Decode recommendations
        recommendations = []
        for output in outputs:
            # Get generated tokens (skip input)
            generated_tokens = output[input_ids.shape[1]:]
            decoded = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # Extract semantic IDs
            tokens = decoded.split()
            semantic_ids = []
            for token in tokens:
                if token in self.tokenizer.token_to_semantic_id:
                    semantic_ids.append(self.tokenizer.token_to_semantic_id[token])
            
            if semantic_ids:
                recommendations.append(semantic_ids)
        
        return recommendations
    
    def save_pretrained(self, save_directory: str):
        """Save model and tokenizer"""
        os.makedirs(save_directory, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(save_directory)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(save_directory)
        
        # Save config
        config_dict = {
            'vocab_size': self.vocab_size,
            'base_model': self.config.name_or_path if hasattr(self.config, 'name_or_path') else 't5-small'
        }
        
        with open(os.path.join(save_directory, 'tiger_config.json'), 'w') as f:
            json.dump(config_dict, f)
    
    @classmethod
    def from_pretrained(cls, load_directory: str):
        """Load model and tokenizer"""
        # Load config
        config_path = os.path.join(load_directory, 'tiger_config.json')
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        # Create model
        model = cls(
            base_model=config_dict.get('base_model', 't5-small'),
            vocab_size=config_dict['vocab_size']
        )
        
        # Load model weights
        model.model = T5ForConditionalGeneration.from_pretrained(load_directory)
        
        # Load tokenizer
        model.tokenizer = TIGERTokenizer.from_pretrained(load_directory)
        
        return model
