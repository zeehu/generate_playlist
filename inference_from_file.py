"""
Phase 5 (Advanced): Inference from a file.

This script loads the final model, reads queries from a file,
and outputs all songs corresponding to the generated semantic IDs.
"""
import os
import sys
import torch
import json
import logging
from typing import Dict, Tuple, List
from collections import defaultdict

# Adjust path to import from playlist_src
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from config import Config
from tiger_model import TIGERModel
from utils import setup_logging

logger = logging.getLogger(__name__)

class PlaylistGeneratorFromFile:
    """Handles loading the model and generating playlists from text prompts in a file."""

    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model()
        self.semantic_to_song_cluster = self._create_reverse_map()
        self.song_info_map = self._load_song_info()

    def _load_model(self) -> TIGERModel:
        model_path = os.path.join(self.config.model_dir, "tiger_final")
        if not os.path.exists(model_path):
            logger.error(f"FATAL: Final model not found at {model_path}")
            sys.exit(1)
        
        logger.info(f"Loading model from {model_path}...")
        model = TIGERModel.from_pretrained(model_path)
        model.model.to(self.device)
        model.eval()
        return model

    def _create_reverse_map(self) -> Dict[Tuple[int, ...], List[str]]:
        # Maps a semantic ID tuple to a LIST of song IDs
        mapping = defaultdict(list)
        semantic_ids_file = os.path.join(self.config.output_dir, "song_semantic_ids.jsonl")
        if not os.path.exists(semantic_ids_file):
            logger.error(f"FATAL: song_semantic_ids.jsonl not found.")
            sys.exit(1)

        logger.info("Creating reverse mapping from semantic IDs to song clusters...")
        with open(semantic_ids_file, 'r') as f:
            for line in f: 
                item = json.loads(line)
                mapping[tuple(item['semantic_ids'])].append(item['song_id'])
        return mapping

    def _load_song_info(self) -> Dict[str, Dict[str, str]]:
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

    def inference_for_query(self, title: str) -> List[Dict[str, str]]:
        """
        Generates a playlist for a given title and returns all matching songs.
        """
        prompt = title
        logger.info(f"Generating with prompt: {prompt}")

        input_ids = self.model.tokenizer.base_tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)

        with torch.no_grad():
            generated_ids = self.model.model.generate(
                input_ids,
                max_new_tokens=self.config.tiger.max_target_length,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.8,
                num_return_sequences=1
            )
        
        decoded_tokens = self.model.tokenizer.base_tokenizer.convert_ids_to_tokens(generated_ids[0], skip_special_tokens=True)

        numerical_ids = [int(token[4:-1]) for token in decoded_tokens if token.startswith("<id_")]
        chunk_size = self.config.rqvae.levels
        semantic_id_tuples = []
        for i in range(0, len(numerical_ids) - chunk_size + 1, chunk_size):
            semantic_id_tuples.append(tuple(numerical_ids[i : i + chunk_size]))
        unique_semantic_ids = list(dict.fromkeys(semantic_id_tuples))

        results = []
        for id_tuple in unique_semantic_ids:
            if id_tuple in self.semantic_to_song_cluster:
                song_cluster = self.semantic_to_song_cluster[id_tuple]
                for song_id in song_cluster:
                    info = self.song_info_map.get(song_id, {"name": "N/A", "singer": "N/A"})
                    results.append({
                        "song_id": song_id,
                        "semantic_id": str(id_tuple),
                        "song_name": info["name"],
                        "singer": info["singer"]
                    })
        return results

if __name__ == "__main__":
    config = Config()
    logger = setup_logging(level=logging.INFO)
    
    generator = PlaylistGeneratorFromFile(config)
    
    query_file = 'semantic_query.txt'
    
    if not os.path.exists(query_file):
        logger.error(f"Query file not found: {query_file}")
        sys.exit(1)
        
    cnt = 0
    queries = []
    with open(query_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip() or line.strip() == "query":
                continue
            cnt += 1
            queries.append(line.strip())
            if cnt >= 1000:
                break

    for query in queries:
        print(f"query:{query}")
        results = generator.inference_for_query(query)
        
        if not results:
            print("No songs found for this query.")
            continue
            
        for result in results:
            print(f"{result['song_id']}\t{result['semantic_id']}\t{result['song_name']}\t{result['singer']}")
        print("\n")
