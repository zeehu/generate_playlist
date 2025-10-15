"""
Phase 5 (Advanced): Interactive Inference using Cluster Expansion.

This script loads the final model and demonstrates the cluster-based recommendation
approach. For each generated semantic ID (cluster), it randomly samples one
song to create a diverse and surprising playlist.
"""
import os
import sys
import torch
import json
import logging
import random
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

class PlaylistGenerator:
    """Handles loading the model and generating playlists from text prompts."""

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

    def generate(self, title: str, tags: str = "") -> List[str]:
        """Generates a playlist for a given title and optional tags."""
        prompt = f"歌单标题：{title} | 歌单标签：{tags}"
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

        # Get unique semantic ID tuples from the generated sequence
        numerical_ids = [int(token[4:-1]) for token in decoded_tokens if token.startswith("<id_")]
        chunk_size = self.config.rqvae.levels
        semantic_id_tuples = []
        for i in range(0, len(numerical_ids) - chunk_size + 1, chunk_size):
            semantic_id_tuples.append(tuple(numerical_ids[i : i + chunk_size]))
        unique_semantic_ids = list(dict.fromkeys(semantic_id_tuples)) # De-duplicate while preserving order

        # For each unique semantic ID, sample one song from its cluster
        reconstructed_song_ids = []
        for id_tuple in unique_semantic_ids:
            if id_tuple in self.semantic_to_song_cluster:
                song_cluster = self.semantic_to_song_cluster[id_tuple]
                # Randomly sample one song from the cluster
                sampled_song = random.choice(song_cluster)
                reconstructed_song_ids.append(sampled_song)
        
        return reconstructed_song_ids

    def interactive_demo(self):
        """Starts an interactive command-line demo."""
        print("\n" + "="*50)
        print("  🎶 歌单生成模型已就绪 (簇扩展模式) 🎶")
        print("="*50)
        print("  输入标题，模型会推荐一个“概念”列表，并从每个概念中随机抽一首歌。")
        print("  每次推荐的歌单可能都不同，体现了多样性！")
        print("  输入 'exit' 或 'quit' 即可退出。")
        print("-"*50)

        while True:
            try:
                prompt = input("\n请输入歌单标题 > ")
                if prompt.lower() in ['exit', 'quit']:
                    print("感谢使用，再见！")
                    break
                
                if not prompt: continue

                print("\n生成中，请稍候...")
                song_ids = self.generate(prompt)

                if not song_ids:
                    print("模型未能生成有效的歌曲列表，请尝试更换标题。")
                    continue
                
                print("\n✨ 为您推荐的歌曲列表: ✨")
                for i, song_id in enumerate(song_ids, 1):
                    info = self.song_info_map.get(song_id, {"name": "N/A", "singer": "N/A"})
                    cluster_size = len(self.semantic_to_song_cluster.get(self._get_sem_id_for_song(song_id), []))
                    print(f"  {i}. {info['name']} - {info['singer']} (来自一个包含 {cluster_size} 首相似歌曲的簇)")

            except KeyboardInterrupt:
                print("\n感谢使用，再见！")
                break

    def _get_sem_id_for_song(self, song_id_to_find: str) -> Tuple[int, ...]:
        # Helper to find the semantic ID for a given song ID (for display purposes)
        for sem_id, song_list in self.semantic_to_song_cluster.items():
            if song_id_to_find in song_list:
                return sem_id
        return None

if __name__ == "__main__":
    config = Config()
    logger = setup_logging(level=logging.WARNING)
    # Add defaultdict to main scope for the class to use
    from collections import defaultdict

    generator = PlaylistGenerator(config)
    generator.interactive_demo()