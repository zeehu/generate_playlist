

"""
import os
import sys
import torch
import json
import logging
from typing import Dict, Tuple, List

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
        self.semantic_to_song_map = self._create_reverse_map()
        self.song_info_map = self._load_song_info()
        self.semantic_id_levels = self.config.rqvae.levels

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

    def _create_reverse_map(self) -> Dict[Tuple[int, ...], str]:
        semantic_ids_file = os.path.join(self.config.output_dir, "song_semantic_ids.jsonl")
        if not os.path.exists(semantic_ids_file):
            logger.error(f"FATAL: Semantic ID file not found at {semantic_ids_file}")
            logger.error("Please run Phase 1 (train_rqvae.py) first.")
            sys.exit(1)

        logger.info("Creating reverse mapping from semantic IDs to song IDs...")
        mapping = {}
        with open(semantic_ids_file, 'r') as f:
            for line in f:
                item = json.loads(line)
                key = tuple(item['semantic_ids'])
                mapping[key] = item['song_id']
        return mapping

    def _load_song_info(self) -> Dict[str, Dict[str, str]]:
        song_info_file = self.config.data.song_info_file
        if not os.path.exists(song_info_file):
            logger.warning(f"Song info file not found at {song_info_file}. Song names will not be displayed.")
            return {}

        logger.info(f"Loading song info from {song_info_file}...")
        mapping = {}
        try:
            with open(song_info_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                # Skip header if exists, simple check
                first_row = next(reader)
                if 'mixsongid' not in first_row[0]:
                    self._process_song_info_row(first_row, mapping)
                
                for row in reader:
                    self._process_song_info_row(row, mapping)
            logger.info(f"Loaded info for {len(mapping)} songs.")
        except Exception as e:
            logger.error(f"Error reading song info file: {e}")
            return {}
        return mapping

    def _process_song_info_row(self, row: List[str], mapping: Dict):
        if len(row) >= 3:
            song_id, song_name, singer_name = row[0], row[1], row[2]
            mapping[song_id] = {"name": song_name, "singer": singer_name}

    def generate(self, title: str, tags: str = "") -> List[str]:
        """Generates a playlist for a given title and optional tags."""
        prompt = f"歌单标题：{title} | 歌单标签：{tags}"
        logger.info(f"Generating with prompt: {prompt}")

        input_ids = self.model.tokenizer.base_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.tiger.max_input_length
        ).input_ids.to(self.device)

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
        
        decoded_tokens = self.model.tokenizer.base_tokenizer.convert_ids_to_tokens(
            generated_ids[0], skip_special_tokens=True
        )

        numerical_ids = []
        for token in decoded_tokens:
            if token.startswith("<id_") and token.endswith(">"):
                try:
                    numerical_ids.append(int(token[4:-1]))
                except ValueError:
                    continue
        
        chunk_size = self.semantic_id_levels
        reconstructed_song_ids = []
        for i in range(0, len(numerical_ids) - chunk_size + 1, chunk_size):
            id_chunk = tuple(numerical_ids[i : i + chunk_size])
            if id_chunk in self.semantic_to_song_map:
                reconstructed_song_ids.append(self.semantic_to_song_map[id_chunk])
        
        return reconstructed_song_ids

    def interactive_demo(self):
        """Starts an interactive command-line demo."""
        print("\n" + "="*50)
        print("  🎶 歌单生成模型已就绪 🎶")
        print("="*50)
        print("  输入一个你想要的歌单标题，然后按 Enter。")
        print("  例如: '适合深夜一个人听的安静歌曲' 或 '周末清晨的咖啡馆背景音乐'")
        print("  输入 'exit' 或 'quit' 即可退出。")
        print("-"*50)

        while True:
            try:
                prompt = input("\n请输入歌单标题 > ")
                if prompt.lower() in ['exit', 'quit']:
                    print("感谢使用，再见！")
                    break
                
                if not prompt:
                    continue

                print("\n生成中，请稍候...")
                song_ids = self.generate(prompt)

                if not song_ids:
                    print("模型未能生成有效的歌曲列表，请尝试更换标题。")
                    continue
                
                print("\n✨ 为您推荐的歌曲列表: ✨")
                for i, song_id in enumerate(song_ids, 1):
                    info = self.song_info_map.get(song_id)
                    if info:
                        print(f"  {i}. {info['name']} - {info['singer']} (ID: {song_id})")
                    else:
                        print(f"  {i}. (ID: {song_id}) - 歌曲信息未找到")

            except KeyboardInterrupt:
                print("\n感谢使用，再见！")
                break

if __name__ == "__main__":
    config = Config()
    logger = setup_logging(level=logging.WARNING)

    if config.data.song_info_file == "path/to/your/gen_song_info.csv":
        logger.warning("="*80)
        logger.warning("提示: 您还未在 'playlist_src/config.py' 中配置 'song_info_file' 的路径。")
        logger.warning("歌曲的详细信息（歌名、歌手）将不会被显示。")
        logger.warning("="*80)

    generator = PlaylistGenerator(config)
    generator.interactive_demo()
