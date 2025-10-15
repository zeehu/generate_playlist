import os
import sys
import pandas as pd
import json
from tqdm import tqdm
import logging
from sklearn.model_selection import train_test_split
import random

# Adjust path to import from playlist_src
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from config import Config
from utils import setup_logging

logger = logging.getLogger(__name__)

class CorpusBuilder:
    """Orchestrates the generation of the training corpus."""

    def __init__(self, config: Config):
        self.config = config
        self.data_config = config.data
        self.tiger_config = config.tiger

    def run(self):
        logger.info("--- Starting Phase 2: Training Corpus Generation ---")

        # 1. Load all data sources
        semantic_id_map = self._load_semantic_ids()
        playlist_info = self._load_playlist_info()
        playlist_songs = self._load_playlist_songs()

        # 2. Build the full corpus
        corpus = self._build_corpus(playlist_info, playlist_songs, semantic_id_map)

        # 3. Split and save the data
        self._split_and_save(corpus)

        logger.info("--- Phase 2 Completed Successfully ---")

    def _load_semantic_ids(self) -> dict:
        logger.info(f"Loading semantic IDs from {self.data_config.semantic_ids_file}...")
        if not os.path.exists(self.data_config.semantic_ids_file):
            logger.error(f"FATAL: Semantic ID file not found. Please run Phase 1 first.")
            sys.exit(1)
        
        mapping = {}
        with open(self.data_config.semantic_ids_file, 'r') as f:
            for line in f:
                item = json.loads(line)
                mapping[item['song_id']] = item['semantic_ids']
        logger.info(f"Loaded {len(mapping)} song-to-semantic-ID mappings.")
        return mapping

    def _load_playlist_info(self) -> dict:
        logger.info(f"Loading playlist info from {self.data_config.playlist_info_file}...")
        try:
            df = pd.read_csv(self.data_config.playlist_info_file, dtype=str)
            df.set_index('glid', inplace=True)
            info_dict = df.to_dict('index')
            logger.info(f"Loaded info for {len(info_dict)} playlists.")
            return info_dict
        except FileNotFoundError:
            logger.error(f"FATAL: Playlist info file not found at {self.data_config.playlist_info_file}")
            logger.error("Please update the path in 'playlist_src/config.py'.")
            sys.exit(1)

    def _load_playlist_songs(self) -> dict:
        logger.info(f"Loading playlist songs from {self.data_config.playlist_songs_file}...")
        try:
            # Enforce string type for ID columns to prevent mismatches
            df = pd.read_csv(self.data_config.playlist_songs_file, dtype=str)
            grouped = df.groupby('special_gid')['mixsongid'].apply(list)
            songs_dict = grouped.to_dict()
            logger.info(f"Loaded song lists for {len(songs_dict)} playlists.")
            return songs_dict
        except FileNotFoundError:
            logger.error(f"FATAL: Playlist songs file not found at {self.data_config.playlist_songs_file}")
            logger.error("Please update the path in 'playlist_src/config.py'.")
            sys.exit(1)

    def _build_corpus(self, playlist_info: dict, playlist_songs: dict, semantic_id_map: dict) -> list:
        logger.info("Building text-to-text corpus...")
        corpus = []
        
        for glid, songs in tqdm(playlist_songs.items(), desc="Processing playlists"):
            if glid not in playlist_info:
                continue

            info = playlist_info[glid]
            title = info.get('listname', '')
            tags = info.get('tag_list', '')
            input_text = f"歌单标题：{title} | 歌单标签：{tags}"

            sorted_songs = sorted(songs)
            
            semantic_tokens = []
            for song_id in sorted_songs:
                if song_id in semantic_id_map:
                    tokens = [f"<id_{sid}>" for sid in semantic_id_map[song_id]]
                    semantic_tokens.extend(tokens)
            
            if not semantic_tokens:
                continue

            # We now embrace collisions, so we do not de-duplicate songs or tokens.
            # We just sort the original song list to have a consistent (but not unique) order.
            sorted_songs = sorted(songs)
            
            semantic_tokens = []
            for song_id in sorted_songs:
                if song_id in semantic_id_map:
                    # Format semantic IDs into tokens
                    tokens = [f"<id_{sid}>" for sid in semantic_id_map[song_id]]
                    semantic_tokens.extend(tokens)
            
            if not semantic_tokens:
                continue

            # Truncate to max target length, leaving space for <eos>
            max_len = self.tiger_config.max_target_length - 1
            truncated_tokens = semantic_tokens[:max_len]
            
            output_sequence = " ".join(truncated_tokens) + " <eos>"
            
            corpus.append((glid, input_text, output_sequence))
        
        logger.info(f"Successfully built corpus with {len(corpus)} entries.")
        return corpus

    def _split_and_save(self, corpus: list):
        logger.info("Splitting data and saving to files...")
        
        random.shuffle(corpus)

        train_ratio = self.data_config.train_split_ratio
        val_ratio = self.data_config.val_split_ratio
        
        train_end_idx = int(len(corpus) * train_ratio)
        val_end_idx = train_end_idx + int(len(corpus) * val_ratio)

        train_data = corpus[:train_end_idx]
        val_data = corpus[train_end_idx:val_end_idx]
        test_data = corpus[val_end_idx:]

        logger.info(f"Data split: {len(train_data)} train, {len(val_data)} validation, {len(test_data)} test.")

        output_dir = self.config.output_dir
        self._save_to_tsv(train_data, os.path.join(output_dir, "train.tsv"))
        self._save_to_tsv(val_data, os.path.join(output_dir, "val.tsv"))
        self._save_to_tsv(test_data, os.path.join(output_dir, "test.tsv"))

    def _save_to_tsv(self, data: list, file_path: str):
        logger.info(f"Saving {len(data)} records to {file_path}...")
        with open(file_path, 'w', encoding='utf-8') as f:
            for glid, input_text, output_sequence in data:
                f.write(f"{glid}\t{input_text}\t{output_sequence}\n")

if __name__ == "__main__":
    config = Config()
    log_file_path = os.path.join(config.log_dir, "phase2_prepare_corpus.log")
    logger = setup_logging(log_file_path)

    if config.data.playlist_info_file == "path/to/your/gen_playlist_info.csv" or \
       config.data.playlist_songs_file == "path/to/your/gen_playlist_song.csv":
        logger.error("="*80)
        logger.error("FATAL: Please edit 'playlist_src/config.py' and set the paths for")
        logger.error("'playlist_info_file' and 'playlist_songs_file'.")
        logger.error("="*80)
        sys.exit(1)

    builder = CorpusBuilder(config)
    builder.run()