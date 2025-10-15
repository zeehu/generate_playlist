
Debug script to detect and report semantic ID collisions.

A collision occurs when two or more different song_ids are mapped to the
exact same semantic ID sequence by the quantization algorithm (RQ-VAE or RQ-KMeans).
"""
import os
import sys
import json
from collections import defaultdict
from tqdm import tqdm
import logging

# Adjust path to import from playlist_src
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from config import Config
from utils import setup_logging

logger = logging.getLogger(__name__)

def find_collisions(config: Config):
    """Finds and reports on semantic ID collisions."""
    logger.info("--- Starting Semantic ID Collision Analysis ---")
    
    semantic_ids_file = os.path.join(config.output_dir, "song_semantic_ids.jsonl")
    if not os.path.exists(semantic_ids_file):
        logger.error(f"FATAL: Semantic ID file not found at {semantic_ids_file}")
        logger.error("Please run a Phase 1 script (train_rqvae.py or train_rq_kmeans.py) first.")
        return

    logger.info(f"Reading semantic IDs from {semantic_ids_file}...")
    
    # A dictionary mapping a semantic ID tuple to a list of song IDs
    reverse_map = defaultdict(list)

    with open(semantic_ids_file, 'r') as f:
        for line in tqdm(f, desc="Building reverse map"):
            item = json.loads(line)
            # Use a tuple as a dictionary key
            key = tuple(item['semantic_ids'])
            reverse_map[key].append(item['song_id'])

    logger.info("Analyzing for collisions...")
    
    collisions = []
    for semantic_id_tuple, song_id_list in reverse_map.items():
        if len(song_id_list) > 1:
            collisions.append((semantic_id_tuple, song_id_list))

    # --- Print Report ---
    print("\n" + "="*80)
    print("  Semantic ID Collision Report")
    print("="*80)

    if not collisions:
        print("\n🎉 Congratulations! No collisions found.")
        print("This means every song has a unique semantic ID.")
    else:
        total_songs_affected = sum(len(songs) for _, songs in collisions)
        print(f"\nFound {len(collisions)} unique semantic IDs that have collisions.")
        print(f"A total of {total_songs_affected} songs are involved in these collisions.")
        print("\n--- Example Collisions (showing first 5) ---")
        
        for i, (semantic_id, songs) in enumerate(collisions[:5]):
            print(f"\n{i+1}. Semantic ID {semantic_id} is shared by {len(songs)} songs:")
            print(f"   {songs}")
            
    print("\n" + "="*80)

if __name__ == "__main__":
    config = Config()
    # We don't need verbose logging for this script
    logger = setup_logging(level=logging.INFO)
    # Suppress other loggers for cleaner output
    logging.getLogger("utils").setLevel(logging.WARNING)

    find_collisions(config)
