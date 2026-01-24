import json
import logging
import os
from typing import List, Dict

# Set up logging to show progress but not overwhelm with debug info
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from relevance_filter.relevance_filter import RelevanceFilter
from relevance_filter.keyword_filter import KeywordFilter
from relevance_filter.semantic_classifier import SemanticClassifier
from relevance_filter.models import FilterConfig

def filter_dataset(input_path: str, output_path: str):
    """
    Load Reddit dataset, apply relevance filter, and save results.
    """
    logger.info(f"Loading dataset from {input_path}...")
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    logger.info(f"Initial raw item count: {len(raw_data)}")

    # Unflatten logic: group comments under posts
    posts_map = {}
    for item in raw_data:
        if item.get('type') == 'post':
            pid = item.get('post_id')
            if pid not in posts_map:
                posts_map[pid] = item
                posts_map[pid]['comments'] = []
            else:
                # Update metadata if post arrived after its comments
                for k, v in item.items():
                    posts_map[pid][k] = v
        elif item.get('type') == 'comment':
            pid = item.get('post_id')
            if pid not in posts_map:
                # Placeholder post if comment precedes post record
                posts_map[pid] = {'id': pid, 'post_id': pid, 'type': 'post', 'comments': []}
            posts_map[pid]['comments'].append(item)

    data = list(posts_map.values())
    logger.info(f"Reconstructed {len(data)} post trees from flattened data.")

    # Initialize components
    config = FilterConfig()
    kw_filter = KeywordFilter()
    semantic_classifier = SemanticClassifier(model_type="embedding")
    
    relevance_filter = RelevanceFilter(kw_filter, semantic_classifier, config)

    logger.info("Applying relevance filter...")
    filtered_results = relevance_filter.filter_posts(data)
    
    logger.info(f"Filtering complete. Total relevant items found: {len(filtered_results)}")

    # Group results by post for better readability/organization in output
    organized_results = {}
    for item in filtered_results:
        post_id = item['post_id']
        if post_id not in organized_results:
            organized_results[post_id] = []
        organized_results[post_id].append(item)

    # Convert back to list format for final JSON
    final_output = [
        {
            "post_id": pid,
            "items": items
        }
        for pid, items in organized_results.items()
    ]

    logger.info(f"Saving filtered results to {output_path}...")
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, indent=2, ensure_ascii=False)
        logger.info("Successfully saved filtered data.")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")

if __name__ == "__main__":
    input_file = r"c:\Users\jaymo\OneDrive\Desktop\ARGOS\Sentiment Analysis of Autonomous Surgery\SentimentCode\reddit_robotic_surgery_temporal_flat.json"
    output_file = r"c:\Users\jaymo\OneDrive\Desktop\ARGOS\Sentiment Analysis of Autonomous Surgery\SentimentCode\filtered_reddit_robotic_surgery.json"
    
    filter_dataset(input_file, output_file)
