import json
import torch
from transformers import pipeline
from tqdm import tqdm
import os

# ---------------- CONFIG ---------------- #
INPUT_FILE = r"c:\Users\jaymo\OneDrive\Desktop\ARGOS\Sentiment Analysis of Autonomous Surgery\SentimentCode\filtered_reddit_robotic_surgery.json"
OUTPUT_FILE = r"c:\Users\jaymo\OneDrive\Desktop\ARGOS\Sentiment Analysis of Autonomous Surgery\SentimentCode\filtered_reddit_robotic_surgery_analyzed.json"

# General Sentiment Config
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
SENTIMENT_BATCH_SIZE = 32

# Aspect Analysis Config
ASPECT = "autonomous surgery"
ASPECT_LABELS = [
    "Supportive of autonomous surgery",
    "Opposed to autonomous surgery",
    "Cautious or mixed about autonomous surgery",
    "Neutral or informational about autonomous surgery"
]
ASPECT_MODEL = "facebook/bart-large-mnli"
ASPECT_BATCH_SIZE = 16  # BART is larger, smaller batch size to avoid OOM

# ---------------------------------------- #

def run_analysis():
    # 1. Check CUDA
    device = 0 if torch.cuda.is_available() else -1
    print(f"CUDA available: {torch.cuda.is_available()}")
    if device == 0:
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: CUDA not available, using CPU (this will be slow)")

    # 2. Load Data
    print(f"Loading filtered data from {INPUT_FILE}...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Flatten items for batched inference
    flattened_items = []
    for post_group in data:
        for item in post_group['items']:
            flattened_items.append(item)

    print(f"Total items to analyze: {len(flattened_items)}")

    # 3. Initialize Pipelines
    print("Initialising General Sentiment Pipeline...")
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model=SENTIMENT_MODEL,
        device=device,
        truncation=True
    )

    print("Initialising Aspect-Based Pipeline...")
    aspect_pipeline = pipeline(
        "zero-shot-classification",
        model=ASPECT_MODEL,
        device=device,
        torch_dtype=torch.float16 if device == 0 else torch.float32
    )
    hypothesis_template = "This text expresses {}."

    # 4. Run General Sentiment Analysis (Batched)
    print("Running General Sentiment Analysis...")
    for i in tqdm(range(0, len(flattened_items), SENTIMENT_BATCH_SIZE), desc="Sentiment"):
        batch = flattened_items[i:i + SENTIMENT_BATCH_SIZE]
        texts = [str(x['text']) if x['text'] else " " for x in batch]
        
        try:
            results = sentiment_pipeline(texts, max_length=512, truncation=True)
            for item, result in zip(batch, results):
                label = result['label'].upper()
                score = float(result['score'])
                if score < 0.6: label = "MIXED"
                item['sentiment'] = {"label": label, "confidence": score}
        except Exception as e:
            print(f"\nError in Sentiment batch starting at {i}: {e}")
            for item in batch: item['sentiment'] = None

    # 5. Run Aspect-Based Analysis (Batched)
    print("Running Aspect-Based Sentiment Analysis...")
    for i in tqdm(range(0, len(flattened_items), ASPECT_BATCH_SIZE), desc="Aspect"):
        batch = flattened_items[i:i + ASPECT_BATCH_SIZE]
        texts = [str(x['text']) if x['text'] else " " for x in batch]
        
        try:
            results = aspect_pipeline(
                texts,
                candidate_labels=ASPECT_LABELS,
                hypothesis_template=hypothesis_template,
                multi_label=False
            )
            if isinstance(results, dict): results = [results]
            for item, result in zip(batch, results):
                item['aspect_sentiment'] = {
                    "aspect": ASPECT,
                    "label": result['labels'][0],
                    "confidence": float(result['scores'][0])
                }
        except Exception as e:
            print(f"\nError in Aspect batch starting at {i}: {e}")
            for item in batch: item['aspect_sentiment'] = None

    # 6. Save Results
    print(f"Saving analyzed data to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print("\n✅ Analysis complete.")
    print(f"📁 Analyzed dataset: {OUTPUT_FILE}")

if __name__ == "__main__":
    run_analysis()
