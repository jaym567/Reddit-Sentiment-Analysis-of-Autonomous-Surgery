from transformers import pipeline
import torch
import json
from tqdm import tqdm

# ---------------- CONFIG ---------------- #

ASPECT = "autonomous surgery"

LABELS = [
    "Supportive of autonomous surgery",
    "Opposed to autonomous surgery",
    "Cautious or mixed about autonomous surgery",
    "Neutral or informational about autonomous surgery"
]

MODEL_NAME = "facebook/bart-large-mnli"
BATCH_SIZE = 16  # RTX 4060 safe (lower to 8 if VRAM spikes)

INPUT_FILE = "reddit_robotic_surgery.json"
OUTPUT_FILE = "reddit_robotic_surgery_aspect_sentiment.json"

# ---------------------------------------- #


def extract_texts(data):
    """Flatten posts + comments into a list."""
    items = []

    for post in data:
        post_text = (post.get("title", "") + "\n" + post.get("selftext", "")).strip()
        items.append({
            "type": "post",
            "id": post["id"],
            "text": post_text
        })

        def walk_comments(comments):
            for c in comments:
                items.append({
                    "type": "comment",
                    "id": c["id"],
                    "text": c.get("body", "").strip()
                })
                walk_comments(c.get("replies", []))

        walk_comments(post.get("comments", []))

    return items


def inject_sentiment(data, sentiment_results):
    """Reattach sentiment into original hierarchy."""
    lookup = {x["id"]: x.get("sentiment") for x in sentiment_results}

    for post in data:
        post["aspect_sentiment"] = lookup.get(post["id"])

        def walk(comments):
            for c in comments:
                c["aspect_sentiment"] = lookup.get(c["id"])
                walk(c.get("replies", []))

        walk(post.get("comments", []))

    return data


def main():
    # ---------- LOAD DATA ---------- #
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        reddit_data = json.load(f)

    items = extract_texts(reddit_data)

    # ---------- GPU CHECK ---------- #
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        raise RuntimeError("CUDA not available — aborting to avoid CPU meltdown.")

    # ---------- LOAD MODEL ---------- #
    classifier = pipeline(
        "zero-shot-classification",
        model=MODEL_NAME,
        device=0,
        torch_dtype=torch.float16  # halves VRAM usage
    )

    hypothesis_template = "This text expresses {}."

    # ---------- BATCHED INFERENCE ---------- #
    for i in tqdm(range(0, len(items), BATCH_SIZE), desc="Analyzing sentiment"):
        batch = items[i:i + BATCH_SIZE]
        batch_texts = [x["text"] for x in batch if x["text"]]

        if not batch_texts:
            continue

        results = classifier(
            batch_texts,
            candidate_labels=LABELS,
            hypothesis_template=hypothesis_template,
            multi_label=False
        )

        # Handle single-item edge case
        if isinstance(results, dict):
            results = [results]

        for item, result in zip(batch, results):
            item["sentiment"] = {
                "aspect": ASPECT,
                "label": result["labels"][0],
                "confidence": float(result["scores"][0])
            }

    # ---------- REBUILD TREE ---------- #
    annotated_data = inject_sentiment(reddit_data, items)

    # ---------- SAVE ---------- #
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(annotated_data, f, indent=2)

    print("\n✅ Aspect-based sentiment analysis complete.")
    print(f"📁 Output saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
