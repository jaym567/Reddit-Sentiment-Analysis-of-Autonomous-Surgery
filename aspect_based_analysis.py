from transformers import pipeline
import json

ASPECT = "autonomous surgery"

LABELS = [
    "Supportive of autonomous surgery",
    "Opposed to autonomous surgery",
    "Cautious or mixed about autonomous surgery",
    "Neutral or informational about autonomous surgery"
]

def extract_texts(data):
    items = []

    for post in data:
        items.append({
            "type": "post",
            "id": post["id"],
            "text": (post.get("title", "") + "\n" + post.get("selftext", "")).strip()
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
    lookup = {x["id"]: x["sentiment"] for x in sentiment_results}

    for post in data:
        post["aspect_sentiment"] = lookup.get(post["id"])

        def walk(comments):
            for c in comments:
                c["aspect_sentiment"] = lookup.get(c["id"])
                walk(c.get("replies", []))

        walk(post.get("comments", []))

    return data


if __name__ == "__main__":
    with open("reddit_robotic_surgery.json", "r", encoding="utf-8") as f:
        reddit_data = json.load(f)

    classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli"
    )

    texts = extract_texts(reddit_data)

    for item in texts:
        text = item["text"]

        if not text:
            item["sentiment"] = None
            continue

        # Ask specifically about the aspect
        hypothesis_template = f"This text expresses {{}}."

        result = classifier(
            text,
            candidate_labels=LABELS,
            hypothesis_template=hypothesis_template,
            multi_label=False
        )

        item["sentiment"] = {
            "aspect": ASPECT,
            "label": result["labels"][0],
            "confidence": float(result["scores"][0])
        }

    annotated_data = inject_sentiment(reddit_data, texts)

    with open("reddit_robotic_surgery_aspect_sentiment.json", "w", encoding="utf-8") as f:
        json.dump(annotated_data, f, indent=2)

    print("Aspect-based sentiment analysis complete.")
