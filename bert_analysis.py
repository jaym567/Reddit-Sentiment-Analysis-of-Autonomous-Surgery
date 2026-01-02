from transformers import pipeline
import json


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
        post["sentiment"] = lookup.get(post["id"])

        def walk(comments):
            for c in comments:
                c["sentiment"] = lookup.get(c["id"])
                walk(c.get("replies", []))

        walk(post.get("comments", []))

    return data


if __name__ == "__main__":
    with open("reddit_robotic_surgery.json", "r", encoding="utf-8") as f:
        reddit_data = json.load(f)

    sentiment_model = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        truncation=True
    )

    texts = extract_texts(reddit_data)

    for item in texts:
        text = item["text"]

        if not text:
            item["sentiment"] = None
            continue

        result = sentiment_model(
            text,
            max_length=512,
            truncation=True
        )[0]

        label = result["label"].upper()
        score = float(result["score"])

        # Optional: collapse low-confidence predictions into MIXED
        if score < 0.6:
            label = "MIXED"

        item["sentiment"] = {
            "label": label,
            "confidence": score
        }

    annotated_data = inject_sentiment(reddit_data, texts)

    with open("reddit_robotic_surgery_sentiment_improved.json", "w", encoding="utf-8") as f:
        json.dump(annotated_data, f, indent=2)

    print("Improved sentiment analysis complete.")
