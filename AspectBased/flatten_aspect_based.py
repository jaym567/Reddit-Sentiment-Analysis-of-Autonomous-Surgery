import json
import csv

INPUT_FILE = r"AspectBased\reddit_robotic_surgery_aspect_sentiment.json"
OUTPUT_FILE = r"AspectBased\aspect_reddit_sentiment_flat.csv"


def flatten_reddit_json(data):
    rows = []

    for post in data:
        # -------- POST ROW -------- #
        rows.append({
            "type": "post",
            "id": post["id"],
            "parent_id": None,
            "subreddit": post.get("subreddit"),
            "author": None,
            "created_utc": post.get("created_utc"),
            "score": post.get("score"),
            "text": (post.get("title", "") + "\n" + post.get("selftext", "")).strip(),
            "aspect": post.get("aspect_sentiment", {}).get("aspect"),
            "sentiment_label": post.get("aspect_sentiment", {}).get("label"),
            "sentiment_confidence": post.get("aspect_sentiment", {}).get("confidence"),
            "depth": 0,
            "permalink": post.get("permalink")
        })

        # -------- COMMENT WALK -------- #
        def walk_comments(comments, parent_id, depth):
            for c in comments:
                rows.append({
                    "type": "comment",
                    "id": c["id"],
                    "parent_id": parent_id,
                    "subreddit": post.get("subreddit"),
                    "author": c.get("author"),
                    "created_utc": c.get("created_utc"),
                    "score": c.get("score"),
                    "text": c.get("body", "").strip(),
                    "aspect": c.get("aspect_sentiment", {}).get("aspect"),
                    "sentiment_label": c.get("aspect_sentiment", {}).get("label"),
                    "sentiment_confidence": c.get("aspect_sentiment", {}).get("confidence"),
                    "depth": depth,
                    "permalink": c.get("permalink")
                })

                walk_comments(c.get("replies", []), c["id"], depth + 1)

        walk_comments(post.get("comments", []), post["id"], depth=1)

    return rows


def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = flatten_reddit_json(data)

    fieldnames = [
        "type",
        "id",
        "parent_id",
        "subreddit",
        "author",
        "created_utc",
        "score",
        "text",
        "aspect",
        "sentiment_label",
        "sentiment_confidence",
        "depth",
        "permalink"
    ]

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ Flattened CSV saved to: {OUTPUT_FILE}")
    print(f"📊 Total rows: {len(rows)}")


if __name__ == "__main__":
    main()
