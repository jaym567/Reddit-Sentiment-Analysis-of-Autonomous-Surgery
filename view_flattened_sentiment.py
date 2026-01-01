import json
import pandas as pd

with open("reddit_robotic_surgery_sentiment.json", "r", encoding="utf-8") as f:
    data = json.load(f)

rows = []

def walk_comments(comments, post_id, post_subreddit):
    for c in comments:
        rows.append({
            "post_id": post_id,
            "comment_id": c["id"],
            "type": "comment",
            "subreddit": c.get("subreddit", post_subreddit),  # fallback to post's subreddit
            "created_utc": c.get("created_utc", None),
            "score": c.get("score", None),
            "text": c.get("body", ""),
            "sentiment_label": c.get("sentiment", {}).get("label"),
            "sentiment_score": c.get("sentiment", {}).get("score")
        })
        walk_comments(c.get("replies", []), post_id, post_subreddit)

for post in data:
    rows.append({
        "post_id": post["id"],
        "comment_id": None,
        "type": "post",
        "subreddit": post.get("subreddit", ""),
        "created_utc": post.get("created_utc", None),
        "score": post.get("score", None),
        "text": post.get("title", "") + "\n" + post.get("selftext", ""),
        "sentiment_label": post.get("sentiment", {}).get("label"),
        "sentiment_score": post.get("sentiment", {}).get("score")
    })
    walk_comments(post.get("comments", []), post["id"], post.get("subreddit", ""))

df = pd.DataFrame(rows)

print(df.head())
print(df["sentiment_label"].value_counts())

df.to_csv("reddit_sentiment_flat.csv", index=False)
