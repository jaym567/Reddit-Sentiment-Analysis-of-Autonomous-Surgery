import praw
import json
import time

# --- CONNECT TO REDDIT ---
reddit = praw.Reddit(
    client_id="po9US0Pej05iDzAg4d2ztQ",
    client_secret="VFdTIqrbDS_Ie1s7UXpaSzNi1TZscA",
    user_agent="surgery_sentiment/0.01"
)

# --- SETTINGS ---
query = "autonomous robotic surgery"
limit_posts = 50
max_depth = 3  # top-level + 3 reply levels
output_file = "reddit_robotic_surgery.json"


# ------------ COMMENT PARSING FUNCTION ---------------
def parse_comment(comment, depth=1):
    """Recursively parse a comment and its replies up to max_depth."""

    if depth > max_depth:
        return None

    # Basic metadata
    c_data = {
        "id": comment.id,
        "author": str(comment.author) if comment.author else "deleted",
        "body": comment.body,
        "created_utc": comment.created_utc,
        "score": comment.score,
        "permalink": "https://reddit.com" + comment.permalink,
        "replies": []
    }

    # Replace "MoreComments" objects
    comment.replies.replace_more(limit=0)

    for reply in comment.replies:
        r = parse_comment(reply, depth + 1)
        if r is not None:
            c_data["replies"].append(r)

    return c_data


# ------------- MAIN SCRAPER ----------------
subreddit = reddit.subreddit("all")
results = []

print("🔍 Searching Reddit...")

for submission in subreddit.search(query, limit=limit_posts):
    print(f"Collecting: {submission.title[:50]}...")

    submission.comments.replace_more(limit=0)

    post_data = {
        "id": submission.id,
        "subreddit": submission.subreddit.display_name,
        "title": submission.title,
        "selftext": submission.selftext,
        "created_utc": submission.created_utc,
        "score": submission.score,
        "url": submission.url,
        "permalink": f"https://reddit.com{submission.permalink}",
        "comments": []
    }

    # Parse top 5 top-level comments
    for top_comment in submission.comments[:5]:
        c = parse_comment(top_comment, depth=1)
        if c is not None:
            post_data["comments"].append(c)

    results.append(post_data)
    time.sleep(1)

# ------------- SAVE JSON ----------------
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Saved {len(results)} posts to {output_file}")
