import praw
import pandas as pd
import time

# --- CONNECT TO REDDIT ---
reddit = praw.Reddit(
    client_id="po9US0Pej05iDzAg4d2ztQ",
    client_secret="VFdTIqrbDS_Ie1s7UXpaSzNi1TZscA",
    user_agent="surgery_sentiment/0.01"
)

# --- SET UP SEARCH ---
subreddit = reddit.subreddit("all")

data = []  # store results here
query = "autonomous robotic surgery"

for submission in subreddit.search(query, limit=50):
    # --- Collect post info ---
    post_text = submission.title + "\n" + submission.selftext
    data.append({
        "type": "post",
        "subreddit": submission.subreddit.display_name,
        "text": post_text,
        "score": submission.score,
        "url": submission.url
    })

    # --- Collect top-level comments ---
    submission.comments.replace_more(limit=0)
    for comment in submission.comments[:5]:  # get first 5 comments for each post
        data.append({
            "type": "comment",
            "subreddit": submission.subreddit.display_name,
            "text": comment.body,
            "score": comment.score,
            "url": submission.url
        })
    time.sleep(1)  # to avoid rate limiting

# --- SAVE TO CSV ---
df = pd.DataFrame(data)
df.to_csv("reddit_robotic_surgery.csv", index=False)

print(f"Saved {len(df)} rows to reddit_robotic_surgery.csv ✅")
