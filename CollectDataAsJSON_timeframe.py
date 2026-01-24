import praw
import json
import time
import requests
from datetime import datetime, timedelta

# ========================
# REDDIT SETUP
# ========================
reddit = praw.Reddit(
    client_id="po9US0Pej05iDzAg4d2ztQ",
    client_secret="VFdTIqrbDS_Ie1s7UXpaSzNi1TZscA",
    user_agent="surgery_sentiment_temporal/1.0"
)

# ========================
# SETTINGS
# ========================
queries = [
    "autonomous robotic surgery",
    "AI surgery",
    "autonomous surgery robot",
    "SRT-H robot",
    "autonomous da Vinci"
]

start_epoch = int(datetime(2020, 1, 1).timestamp())
end_epoch = int(datetime.now().timestamp())
MAX_COMMENTS_PER_POST = 50
REQUEST_DELAY = 1.0
output_file = "reddit_robotic_surgery_temporal_flat.json"

# ========================
# HELPERS
# ========================
def month_bucket(ts):
    return datetime.utcfromtimestamp(ts).strftime("%Y-%m")

def pullpush_search(query, after, before, size=500):
    """Fetch submission IDs from PullPush mirror."""
    url = "https://api.pullpush.io/reddit/search/submission/"
    params = {
        "q": query,
        "after": after,
        "before": before,
        "size": size,
        "sort": "asc"
    }
    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        return r.json().get("data", [])
    except Exception as e:
        print(f"⚠️ PullPush error: {e}")
        return []

def flatten_comments(submission, max_comments=50):
    flat_comments = []
    submission.comments.replace_more(limit=0)
    for c in submission.comments.list()[:max_comments]:
        body = getattr(c, 'body', '')
        if not body: continue
        flat_comments.append({
            "type": "comment",
            "id": c.id,
            "comment_id": c.id,
            "post_id": submission.id,
            "parent_id": str(getattr(c, 'parent_id', submission.id)),
            "author": str(c.author) if c.author else "[deleted]",
            "body": body,
            "created_utc": c.created_utc,
            "month": month_bucket(c.created_utc),
            "score": c.score
        })
    return flat_comments

# ========================
# MAIN SCRAPE
# ========================
results = []
seen_ids = set()

print(f"🚀 Starting Temporal Scrape (2020–now) via PullPush")

# Loop month by month for search reliability
current_start = start_epoch
while current_start < end_epoch:
    current_end = min(current_start + 30*24*3600, end_epoch) # ~1 month
    month_str = datetime.utcfromtimestamp(current_start).strftime("%Y-%m")
    
    print(f"\n📅 Period: {month_str}")
    
    for query in queries:
        ps_posts = pullpush_search(query, current_start, current_end)
        if not ps_posts: continue
        
        print(f"  🔍 {query}: Found {len(ps_posts)} potential posts")
        
        for ps_post in ps_posts:
            pid = ps_post.get('id')
            if not pid or pid in seen_ids: continue
            seen_ids.add(pid)
            
            try:
                submission = reddit.submission(id=pid)
                # Ensure it's fully loaded
                _ = submission.title 
                
                post_data = {
                    "type": "post",
                    "id": submission.id,
                    "post_id": submission.id,
                    "query": query,
                    "subreddit": submission.subreddit.display_name,
                    "title": submission.title,
                    "body": submission.selftext or "",
                    "selftext": submission.selftext or "",
                    "created_utc": submission.created_utc,
                    "month": month_bucket(submission.created_utc),
                    "score": submission.score,
                    "num_comments": submission.num_comments,
                    "permalink": f"https://reddit.com{submission.permalink}"
                }
                results.append(post_data)
                
                # Fetch comments
                comments = flatten_comments(submission, MAX_COMMENTS_PER_POST)
                results.extend(comments)
                
                # Progress logging
                if len(seen_ids) % 10 == 0:
                    print(f"    📦 Collected {len(results)} items ({len(seen_ids)} unique posts)")
                
                time.sleep(REQUEST_DELAY)
                
            except Exception as e:
                print(f"    ⚠️ Hydration error for {pid}: {e}")
                time.sleep(2)
                
    current_start = current_end

# ========================
# SAVE
# ========================
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=1)

print(f"\n✅ Scrape Complete!")
print(f"📊 Total records: {len(results)}")
print(f"📦 Unique posts: {len(seen_ids)}")
print(f"💾 Saved to {output_file}")
