import praw

# Replace these with your Reddit app credentials
reddit = praw.Reddit(
    client_id="po9US0Pej05iDzAg4d2ztQ",
    client_secret="VFdTIqrbDS_Ie1s7UXpaSzNi1TZscA",
    user_agent="surgery_sentiment/0.01"
)

# Try searching for posts mentioning robotic surgery
subreddit = reddit.subreddit("all")
for submission in subreddit.search("autonomous robotic surgery", limit=5):
    print("Title:", submission.title)
    print("Subreddit:", submission.subreddit.display_name)
    print("Score:", submission.score)
    print("URL:", submission.url)
    print("-" * 80)