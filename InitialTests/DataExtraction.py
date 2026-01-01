import praw
import os
import json
from datetime import datetime


# Reddit API credentials
# Initialize the Reddit API client - using read-only mode for simplicity
REDDIT_CLIENT_ID = 'YOUR_REDDIT_CLIENT_ID'
REDDIT_CLIENT_SECRET = 'YOUR_REDDIT_CLIENT_SECRET'
REDDIT_USER_AGENT = 'your_app_name/0.01 by YOUR_REDDIT_USERNAME'
REDDIT_USERNAME = 'YOUR_REDDIT_USERNAME'
REDDIT_PASSWORD = 'YOUR_REDDIT_PASSWORD'
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# Init Reddit
reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
    user_agent=REDDIT_USER_AGENT,
    username=REDDIT_USERNAME,
    password=REDDIT_PASSWORD
)


def search_subreddits(topic, subreddits=None, limit=10, time_filter='month'):
    """
    Search for posts about a specific topic across one or more subreddits.
    
    Args:
        topic (str): The search query
        subreddits (list): List of subreddit names to search. If None, searches all of Reddit
        limit (int): Maximum number of posts to retrieve per subreddit
        time_filter (str): 'hour', 'day', 'week', 'month', 'year', or 'all'
        
    Returns:
        list: List of post data dictionaries
    """
    results = []
    
    # If no specific subreddits provided, search all of Reddit
    if not subreddits:
        print(f"Searching all of Reddit for: {topic}")
        for post in reddit.subreddit('all').search(topic, sort='relevance', 
                                                 time_filter=time_filter, limit=limit):
            post_data = extract_post_data(post)
            results.append(post_data)
            print(f"Found post: {post.title[:60]}...")
    else:
        # Search each specified subreddit
        for sub_name in subreddits:
            try:
                print(f"Searching r/{sub_name} for: {topic}")
                subreddit = reddit.subreddit(sub_name)
                
                for post in subreddit.search(topic, sort='relevance', 
                                           time_filter=time_filter, limit=limit):
                    post_data = extract_post_data(post)
                    results.append(post_data)
                    print(f"Found post: {post.title[:60]}...")
            except Exception as e:
                print(f"Error searching r/{sub_name}: {str(e)}")
    
    print(f"Retrieved {len(results)} posts total")
    return results


def extract_post_data(post):
    """
    Extract relevant data from a Reddit post and its comments.
    
    Args:
        post: A PRAW post object
        
    Returns:
        dict: Dictionary containing post data and comments
    """
    # Extract basic post information
    post_data = {
        "id": post.id,
        "title": post.title,
        "body": post.selftext,
        "author": str(post.author),
        "score": post.score,
        "upvote_ratio": post.upvote_ratio,
        "url": post.url,
        "created_utc": post.created_utc,
        "created_date": datetime.fromtimestamp(post.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
        "num_comments": post.num_comments,
        "subreddit": post.subreddit.display_name,
        "is_self": post.is_self,  # True if text post, False if link post
        "permalink": f"https://www.reddit.com{post.permalink}",
        "comments": []
    }
    
    # Get comments (limiting to top-level comments for simplicity)
    post.comments.replace_more(limit=0)  # Remove "load more comments" objects
    
    for comment in post.comments[:20]:  # Get top 20 comments
        try:
            comment_data = {
                "id": comment.id,
                "author": str(comment.author),
                "body": comment.body,
                "score": comment.score,
                "created_utc": comment.created_utc,
                "created_date": datetime.fromtimestamp(comment.created_utc).strftime('%Y-%m-%d %H:%M:%S')
            }
            post_data["comments"].append(comment_data)
        except Exception as e:
            print(f"Error processing comment: {str(e)}")
    
    return post_data


def save_to_json(data, filename="reddit_data.json"):
    """Save the Reddit data to a JSON file"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Data saved to {filename}")


# Example usage
if __name__ == "__main__":
    # Define search parameters
    TOPIC = "artificial intelligence"
    SUBREDDITS = ["technology", "MachineLearning", "futurology", "ArtificialIntelligence"]
    POST_LIMIT = 5  # Posts per subreddit
    
    # Search for posts
    results = search_subreddits(TOPIC, SUBREDDITS, limit=POST_LIMIT)
    
    # Save data to file
    save_to_json(results, f"{TOPIC.replace(' ', '_')}_reddit_data.json")
