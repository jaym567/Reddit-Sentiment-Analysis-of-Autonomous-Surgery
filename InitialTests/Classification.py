import os
import re
import asyncio
import praw
from litellm import acompletion
from collections import Counter
import weave; weave.init('reddit_sentiment')


# Reddit creds
REDDIT_CLIENT_ID = 'po9US0Pej05iDzAg4d2ztQ'
REDDIT_CLIENT_SECRET = 'VFdTIqrbDS_Ie1s7UXpaSzNi1TZscA'
REDDIT_USER_AGENT = 'surgery_sentiment/0.01 by CopperScholar'
REDDIT_USERNAME = 'YOUR_USERNAME'
REDDIT_PASSWORD = 'YOUR_PASSWORD'




OPENAI_API_KEY = os.getenv("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")


# Init Reddit
reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
    user_agent=REDDIT_USER_AGENT,
    username=REDDIT_USERNAME,
    password=REDDIT_PASSWORD
)


@weave.op 
async def get_subreddits(query, num=2):
    """Get relevant subreddits for a query"""
    prompt = f"""List {num} relevant subreddits where people would discuss and answer the following question:


{query}


Return only the subreddit names, no hashtags or explanations."""
    res = await acompletion(
        model="gpt-4o-mini",
        api_key=OPENAI_API_KEY,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=200,
    )
    content = res["choices"][0]["message"]["content"]
    return [
        re.sub(r"^\d+[\.\)]\s*", "", line.strip().replace("r/", "").replace("/r/", ""))
        for line in content.splitlines() if line.strip()
    ]


@weave.op 
async def summarize_post(post_title, post_body):
    """Summarize a post into 30-50 words"""
    if len(post_body) < 200:  # If post is already short, no need to summarize
        return post_title + " " + post_body


    prompt = f"""Summarize the following Reddit post in 30-50 words, capturing the main points and sentiment:


Title: {post_title}
Body: {post_body}


Provide ONLY the summary."""
    
    try:
        res = await acompletion(
            model="gpt-4o-mini",
            api_key=OPENAI_API_KEY,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=100,
        )
        
        summary = res["choices"][0]["message"]["content"].strip()
        return summary
    except Exception as e:
        print(f"Error summarizing post: {str(e)}")
        # Fallback: just use the title and first 40 words of body
        words = post_body.split()
        short_body = " ".join(words[:40]) + ("..." if len(words) > 40 else "")
        return post_title + " " + short_body






@weave.op 
async def generate_search_terms(query):
    """Generate effective search terms from the query"""
    prompt = f"""Generate 3 effective Reddit search terms for finding discussions about this question:


{query}


The terms should be effective for Reddit's search function (keywords, not full sentences).
Return only the search terms, one per line, no numbering or explanations."""
    
    res = await acompletion(
        model="gpt-4o-mini",
        api_key=OPENAI_API_KEY,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=100,
    )
    content = res["choices"][0]["message"]["content"]
    return [line.strip() for line in content.splitlines() if line.strip()]


async def assess_post_relevance(post_title, post_body, topic):
    """Determine if a post is relevant to the given topic"""
    prompt = f"""Assess if the following Reddit post is relevant to this topic: "{topic}"


Post title: "{post_title}"
Post body: "{post_body}"


Respond with a number between 0 and 1 indicating relevance:
0 = Not at all relevant to the topic
0.5 = Somewhat relevant
1 = Highly relevant to the topic


Respond with only a number between 0 and 1."""
    
    try:
        res = await acompletion(
            model="gpt-4o-mini",
            api_key=OPENAI_API_KEY,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=10,
        )
        
        score_text = res["choices"][0]["message"]["content"].strip()
        try:
            score = float(score_text)
            return score
        except ValueError:
            # If we can't parse as float, make a rough estimate based on text
            if "1" in score_text:
                return 1.0
            elif "0.5" in score_text:
                return 0.5
            else:
                return 0.0
    except Exception:
        return 0.5  # Default to maybe relevant on error


async def fetch_threads(subs, topic, limit=5, comments_per_post=7):
    """Fetch threads related to the topic, filtering for relevance"""
    results = []
    processed_urls = set()  # Track processed posts to avoid duplicates
    
    # Generate search terms from the topic
    search_terms = await generate_search_terms(topic)
    print(f"Search terms generated: {search_terms}")
    
    # Extract keywords for fallback
    topic_keywords = topic.lower().strip()
    search_terms.append(topic_keywords)
    
    for sub in subs:
        try:
            for term in search_terms:
                print(f"Searching r/{sub} for: {term}")
                
                for post in reddit.subreddit(sub).search(term, limit=limit, sort='relevance'):
                    # Skip if we've already processed this post
                    if post.permalink in processed_urls:
                        continue
                    processed_urls.add(post.permalink)
                    
                    # Check if post is relevant to the topic
                    relevance_score = await assess_post_relevance(post.title, post.selftext, topic)
                    if relevance_score < 0.6:  # Threshold for relevance
                        print(f"Skipping irrelevant post: {post.title} (relevance: {relevance_score:.2f})")
                        continue
                    
                    # Generate a concise summary of the post
                    post_summary = await summarize_post(post.title, post.selftext)
                    
                    post.comments.replace_more(limit=0)
                    comments = []
                    
                    for c in post.comments[:comments_per_post]:
                        comments.append({
                            "text": c.body,
                            "score": c.score
                        })
                        
                    if comments:
                        results.append({
                            "subreddit": sub,
                            "title": post.title,
                            "body": post.selftext,
                            "summary": post_summary,
                            "url": f"https://www.reddit.com{post.permalink}",
                            "post_score": post.score,
                            "comments": comments,
                            "relevance_score": relevance_score
                        })
                        print(f"Added relevant post: {post.title} (relevance: {relevance_score:.2f})")
                        print(f"Summary: {post_summary}\n")
        except Exception as e:
            print(f"Error processing subreddit {sub}: {str(e)}")
            continue
            
    return results




@weave.op 
async def analyze_post_sentiment(post, query):
    """Analyze post sentiment towards a specific topic"""
    # Use the summary instead of the full body if available
    if "summary" in post and post["summary"]:
        combined_text = post["summary"]
    else:
        combined_text = f"Title: {post['title']}\nBody: {post['body']}"
    
    # If text or query is missing, return Neutral
    if not (combined_text and query):
        return "Neutral"
    
    prompt = f"""You are a sentiment classifier analyzing a Reddit post to determine opinions about a specific topic.


Topic: "{query}"
Post: "{combined_text}"


Classify the sentiment ONLY if the post directly discusses the topic:


- "Good": The post clearly expresses positive sentiment about the topic
- "Bad": The post clearly expresses negative sentiment about the topic 
- "Neutral": The post does not clearly discuss the topic, expresses mixed feelings, or the relevance to the topic is unclear


If you're uncertain whether the post is directly relevant to the topic, classify as "Neutral".


Respond with exactly one word - either "Good", "Bad", or "Neutral"."""
    
    try:
        res = await acompletion(
            model="gpt-4o-mini",
            api_key=OPENAI_API_KEY,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=10,
        )
        
        label = res["choices"][0]["message"]["content"].strip()
        # Normalize output to ensure we get one of our three labels
        if "good" in label.lower():
            return "Good"
        elif "bad" in label.lower():
            return "Bad"
        else:
            return "Neutral"
    except Exception:
        return "Neutral"  # Return Neutral on any error






@weave.op 
async def analyze_comment_sentiment(comment, post_context, query):
    """Analyze comment sentiment towards a specific topic with post context"""
    # Use the summary instead of the full post content if available
    if "summary" in post_context and post_context["summary"]:
        post_context_text = post_context["summary"]
    else:
        post_context_text = f"Post title: {post_context['title']}\nPost body: {post_context['body']}"
    
    # If any required parameter is missing, return Neutral
    if not (comment["text"] and post_context_text and query):
        return "Neutral"
    
    prompt = f"""You are a sentiment classifier analyzing a Reddit comment to determine opinions about a specific topic. Dont classify the post, ONLY the comment


Topic: "{query}"
Original Post Context: "{post_context_text}"
The Comment to classify: "{comment["text"]}"


Classify the sentiment ONLY if the comment directly discusses the topic:


- "Good": The comment clearly expresses positive sentiment about the topic
- "Bad": The comment clearly expresses negative sentiment about the topic 
- "Neutral": The comment does not clearly discuss the topic, expresses mixed feelings, or the relevance to the topic is unclear


If you're uncertain whether the comment is directly relevant to the topic, classify as "Neutral".


Respond with exactly one word - either "Good", "Bad", or "Neutral"."""
    
    try:
        res = await acompletion(
            model="gpt-4o-mini",
            api_key=OPENAI_API_KEY,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=10,
        )
        
        label = res["choices"][0]["message"]["content"].strip()
        # Normalize output to ensure we get one of our three labels
        if "good" in label.lower():
            return "Good"
        elif "bad" in label.lower():
            return "Bad"
        else:
            return "Neutral"
    except Exception:
        return "Neutral"  # Return Neutral on any error


async def main():
    query = input("Enter a topic to analyze sentiment on Reddit (e.g., 'ChatGPT for coding'): ")
    print(f"Topic: {query}")
    
    print("Finding relevant subreddits...")
    subreddits = await get_subreddits(query)
    print(f"Subreddits found: {subreddits}")
    
    print("Fetching and filtering relevant threads...")
    threads = await fetch_threads(subreddits, query)
    print(f"Found {len(threads)} relevant threads\n")
    
    if not threads:
        print("No relevant threads found. Try a different topic.")
        return
    
    # Track all sentiment labels
    post_labels = []
    comment_labels = []
    all_labels = []
    
    # Analyze each thread
    for thread in threads:
        print(f"\nAnalyzing: {thread['title']} [Score: {thread['post_score']}, Relevance: {thread.get('relevance_score', 'N/A')}]")
        print(f" → {thread['url']}")
        
        # Analyze post sentiment
        post_sentiment = await analyze_post_sentiment(thread, query)
        post_labels.append(post_sentiment)
        all_labels.append(post_sentiment)
        
        print(f"Post sentiment about topic: [{post_sentiment}]")
        
        # Post summary
        print(f"Post summary: {thread.get('summary', 'N/A')}")
        
        # Analyze comments
        print("\nComments:")
        for c in thread["comments"]:
            comment_sentiment = await analyze_comment_sentiment(c, thread, query)
            comment_labels.append(comment_sentiment)
            all_labels.append(comment_sentiment)
            
            # Comment preview
            text_preview = c["text"][:100] + "..." if len(c["text"]) > 100 else c["text"]
            print(f"[{comment_sentiment}] (score: {c['score']}) {text_preview}")
    
    # Generate sentiment summaries
    post_count = Counter(post_labels)
    comment_count = Counter(comment_labels)
    overall_count = Counter(all_labels)
    
    print("\n" + "="*50)
    print(f"SENTIMENT ANALYSIS FOR: {query}")
    print("="*50)
    
    print("\nPost Sentiment:")
    for k in ["Good", "Neutral", "Bad"]:
        pct = (post_count.get(k, 0) / max(len(post_labels), 1)) * 100
        print(f"  {k}: {post_count.get(k, 0)} ({pct:.1f}%)")
    
    print("\nComment Sentiment:")
    for k in ["Good", "Neutral", "Bad"]:
        pct = (comment_count.get(k, 0) / max(len(comment_labels), 1)) * 100
        print(f"  {k}: {comment_count.get(k, 0)} ({pct:.1f}%)")
    
    print("\nOverall Sentiment:")
    for k in ["Good", "Neutral", "Bad"]:
        pct = (overall_count.get(k, 0) / max(len(all_labels), 1)) * 100
        print(f"  {k}: {overall_count.get(k, 0)} ({pct:.1f}%)")
    
    # Calculate relevant sentiment (excluding neutral)
    relevant_counts = {k: v for k, v in overall_count.items() if k != "Neutral"}
    if relevant_counts:
        dominant_sentiment = max(relevant_counts.items(), key=lambda x: x[1])[0]
        dominant_count = relevant_counts[dominant_sentiment]
        total_relevant = sum(relevant_counts.values())
        confidence = (dominant_count / total_relevant) * 100
        
        print(f"\nOverall sentiment about '{query}': {dominant_sentiment}")
        print(f"Confidence: {confidence:.1f}% (based on {total_relevant} relevant opinions)")
    else:
        print(f"\nNot enough relevant opinions about '{query}' were found.")
    
    # Log to Weights & Biases




if __name__ == "__main__":
    asyncio.run(main())
