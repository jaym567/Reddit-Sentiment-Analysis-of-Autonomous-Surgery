import json
import pandas as pd
import os

INPUT_FILE = r"c:\Users\jaymo\OneDrive\Desktop\ARGOS\Sentiment Analysis of Autonomous Surgery\SentimentCode\filtered_reddit_robotic_surgery_analyzed.json"
OUTPUT_FILE = r"c:\Users\jaymo\OneDrive\Desktop\ARGOS\Sentiment Analysis of Autonomous Surgery\SentimentCode\filtered_reddit_sentiment_flat.csv"

def flatten_analysis_results():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    print(f"Reading {INPUT_FILE}...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    
    for post_group in data:
        for item in post_group['items']:
            sentiment = item.get('sentiment')
            if sentiment is None: sentiment = {}
            
            aspect_sentiment = item.get('aspect_sentiment')
            if aspect_sentiment is None: aspect_sentiment = {}

            rows.append({
                "id": item.get('id'),
                "type": item.get('type'),
                "depth": item.get('depth'),
                "parent_id": item.get('parent_id'),
                "post_id": item.get('post_id'),
                "text": item.get('text', '').replace('\n', ' '),
                # General Sentiment
                "gen_sentiment_label": sentiment.get('label'),
                "gen_sentiment_confidence": sentiment.get('confidence'),
                # Aspect Sentiment
                "aspect_label": aspect_sentiment.get('label'),
                "aspect_confidence": aspect_sentiment.get('confidence'),
                # Relevance Info
                "relevance_state": item.get('relevance_state'),
                "relevance_reason": item.get('relevance_reason')
            })

    df = pd.DataFrame(rows)
    
    # Save to CSV
    try:
        if os.path.exists(OUTPUT_FILE):
            os.remove(OUTPUT_FILE) # Try to remove if exists to avoid locks
        df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
        print(f"✅ Flattened analysis saved to: {OUTPUT_FILE}")
    except Exception as e:
        print(f"❌ Error saving main CSV: {e}")
        # Definitive fallback with timestamp to bypass any lock
        import datetime
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback_name = f"filtered_reddit_sentiment_flat_{ts}.csv"
        df.to_csv(fallback_name, index=False, encoding='utf-8-sig')
        print(f"✅ Saved to timestamped backup: {fallback_name}")
    
    print(f"📊 Total items: {len(df)}")
    
    # Display summary
    print("\n--- General Sentiment Summary ---")
    print(df['gen_sentiment_label'].value_counts())
    
    print("\n--- Aspect Sentiment Summary ---")
    print(df['aspect_label'].value_counts())

if __name__ == "__main__":
    flatten_analysis_results()
