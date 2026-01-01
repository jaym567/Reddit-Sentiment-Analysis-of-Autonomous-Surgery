import os
import pandas as pd
from openai import OpenAI
import time
from dotenv import load_dotenv

load_dotenv()

# --- SETUP ---
# either export your key first (in terminal: export OPENAI_API_KEY="sk-...")
# or paste directly below (not recommended for public sharing)
client = OpenAI(api_key="blank")

# --- LOAD CLEANED DATA ---
df = pd.read_csv("reddit_robotic_surgery_clean.csv")
print(f"Loaded {len(df)} rows for analysis ✅")

# --- FUNCTION TO CLASSIFY SENTIMENT ---
def classify_sentiment(text):
    prompt = f"""
    You are an expert in medical robotics.
    Determine the sentiment of the following Reddit comment about autonomous robotic surgery.
    Respond with only one word: Positive, Negative, or Neutral.

    Comment: \"\"\"{text}\"\"\"
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=5,
        )
        sentiment = response.choices[0].message.content.strip()
        return sentiment
    except Exception as e:
        print("Error:", e)
        return "Error"

# --- RUN SENTIMENT CLASSIFICATION ---
results = []
for i, row in df.iterrows():
    text = row["clean_text"]
    sentiment = classify_sentiment(text)
    results.append(sentiment)

    # print progress every 10 rows
    if (i + 1) % 10 == 0:
        print(f"Processed {i+1}/{len(df)}")

    time.sleep(0.5)  # be kind to the API

df["sentiment"] = results

# --- SAVE RESULTS ---
df.to_csv("reddit_robotic_surgery_sentiment.csv", index=False)
print("\n✅ Sentiment analysis complete! Results saved to reddit_robotic_surgery_sentiment.csv")
